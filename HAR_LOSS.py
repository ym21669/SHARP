import torch
import torch.nn as nn
import torch.nn.functional as F


class HAR_Loss(nn.Module):
    """
    Hard-Aware Reweighted Loss for Contrastive Learning

    Args:
        tau (float): Temperature parameter for similarity scaling
        tau_plus (float): Debiased parameter for positive-negative balance
        beta (float): Scaling factor for negative sample reweighting
        beta2 (float): Scaling factor for positive sample reweighting
    """

    def __init__(self, tau=0.4, tau_plus=0.1, beta=0.7, beta2=0.5):
        super().__init__()
        self.tau = tau
        self.tau_plus = tau_plus
        self.beta = beta
        self.beta2 = beta2

    def normalize(self, x):
        """L2 normalize embeddings"""
        return F.normalize(x, dim=1)

    def compute_similarity(self, x1, x2):
        """Compute cosine similarity matrix"""
        x1 = self.normalize(x1)
        x2 = self.normalize(x2)
        return torch.mm(x1, x2.t())

    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: [batch_size, embedding_dim] - anchor embeddings
            positive: [batch_size, embedding_dim] - positive embeddings
            negative: [batch_size, embedding_dim] - negative embeddings
        """
        batch_size = anchor.size(0)
        device = anchor.device

        # Combine all embeddings for similarity computation
        all_embeddings = torch.cat([anchor, positive, negative], dim=0)

        # Compute similarity matrices
        f = lambda x: torch.exp(x / self.tau)
        sim_matrix = f(self.compute_similarity(all_embeddings, all_embeddings))

        # Create masks for anchor-positive and anchor-negative pairs
        # For triplet: anchor[i] matches positive[i], all others are negatives
        pos_mask = torch.zeros(batch_size * 3, batch_size * 3, device=device)
        neg_mask = torch.zeros(batch_size * 3, batch_size * 3, device=device)

        # Set positive pairs: anchor[i] <-> positive[i]
        for i in range(batch_size):
            pos_mask[i, i + batch_size] = 1  # anchor[i] -> positive[i]
            pos_mask[i + batch_size, i] = 1  # positive[i] -> anchor[i]

        # Add self-similarity (diagonal)
        pos_mask += torch.eye(batch_size * 3, device=device)

        # Set negative pairs: anchor[i] <-> negative[j] for all j
        for i in range(batch_size):
            for j in range(batch_size):
                neg_mask[i, j + 2 * batch_size] = 1  # anchor[i] -> negative[j]
                neg_mask[j + 2 * batch_size, i] = 1  # negative[j] -> anchor[i]

        # Remove diagonal from negative mask
        neg_mask = neg_mask - torch.eye(batch_size * 3, device=device)

        # Compute positive and negative similarities
        pos_sim = sim_matrix * pos_mask
        neg_sim = sim_matrix * neg_mask

        # Positive sample reweighting (give higher weight to harder positives)
        pos_sum = pos_sim.sum(dim=1, keepdim=True)
        max_pos = pos_sum.max()
        pos_weights = max_pos - pos_sum
        pos_weights = pos_weights / (pos_weights.max() - pos_weights.min() + 1e-8)
        pos_weights = self.beta2 * pos_weights + 1.0

        # Apply weights to positive similarities
        weighted_pos_sim = pos_weights * pos_sim

        # Negative sample reweighting (hard negative mining)
        neg_sum = neg_sim.sum(dim=1, keepdim=True)
        max_neg = max(neg_sum.max().abs(), neg_sum.min().abs())
        neg_reweight = 2 * neg_sum / (max_neg + 1e-8)
        neg_reweight = self.beta * neg_reweight / (neg_reweight.mean() + 1e-8)

        # Apply reweighting to negative similarities
        weighted_neg_sim = neg_reweight * neg_sim

        # Compute M_pos and N_neg (average number of positives and negatives)
        M_pos = pos_mask.sum(dim=1).mean()
        N_neg = neg_mask.sum(dim=1).mean()

        # Debiased negative term computation
        pos_term = weighted_pos_sim.sum(dim=1)
        neg_term_1 = (-N_neg / M_pos * self.tau_plus * pos_term +
                      weighted_neg_sim.sum(dim=1)) / (1 - self.tau_plus)
        neg_term_2 = torch.exp(torch.tensor(-1 / self.tau, device=device))
        neg_term = torch.max(neg_term_1, neg_term_2)

        # Final loss computation (only for anchor samples)
        anchor_pos = pos_term[:batch_size]
        anchor_neg = neg_term[:batch_size]

        # HAR loss: -log(pos / (pos + neg))
        loss = -torch.log(anchor_pos / (anchor_pos + anchor_neg + 1e-8))

        return loss.mean()
