# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Reference PyTorch implementation of DeepSeek-V3 MoE Gate.
This is copied from models/demos/deepseek_v3/reference/modeling_deepseek.py
for use as a golden reference in tt-moe tests.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


def topk_bitonic(x, k, dim=-1, largest=True, sorted=True):
    """
    Bitonic top-k implementation for deterministic results.
    This is imported from models.demos.deepseek_v3.reference.reference_utils
    """
    if not sorted:
        raise ValueError("topk_bitonic only supports sorted=True")
    if not largest:
        raise ValueError("topk_bitonic only supports largest=True")

    n = x.shape[dim]
    if k > n:
        raise ValueError(f"k ({k}) must be less than or equal to input size ({n}) at dimension {dim}")

    # Get indices for sorting
    indices = torch.arange(n, device=x.device, dtype=torch.long)
    if dim != -1 and dim != x.ndim - 1:
        # Need to reshape for non-last dimension
        perm = list(range(x.ndim))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        x = x.permute(perm)
        indices = indices.view(*([1] * (x.ndim - 1)), n)

    # Bitonic sort - simplified version
    sorted_x, sorted_idx = torch.sort(x, dim=-1, descending=largest, stable=True)
    topk_x = sorted_x[..., :k]
    topk_idx = sorted_idx[..., :k]

    if dim != -1 and dim != x.ndim - 1:
        # Permute back
        perm = list(range(x.ndim))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        topk_x = topk_x.permute(perm)
        topk_idx = topk_idx.permute(perm)

    return topk_x, topk_idx


class MoEGate(nn.Module):
    """
    Reference implementation of the MoE Gate from DeepSeek-V3.

    This module performs hierarchical expert routing using grouped top-k selection
    with sigmoid scoring.
    """

    def __init__(self, config, use_bitonic_sort=True):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.training = False

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(torch.empty((self.n_routed_experts)))
        self.reset_parameters()

        # initialize whether to use bitonic topk or torch.topk
        self.use_bitonic_sort = use_bitonic_sort
        self.topk_fn = torch.topk
        if self.use_bitonic_sort:
            self.topk_fn = topk_bitonic

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Note: The original implementation doesn't initialize e_score_correction_bias
        # This should be initialized to zeros for proper operation
        if hasattr(self, "e_score_correction_bias"):
            init.zeros_(self.e_score_correction_bias)

    def grouped_gate_golden(
        self, scores, bias, route_scale, epsilon, n_groups, summed_experts_per_group, topk_groups, n_activated_experts
    ):
        """Golden reference implementation for debugging."""
        # first run sigmoid on scores
        scores = torch.sigmoid(scores)

        # then add bias (used for selection only)
        biased_scores = scores + bias

        # then reshape based on number of groups
        grouped_scores = biased_scores.reshape(scores.shape[:-1] + (n_groups, scores.shape[-1] // n_groups))

        # then sort the scores within each group
        top_p_experts_scores, _ = torch.topk(grouped_scores, summed_experts_per_group, dim=-1, sorted=True)

        # then sum the scores of the top p experts in each group
        summed_scores = top_p_experts_scores.sum(dim=-1, keepdim=False)
        logger.info(f"summed_scores: {summed_scores}")

        # find the top k groups
        _, top_k_groups_indices = torch.topk(summed_scores, topk_groups, dim=-1, sorted=True)
        logger.info(f"top_k_groups_indices: {top_k_groups_indices}")

        # Create a mask for valid groups
        # We initialize a mask of allowed groups and fill others with -inf
        group_mask = torch.ones(grouped_scores.shape[:-1], dtype=torch.bool, device=scores.device)
        group_mask.scatter_(-1, top_k_groups_indices, False)  # Set selected groups to False (keep)

        # Fill ignored groups with -inf
        masked_grouped_scores = grouped_scores.masked_fill(group_mask.unsqueeze(-1), float("-inf"))

        # reshape back to the original shape
        masked_scores = masked_grouped_scores.reshape(scores.shape)

        # then run topk to find expert indices
        _, top_k_experts_indices = torch.topk(masked_scores, n_activated_experts, dim=-1, sorted=True)

        # then gather the UNBIASED scores (original sigmoid output) based on the top k experts indices
        # The reference uses 'original_scores' (no bias) for the final weights
        chosen_scores = torch.gather(scores, dim=-1, index=top_k_experts_indices)

        # normalize the chosen scores
        normalized_scores = chosen_scores / (chosen_scores.sum(dim=-1, keepdim=True) + epsilon)

        # then scale the normalized scores by the scales
        scaled_scores = normalized_scores * route_scale

        return top_k_experts_indices, scaled_scores

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32), None)
        if self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            raise NotImplementedError(f"insupportable scoring function for MoE gating: {self.scoring_func}")

        ### select top-k experts
        if self.topk_method == "noaux_tc":
            assert not self.training
            scores_for_choice = scores.view(bsz * seq_len, -1) + self.e_score_correction_bias.unsqueeze(0)
            group_scores = self.topk_fn(
                scores_for_choice.view(bsz * seq_len, self.n_group, -1), k=2, dim=-1, sorted=True
            )[0].sum(
                dim=-1
            )  # [n, n_group]

            group_idx = self.topk_fn(group_scores, k=self.topk_group, dim=-1, sorted=True)[1]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group)
                .reshape(bsz * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]
            _, topk_idx = self.topk_fn(tmp_scores, k=self.top_k, dim=-1, sorted=True)
            topk_weight = scores.gather(1, topk_idx)
        else:
            raise NotImplementedError(f"insupportable TopK function for MoE gating: {self.topk_method}")

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor  # must multiply the scaling factor

        return topk_idx, topk_weight

    def grouped_forward(self, hidden_states):
        """Alternative forward implementation using the golden reference."""
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32), None)
        return self.grouped_gate_golden(
            logits,
            self.e_score_correction_bias if hasattr(self, "e_score_correction_bias") else 0,
            self.routed_scaling_factor,
            1e-20,  # epsilon
            self.n_group,
            2,  # summed_experts_per_group
            self.topk_group,
            self.top_k,
        )


# Alias for backward compatibility
ReferenceMoEGate = MoEGate
