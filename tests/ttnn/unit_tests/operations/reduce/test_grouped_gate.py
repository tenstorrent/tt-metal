# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from loguru import logger


def grouped_gate_golden(
    scores, bias, route_scale, epsilon, n_groups, summed_experts_per_group, topk_groups, n_activated_experts
):
    # first run sigmoid on scores
    scores = torch.sigmoid(scores)

    # then add bias (used for selection only)
    biased_scores = scores + bias

    # then reshape based on number of groups
    grouped_scores = biased_scores.reshape(scores.shape[:-1] + (n_groups, scores.shape[-1] // n_groups))

    # then sort the scores within each group
    top_p_experts_scores, _ = torch.topk(grouped_scores, summed_experts_per_group, dim=-1)

    # then sum the scores of the top p experts in each group
    summed_scores = top_p_experts_scores.sum(dim=-1, keepdim=False)
    logger.info(f"summed_scores: {summed_scores}")

    # find the top k groups
    _, top_k_groups_indices = torch.topk(summed_scores, topk_groups, dim=-1)
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
    _, top_k_experts_indices = torch.topk(masked_scores, n_activated_experts, dim=-1)

    # then gather the UNBIASED scores (original sigmoid output) based on the top k experts indices
    # The reference uses 'original_scores' (no bias) for the final weights
    chosen_scores = torch.gather(scores, dim=-1, index=top_k_experts_indices)

    # normalize the chosen scores
    normalized_scores = chosen_scores / (chosen_scores.sum(dim=-1, keepdim=True) + epsilon)

    # then scale the normalized scores by the scales
    scaled_scores = normalized_scores * route_scale

    return scaled_scores, top_k_experts_indices


def test_grouped_gate(device):
    torch.manual_seed(0)
    seq_len = 1
    total_experts = 256
    scores = torch.ones(1, 1, seq_len, total_experts, dtype=torch.bfloat16)
    bias = 2 * torch.ones(1, 1, seq_len, total_experts, dtype=torch.bfloat16)  # no bias for simplicity
    n_groups = 8
    summed_experts_per_group = 2  # number of experts to sum per group
    topk_groups = 4  # top groups to keep
    n_activated_experts = 8  # chosen experts per token
    route_scale = 1.0  # scales for the final weights
    epsilon = 1e-20  # epsilon for stability

    golden_scores, golden_top_k_experts_indices = grouped_gate_golden(
        scores, bias, route_scale, epsilon, n_groups, summed_experts_per_group, topk_groups, n_activated_experts
    )
    logger.info(f"golden_scores: {golden_scores}")
    logger.info(f"golden_top_k_experts_indices: {golden_top_k_experts_indices}")
    logger.info(
        f"golden top k expert indices group number: {golden_top_k_experts_indices // (total_experts // n_groups)}"
    )

    ttnn_scores = ttnn.from_torch(scores, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_bias = ttnn.from_torch(bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_scores, ttnn_top_k_experts_indices = ttnn.grouped_gate(
        ttnn_scores,
        ttnn_bias,
        n_groups,
        summed_experts_per_group,
        topk_groups,
        n_activated_experts,
        route_scale,
        epsilon,
    )

    ttnn_scores = ttnn.to_torch(ttnn_scores)
    ttnn_top_k_experts_indices = ttnn.to_torch(ttnn_top_k_experts_indices)

    logger.info(f"ttnn_scores: {ttnn_scores}")
    logger.info(f"ttnn_top_k_experts_indices: {ttnn_top_k_experts_indices}")
