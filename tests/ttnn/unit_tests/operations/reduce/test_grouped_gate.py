# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from loguru import logger


def grouped_gate_golden(
    scores, bias, n_group, summed_experts_per_token, top_groups, chosen_experts_per_token, scales, epsilon
):
    # first run sigmoid on scores
    scores = torch.sigmoid(scores)

    # then add bias (used for selection only)
    biased_scores = scores + bias

    # then reshape based on number of groups
    grouped_scores = biased_scores.reshape(scores.shape[:-1] + (n_group, scores.shape[-1] // n_group))

    # then sort the scores within each group
    top_p_experts_scores, _ = torch.topk(grouped_scores, summed_experts_per_token, dim=-1)

    # then sum the scores of the top p experts in each group
    summed_scores = top_p_experts_scores.sum(dim=-1, keepdim=False)

    # find the top k groups
    _, top_k_groups_indices = torch.topk(summed_scores, top_groups, dim=-1)

    # Create a mask for valid groups
    # We initialize a mask of allowed groups and fill others with -inf
    group_mask = torch.ones(grouped_scores.shape[:-1], dtype=torch.bool, device=scores.device)
    group_mask.scatter_(-1, top_k_groups_indices, False)  # Set selected groups to False (keep)

    # Fill ignored groups with -inf
    masked_grouped_scores = grouped_scores.masked_fill(group_mask.unsqueeze(-1), float("-inf"))

    # reshape back to the original shape
    masked_scores = masked_grouped_scores.reshape(scores.shape)

    # then run topk to find expert indices
    _, top_k_experts_indices = torch.topk(masked_scores, chosen_experts_per_token, dim=-1)

    # then gather the UNBIASED scores (original sigmoid output) based on the top k experts indices
    # The reference uses 'original_scores' (no bias) for the final weights
    chosen_scores = torch.gather(scores, dim=-1, index=top_k_experts_indices)

    # normalize the chosen scores
    normalized_scores = chosen_scores / chosen_scores.sum(dim=-1, keepdim=True)

    # Note: The reference implementation does NOT add epsilon.
    # If you need it for stability, keep it, but it deviates from the reference.
    # normalized_scores = normalized_scores + epsilon

    # then scale the normalized scores by the scales
    scaled_scores = normalized_scores * scales

    return scaled_scores, top_k_experts_indices


def test_grouped_gate(device):
    torch.manual_seed(0)
    seq_len = 1
    total_experts = 256
    scores = torch.randn(1, 1, seq_len, total_experts, dtype=torch.bfloat16)
    bias = torch.zeros(1, 1, seq_len, total_experts, dtype=torch.bfloat16)  # no bias for simplicity
    n_group = 8
    summed_experts_per_group = 2  # summed experts per group
    top_groups = 4  # top groups to keep
    chosen_experts_per_token = 8  # chosen experts per token
    scales = torch.randn(1, 1, seq_len, chosen_experts_per_token, dtype=torch.bfloat16)  # scales for the final weights
    epsilon = 1e-5  # epsilon for stability

    golden_scores, golden_top_k_experts_indices = grouped_gate_golden(
        scores, bias, n_group, summed_experts_per_group, top_groups, chosen_experts_per_token, scales, epsilon
    )

    logger.info(f"scores: {scores}")
    logger.info(
        f"scores with grouped: {torch.topk(torch.sort(scores.reshape(1, 1, seq_len, n_group, total_experts // n_group), dim=-1, descending=True)[0][..., 0:summed_experts_per_group].sum(dim=-1), dim=-1, k=top_groups)}"
    )
    logger.info(f"golden_scores: {golden_scores}")
    logger.info(f"golden_top_k_experts_indices: {golden_top_k_experts_indices}")
    logger.info(
        f"golden top k expert indices group number: {golden_top_k_experts_indices // (total_experts // n_group)}"
    )

    # ttnn_scores, ttnn_top_k_experts_indices = ttnn.grouped_gate(scores, bias, n_group, summed_experts_per_token, top_groups, chosen_experts_per_token, scales, epsilon)
    # ttnn_scores = ttnn.to_torch(ttnn_scores)
    # ttnn_top_k_experts_indices = ttnn.to_torch(ttnn_top_k_experts_indices)

    # logger.info(f"ttnn_scores: {ttnn_scores}")
    # logger.info(f"ttnn_top_k_experts_indices: {ttnn_top_k_experts_indices}")
