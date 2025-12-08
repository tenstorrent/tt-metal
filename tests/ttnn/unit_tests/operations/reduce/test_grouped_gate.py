# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from loguru import logger

# Import from local reference files
from models.demos.deepseek_v3.reference.configuration_deepseek import DeepseekV3Config
from models.demos.deepseek_v3.reference.modeling_deepseek import MoEGate


def grouped_gate_golden(
    scores, bias, route_scale, epsilon, n_groups, summed_experts_per_group, topk_groups, n_activated_experts
):
    # first run sigmoid on scores
    scores = torch.sigmoid(scores)
    logger.info(f"scores: {scores}")

    # then add bias (used for selection only)
    biased_scores = scores + bias
    # index3 = 3
    # logger.info(f"group 3 scores: {biased_scores[:, :, :, index3*32:(index3+1)*32]}")
    # index6 = 6
    # logger.info(f"group 6 scores: {biased_scores[:, :, :, index6*32:(index6+1)*32]}")

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
    logger.info(f"chosen_scores: {chosen_scores}")
    normalized_scores = chosen_scores / (chosen_scores.sum(dim=-1, keepdim=True) + epsilon)

    # then scale the normalized scores by the scales
    scaled_scores = normalized_scores * route_scale

    return scaled_scores, top_k_experts_indices


def test_grouped_gate_against_reference():
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)

    hf_config = DeepseekV3Config()
    moe_gate = MoEGate(hf_config, use_bitonic_sort=False).eval()
    # set the e_score_correction_bias to a random values
    moe_gate.e_score_correction_bias.data = torch.randn(
        moe_gate.e_score_correction_bias.data.shape, dtype=torch.bfloat16
    )

    seq_lens = [32, 128, 2048]
    batch_size = 1

    for seq_len in seq_lens:
        torch_input = torch.randn(batch_size, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)

        reference_topk_indices, reference_scores = moe_gate.forward(torch_input)
        grouped_fn_topk_indices, grouped_fn_scores = moe_gate.grouped_forward(torch_input)

        assert torch.allclose(reference_topk_indices, grouped_fn_topk_indices)
        assert torch.allclose(reference_scores, grouped_fn_scores)


def test_grouped_gate(device):
    torch.manual_seed(0)
    batch_size = 1
    num_batches = 1
    seq_len = 2
    total_experts = 256
    # scores = torch.randint(-3, 3, (1, 1, seq_len, total_experts), dtype=torch.bfloat16)
    scores = torch.randn(num_batches, batch_size, seq_len, total_experts, dtype=torch.bfloat16)
    bias = 2 * torch.ones(
        num_batches, batch_size, seq_len, total_experts, dtype=torch.bfloat16
    )  # no bias for simplicity

    n_groups = 8
    summed_experts_per_group = 2  # number of experts to sum per group
    topk_groups = 4  # top groups to keep
    n_activated_experts = 8  # chosen experts per token
    route_scale = 1.5  # scales for the final weights
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
