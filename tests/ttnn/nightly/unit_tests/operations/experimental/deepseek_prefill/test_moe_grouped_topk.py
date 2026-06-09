# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Small unit test for ttnn.experimental.deepseek_prefill.moe_grouped_topk().

Verifies that the new op produces results matching a PyTorch golden reference
using the recall metric (fraction of correctly selected experts per token).
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.reference.configuration_deepseek import DeepseekV3Config
from models.demos.deepseek_v3.reference.modeling_deepseek import MoEGate
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import calculate_average_recall
from tests.ttnn.utils_for_testing import comp_pcc


def generate_distinct_sigmoid_inputs(shape, min_val=0.05, max_val=0.95, dtype=torch.float32):
    row_size = shape[-1]
    num_rows = torch.tensor(shape[:-1]).prod().item()

    num_candidates = row_size * 4
    candidates = torch.linspace(min_val, max_val, num_candidates, dtype=dtype)
    unique = candidates.unique()

    if unique.numel() < row_size:
        raise ValueError(f"Cannot generate {row_size} distinct sigmoid outputs in [{min_val}, {max_val}].")

    all_rows = []
    for _ in range(num_rows):
        perm = torch.randperm(unique.numel())[:row_size]
        sigmoid_outputs = unique[perm]
        sigmoid_outputs_f32 = sigmoid_outputs.float()
        row_inputs = torch.log(sigmoid_outputs_f32 / (1 - sigmoid_outputs_f32))
        all_rows.append(row_inputs)

    return torch.stack(all_rows).to(dtype).reshape(shape)


TEST_PARAMS = [(1, 1, 1), (1, 1, 33), (1, 1, 128), (1, 1, 3200)]

TEST_PARAM_IDS = ["minimal", "just_over_one_tile", "four_tiles", "realistic"]


@pytest.mark.parametrize("num_batches,batch_size,seq_len", TEST_PARAMS, ids=TEST_PARAM_IDS)
def test_moe_grouped_topk(device, num_batches, batch_size, seq_len):
    """Verify moe_grouped_topk matches the PyTorch golden reference using recall and PCC."""
    torch.manual_seed(42)

    total_experts = 256
    n_groups = 8
    summed_experts_per_group = 2
    topk_groups = 4
    n_activated_experts = 8
    epsilon = 1e-20
    route_scale = 0.5

    config = DeepseekV3Config(
        hidden_size=64,
        n_routed_experts=total_experts,
        n_group=n_groups,
        topk_group=topk_groups,
        num_experts_per_tok=n_activated_experts,
        routed_scaling_factor=route_scale,
    )
    gate = MoEGate(config, use_bitonic_sort=False)

    scores = generate_distinct_sigmoid_inputs((num_batches, batch_size, seq_len, total_experts), dtype=torch.float32)
    bias = torch.randn(num_batches, batch_size, seq_len, total_experts, dtype=torch.float32)

    ref_indices, ref_weights = gate.grouped_gate_golden(
        scores, bias, route_scale, epsilon, n_groups, summed_experts_per_group, topk_groups, n_activated_experts
    )

    ttnn_scores_in = ttnn.from_torch(scores, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_bias_in = ttnn.from_torch(bias, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_weights_out, ttnn_indices_out = ttnn.experimental.deepseek_prefill.moe_grouped_topk(
        ttnn_scores_in,
        ttnn_bias_in,
        n_groups=n_groups,
        summed_experts_per_group=summed_experts_per_group,
        topk_groups=topk_groups,
        n_activated_experts=n_activated_experts,
        route_scale=route_scale,
        epsilon=epsilon,
    )

    tt_weights_torch = ttnn.to_torch(ttnn_weights_out)
    tt_indices_torch = ttnn.to_torch(ttnn_indices_out)

    # Trim padding (TILE layout pads to tile boundaries)
    tt_weights_torch = tt_weights_torch[:num_batches, :batch_size, :seq_len, :n_activated_experts]
    tt_indices_torch = tt_indices_torch[:num_batches, :batch_size, :seq_len, :n_activated_experts]

    # Flatten to 2D [num_tokens, n_activated_experts] for recall calculation
    tt_indices_2d = tt_indices_torch.reshape(-1, n_activated_experts)
    ref_indices_2d = ref_indices.reshape(-1, n_activated_experts)

    recall = calculate_average_recall(tt_indices_2d, ref_indices_2d)
    recall_threshold = 0.9
    recall_passed = recall >= recall_threshold
    status = "PASS" if recall_passed else "FAIL"
    logger.info(
        f"[{status}] Recall = {recall:.4f} (threshold: {recall_threshold}) "
        f"for num_batches={num_batches}, batch_size={batch_size}, seq_len={seq_len}"
    )

    pcc_threshold = 0.97
    weights_passed, weights_pcc = comp_pcc(tt_weights_torch, ref_weights, pcc_threshold)
    status = "PASS" if weights_passed else "FAIL"
    logger.info(
        f"[{status}] Weights PCC = {weights_pcc:.4f} (threshold: {pcc_threshold}) "
        f"for num_batches={num_batches}, batch_size={batch_size}, seq_len={seq_len}"
    )

    assert recall_passed, (
        f"Recall is {recall:.4f}, expected >= {recall_threshold} "
        f"for num_batches={num_batches}, batch_size={batch_size}, seq_len={seq_len}"
    )
    assert weights_passed, (
        f"Weights PCC is {weights_pcc:.4f}, expected >= {pcc_threshold} "
        f"for num_batches={num_batches}, batch_size={batch_size}, seq_len={seq_len}"
    )
