# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Small unit test for ttnn.experimental.deepseek_prefill.moe_grouped_topk().

Verifies that the new op produces results matching a PyTorch golden reference
using the recall metric (fraction of correctly selected experts per token).
"""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import (
    assert_gate_output,
    build_padding_config,
    distinct_logits,
    grouped_gate_golden_act,
)


TEST_PARAMS = [(1, 1, 1), (1, 1, 33), (1, 1, 128), (1, 1, 3200)]

TEST_PARAM_IDS = ["minimal", "just_over_one_tile", "four_tiles", "realistic"]

# (n_groups, total_experts, summed_experts_per_group, topk_groups, n_activated_experts, route_scale)
ROUTING_CONFIGS = [
    (8, 256, 2, 4, 8, 0.5),  # DeepSeek grouped routing
    (1, 384, 1, 1, 8, 2.827),  # Kimi single expert group -> collapses to a plain top-k
    (1, 256, 1, 1, 6, 1.5),  # DeepSeek-V4 Flash single-group top-k
    (1, 384, 1, 1, 6, 2.5),  # DeepSeek-V4 Pro single-group top-k
]
ROUTING_CONFIG_IDS = ["deepseek-8g256e", "kimi-1g384e", "dsv4flash-1g256e", "dsv4pro-1g384e"]

# Router affinity activation: sigmoid for DeepSeek-V3/Kimi, sqrtsoftplus for DeepSeek-V4.
SCORE_FUNCS = ["sigmoid", "sqrtsoftplus"]


# Input dtype for logits/bias. The op upcasts internally bf16 input to fp32.
INPUT_DTYPES = [ttnn.float32, ttnn.bfloat16]
INPUT_DTYPE_IDS = ["in_fp32", "in_bf16"]


@pytest.mark.parametrize("input_dtype", INPUT_DTYPES, ids=INPUT_DTYPE_IDS)
@pytest.mark.parametrize("score_func", SCORE_FUNCS)
@pytest.mark.parametrize(
    "n_groups,total_experts,summed_experts_per_group,topk_groups,n_activated_experts,route_scale",
    ROUTING_CONFIGS,
    ids=ROUTING_CONFIG_IDS,
)
@pytest.mark.parametrize("num_batches,batch_size,seq_len", TEST_PARAMS, ids=TEST_PARAM_IDS)
@pytest.mark.parametrize("padded_percent", [0, 50], ids=lambda p: f"pad{p}")
def test_moe_grouped_topk(
    device,
    num_batches,
    batch_size,
    seq_len,
    n_groups,
    total_experts,
    summed_experts_per_group,
    topk_groups,
    n_activated_experts,
    route_scale,
    padded_percent,
    score_func,
    input_dtype,
):
    """Verify moe_grouped_topk matches the PyTorch golden reference using recall and PCC.

    Covers both DeepSeek grouped routing and the single-group (n_groups == 1) path that
    collapses to a plain top-k over a variable expert count (e.g. Kimi's 384 experts).

    ``padded_percent`` right-pads the token sequence via a padding_config: padded rows must be
    routed to the sentinel expert id (== total_experts) while real rows still match the golden.
    Degenerate splits (0 or all padded, e.g. 50% of a single token) fall back to the no-padding
    path so the recall/PCC checks stay meaningful.
    """
    torch.manual_seed(42)

    epsilon = 1e-20

    # For the bf16 input path, generate/quantize inputs to bf16 first so the fp32 golden reference
    # sees the exact same values the device does (the op upcasts bf16 -> fp32 internally).
    is_bf16 = input_dtype == ttnn.bfloat16
    logits_gen_dtype = torch.bfloat16 if is_bf16 else torch.float32

    scores = distinct_logits((num_batches, batch_size, seq_len, total_experts), dtype=logits_gen_dtype).float()
    bias = torch.randn(num_batches, batch_size, seq_len, total_experts, dtype=torch.float32)
    if is_bf16:
        bias = bias.to(torch.bfloat16).float()

    ref_indices, ref_weights = grouped_gate_golden_act(
        scores,
        bias,
        route_scale,
        epsilon,
        n_groups,
        summed_experts_per_group,
        topk_groups,
        n_activated_experts,
        score_func,
    )

    # Right-padding: real tokens occupy the leading rows, padding the trailing rows.
    total_tokens = num_batches * batch_size * seq_len
    num_real = total_tokens - int(total_tokens * padded_percent / 100)
    apply_padding = 0 < num_real < total_tokens
    padding_config = build_padding_config(device, num_real) if apply_padding else None

    ttnn_scores_in = ttnn.from_torch(scores, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_bias_in = ttnn.from_torch(bias, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_weights_out, ttnn_indices_out = ttnn.experimental.deepseek_prefill.moe_grouped_topk(
        ttnn_scores_in,
        ttnn_bias_in,
        n_groups=n_groups,
        summed_experts_per_group=summed_experts_per_group,
        topk_groups=topk_groups,
        n_activated_experts=n_activated_experts,
        route_scale=route_scale,
        epsilon=epsilon,
        score_func=score_func,
        padding_config=padding_config,
    )

    # Trim padding (TILE layout pads to tile boundaries); assert_gate_output flattens/compares.
    tt_weights_torch = ttnn.to_torch(ttnn_weights_out)[:num_batches, :batch_size, :seq_len, :n_activated_experts]
    tt_indices_torch = ttnn.to_torch(ttnn_indices_out)[:num_batches, :batch_size, :seq_len, :n_activated_experts]

    assert_gate_output(
        tt_indices_torch,
        tt_weights_torch,
        ref_indices,
        ref_weights,
        n_activated_experts,
        total_experts,
        num_real,
        apply_padding,
        recall_threshold=0.9,
        pcc_threshold=0.96,
    )
