# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test for the fused topk_router_gpt operation.

The fused op computes:  matmul + bias → logits → topk(k=4) → softmax
Outputs: (indices_rm, weights_rm) both in ROW_MAJOR format
  - indices_rm: [B, k_padded] uint16 expert indices
  - weights_rm: [B, k_padded] bf16 softmax weights
"""

import pytest
import torch
import torch.nn.functional as F
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc

PCC_THRESHOLD = 0.95

SHAPE2TIME = {
    (32, 2880, 128, 4): 27,
}


def run_fused_op(device, torch_input, torch_weight, torch_bias, B, K, N, k=4):
    """Run the fused op and return the torch result."""
    torch_bias_bcast = torch_bias.expand(B, N).contiguous()
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_bias = ttnn.from_torch(torch_bias_bcast, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    indices_rm, weights_rm = ttnn.experimental.topk_router_gpt(
        tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        k=k,
        num_experts=N,
    )

    indices = ttnn.to_torch(indices_rm)[:B, :k].long()
    weights = ttnn.to_torch(weights_rm)[:B, :k].float()
    return weights, indices


@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW,
            },
            id="dispatch_row",
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "B, K, N, TOP_K",
    SHAPE2TIME.keys(),
)
def test_topk_router_gpt_deterministic(device, B, K, N, TOP_K):
    """Known logits → verify topk indices are correct.

    Use a weight that produces known logits: all zeros except bias = arange.
    Top-4 should be cols 127,126,125,124.
    """
    torch_input = torch.zeros(B, K, dtype=torch.bfloat16)
    torch_weight = torch.zeros(K, N, dtype=torch.bfloat16)
    torch_bias = torch.arange(N, dtype=torch.float32).unsqueeze(0).to(torch.bfloat16)

    weights_tt, indices_tt = run_fused_op(device, torch_input, torch_weight, torch_bias, B, K, N, TOP_K)

    ref_logits = torch_bias.float().expand(B, N)
    ref_vals, ref_idxs = torch.topk(ref_logits, TOP_K, dim=-1)
    ref_weights = F.softmax(ref_vals, dim=-1)

    logger.info(f"  Expected indices row 0: {ref_idxs[0].tolist()}")
    logger.info(f"  TT       indices row 0: {indices_tt[0].tolist()}")

    idx_match = (indices_tt == ref_idxs).all().item()
    _pcc_passed, pcc_val = comp_pcc(ref_weights, weights_tt)
    logger.info(f"  Indices exact match: {idx_match}")
    logger.info(f"  Weight PCC: {pcc_val}")

    assert idx_match, f"Indices mismatch: expected {ref_idxs[0].tolist()}, got {indices_tt[0].tolist()}"
    assert pcc_val >= 0.99, f"Weight PCC {pcc_val} below threshold 0.99"


@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW,
            },
            id="dispatch_row",
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "B, K, N, TOP_K",
    SHAPE2TIME.keys(),
)
@pytest.mark.parametrize("seed", [42, 99], ids=["seed_42", "seed_99"])
def test_topk_router_gpt_random_matmul(device, B, K, N, TOP_K, seed):
    """Random matmul + bias → verify topk + softmax accuracy."""
    torch.manual_seed(seed)
    torch_input = (torch.randn(B, K) * 0.1).to(torch.bfloat16)
    torch_weight = (torch.randn(K, N) * 0.01).to(torch.bfloat16)
    torch_bias = (torch.randn(1, N) * 0.1).to(torch.bfloat16)

    weights_tt, indices_tt = run_fused_op(device, torch_input, torch_weight, torch_bias, B, K, N, TOP_K)

    ref_logits = (torch_input.float() @ torch_weight.float() + torch_bias.float()).to(torch.bfloat16).float()
    ref_vals, ref_idxs = torch.topk(ref_logits, TOP_K, dim=-1)
    ref_weights = F.softmax(ref_vals, dim=-1)

    logger.info(f"  Ref indices row 0: {ref_idxs[0].tolist()}")
    logger.info(f"  TT  indices row 0: {indices_tt[0].tolist()}")

    idx_match_count = (indices_tt == ref_idxs).sum().item()
    total_indices = B * TOP_K
    idx_match_pct = idx_match_count / total_indices * 100
    logger.info(f"  Index match: {idx_match_count}/{total_indices} ({idx_match_pct:.1f}%)")

    _pcc_passed, pcc_val = comp_pcc(ref_weights, weights_tt)
    logger.info(f"  Weight PCC: {pcc_val}")

    # Verify softmax properties
    row_sums = weights_tt.sum(dim=-1)
    all_positive = (weights_tt > 0).all().item()
    logger.info(f"  Weight row sums - min: {row_sums.min():.4f}, max: {row_sums.max():.4f}")
    logger.info(f"  All weights positive: {all_positive}")

    assert idx_match_pct >= 90.0, f"Index match {idx_match_pct:.1f}% below threshold 90%"
    assert pcc_val >= PCC_THRESHOLD, f"Weight PCC {pcc_val} below threshold {PCC_THRESHOLD}"
    assert all_positive, "Some weights are not positive"


@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW,
            },
            id="dispatch_row",
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "B, K, N, TOP_K",
    SHAPE2TIME.keys(),
)
def test_topk_router_gpt_dtype_verification(device, B, K, N, TOP_K):
    """Verify output dtypes are uint16 for indices and bfloat16 for weights."""
    torch.manual_seed(42)
    torch_input = (torch.randn(B, K) * 0.1).to(torch.bfloat16)
    torch_weight = (torch.randn(K, N) * 0.01).to(torch.bfloat16)
    torch_bias = (torch.randn(1, N) * 0.1).to(torch.bfloat16)

    torch_bias_bcast = torch_bias.expand(B, N).contiguous()
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_bias = ttnn.from_torch(torch_bias_bcast, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    indices_rm, weights_rm = ttnn.experimental.topk_router_gpt(
        tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        k=TOP_K,
        num_experts=N,
    )

    logger.info(f"  indices_rm dtype: {indices_rm.dtype}")
    logger.info(f"  weights_rm dtype: {weights_rm.dtype}")

    assert indices_rm.dtype == ttnn.uint16, f"Expected uint16 dtype for indices, got {indices_rm.dtype}"
    assert weights_rm.dtype == ttnn.bfloat16, f"Expected bfloat16 dtype for weights, got {weights_rm.dtype}"
