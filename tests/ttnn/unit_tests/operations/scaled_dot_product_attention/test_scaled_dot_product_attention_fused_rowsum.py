# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""R3e deterministic debug/regression test — the L1-accumulate-during-exp row-sum fusion.

The fused row-sum path (raw-LLK dual-pack: pack exp -> cb_exp AND L1-accumulate
the partial row-sum -> cb_sum_chunk in one DEST window, collapsed once after the
KV loop) is gated to fp32_dest_acc_en=False (the throughput regime). These tests
pin that regime and check exact softmax denominators against hand-calculable
references so any error in the fused sum shows up immediately.

DO NOT DELETE — documents the R3e debugging + acceptance.

Run:
  scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_scaled_dot_product_attention_fused_rowsum.py
"""

from __future__ import annotations

import math

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


# The fused row-sum path is only active in this regime.
LOOSE_CFG = ttnn.ComputeConfigDescriptor(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    fp32_dest_acc_en=False,
    math_approx_mode=False,
)


def _ref(q, k, v, scale=None, mask=None):
    B, H, Sq, D = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if mask is not None:
        scores = scores + mask.float()
    w = torch.softmax(scores, dim=-1)
    return torch.matmul(w, v.float())


def test_fused_all_ones_single_tile(device):
    """All-ones Q,K,V: uniform scores -> uniform softmax (1/Skv) -> output = 1.0.
    The row-sum denominator l = Skv * exp(0) = Skv exactly; if the fused L1-acc
    partial sum or the final column-collapse is wrong, the output != 1.0."""
    q = torch.ones(1, 1, 32, 32, dtype=torch.bfloat16)
    k = torch.ones(1, 1, 32, 32, dtype=torch.bfloat16)
    v = torch.ones(1, 1, 32, 32, dtype=torch.bfloat16)

    tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out = scaled_dot_product_attention(tq, tk, tv, compute_kernel_config=LOOSE_CFG)
    res = ttnn.to_torch(out).float()
    expected = torch.ones(1, 1, 32, 32)
    diff = (res - expected).abs().max()
    assert diff < 0.05, f"max diff {diff}\n{res[0,0,:2,:4]}"


def test_fused_multichunk_ones(device):
    """Multi-KV-chunk all-ones (Skv=512 -> many chunks): the running l must sum the
    partial (alpha-rescaled) chunk sums across the whole KV loop. Output still 1.0."""
    B, H, S, D = 1, 1, 512, 128
    q = torch.ones(B, H, S, D, dtype=torch.bfloat16)
    k = torch.ones(B, H, S, D, dtype=torch.bfloat16)
    v = torch.ones(B, H, S, D, dtype=torch.bfloat16)

    tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out = scaled_dot_product_attention(tq, tk, tv, compute_kernel_config=LOOSE_CFG)
    res = ttnn.to_torch(out).float()
    expected = torch.ones(B, H, S, D)
    diff = (res - expected).abs().max()
    assert diff < 0.05, f"max diff {diff}\n{res[0,0,:2,:4]}"


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 128, 128), (1, 2, 256, 64), (1, 1, 512, 128), (2, 2, 384, 64)],
    ids=["128x128", "256x64", "512x128", "b2h2_384x64"],
)
def test_fused_random(device, shape):
    torch.manual_seed(0)
    B, H, S, D = shape
    q = torch.randn(shape, dtype=torch.bfloat16)
    k = torch.randn(shape, dtype=torch.bfloat16)
    v = torch.randn(shape, dtype=torch.bfloat16)

    tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out = scaled_dot_product_attention(tq, tk, tv, compute_kernel_config=LOOSE_CFG)
    res = ttnn.to_torch(out).float()
    ref = _ref(q, k, v)
    assert_with_pcc(ref, res, pcc=0.99)


@pytest.mark.parametrize("mask_mode", ["none", "custom"])
def test_fused_mask_modes(device, mask_mode):
    """Both regimes (has_mask false/true) through the fused path."""
    torch.manual_seed(2)
    B, H, S, D = 1, 2, 256, 64
    q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, dtype=torch.bfloat16)

    torch_mask = None
    if mask_mode == "custom":
        torch_mask = torch.zeros(B, 1, S, S, dtype=torch.bfloat16)
        torch_mask.masked_fill_(torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1), float("-inf"))

    tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tm = (
        ttnn.from_torch(torch_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        if torch_mask is not None
        else None
    )

    out = scaled_dot_product_attention(tq, tk, tv, attn_mask=tm, compute_kernel_config=LOOSE_CFG)
    res = ttnn.to_torch(out).float()
    ref = _ref(q, k, v, mask=torch_mask)
    assert_with_pcc(ref, res, pcc=0.99)
