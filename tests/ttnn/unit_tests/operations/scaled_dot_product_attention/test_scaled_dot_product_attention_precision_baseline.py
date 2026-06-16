# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for scaled_dot_product_attention (Phase 0).

Measures PCC, max abs error, mean abs error, and relative RMS error for the
bfloat16 / TILE / tile-aligned / MHA Phase-0 surface across a small shape
ladder (single-tile, multi-tile, multi-head, one larger). Recorded in
verification_report.md.

These are baseline measurements, NOT pass/fail gates for refinements — the
golden suite + eval.verify_supported own the support contract. PCC thresholds
here are intentionally loose (SDPA chains matmul→softmax→matmul, so rounding
compounds); the assertions exist only to catch a gross regression.
"""

import math

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc


def _reference_sdpa(Q, K, V, *, scale=None):
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    D = Qf.shape[-1]
    s = scale if scale is not None else 1.0 / math.sqrt(D)
    scores = torch.matmul(Qf, Kf.transpose(-2, -1)) * s
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, Vf)


# (B, H, S, D): single-tile, multi-tile, multi-head, larger.
SHAPES = [
    pytest.param((1, 1, 32, 32), id="single_tile"),
    pytest.param((1, 4, 128, 64), id="multi_tile_multi_head"),
    pytest.param((2, 8, 256, 64), id="batched_multi_head"),
    pytest.param((1, 8, 512, 128), id="larger"),
]


@pytest.mark.parametrize("shape", SHAPES)
def test_precision_baseline(device, shape):
    B, H, S, D = shape
    torch.manual_seed(0)
    Q = torch.randn((B, H, S, D), dtype=torch.bfloat16)
    K = torch.randn((B, H, S, D), dtype=torch.bfloat16)
    V = torch.randn((B, H, S, D), dtype=torch.bfloat16)

    expected = _reference_sdpa(Q, K, V)

    ttnn_Q = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_K = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_V = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out = ttnn.operations.scaled_dot_product_attention.scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V)
    got = ttnn.to_torch(out).float()
    ref = expected.float()

    abs_err = (got - ref).abs()
    max_abs = abs_err.max().item()
    mean_abs = abs_err.mean().item()
    rms = torch.sqrt((abs_err**2).mean()).item()
    rel_rms = rms / (ref.std().item() + 1e-12)

    _, allclose_str = comp_allclose(ref, got, rtol=0.05, atol=0.05)
    print(
        f"\n[precision] shape={shape} max_abs={max_abs:.5f} mean_abs={mean_abs:.5f} "
        f"rel_rms={rel_rms:.5f} | {allclose_str}"
    )

    # Loose gross-regression guard only (see module docstring).
    assert_with_pcc(ref, got, pcc=0.99)
