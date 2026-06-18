# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for the Flash-Attention scaled_dot_product_attention op.

Measures PCC, max/mean absolute error, and relative RMS error across a small
set of representative shapes (single-tile, multi-tile, multi-head, larger).
This is the Phase-0 numerical fingerprint the refinement loop compares against.

Uses `assert_with_pcc` (tests.ttnn.utils_for_testing) and `comp_allclose`
(models.common.utility_functions) — no hand-rolled metric math. The asserted
threshold is the bf16 golden-suite floor (0.995); the per-shape PCC / error
table is printed for the verification report.
"""

from __future__ import annotations

import math

import pytest
import torch

import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


# bf16 PCC floor (matches the golden suite's per-dtype tolerance).
PCC_THRESHOLD = 0.995


def _reference_sdpa(Q, K, V, *, scale=None):
    """fp32 reference SDPA (no mask, MHA). Q,K,V are torch tensors (B,H,S,D)."""
    Qf, Kf, Vf = Q.to(torch.float32), K.to(torch.float32), V.to(torch.float32)
    D = Qf.shape[-1]
    s = scale if scale is not None else 1.0 / math.sqrt(D)
    scores = torch.matmul(Qf, Kf.transpose(-2, -1)) * s
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, Vf)


# (B, H, S, D) — single-tile, multi-tile, multi-head, larger.
SHAPES = [
    pytest.param((1, 1, 32, 64), id="single_tile"),
    pytest.param((1, 4, 128, 64), id="multi_tile_multi_head"),
    pytest.param((1, 8, 256, 64), id="medium"),
    pytest.param((2, 4, 512, 64), id="larger_batched"),
]


@pytest.mark.parametrize("shape", SHAPES)
def test_sdpa_precision_baseline(device, shape):
    torch.manual_seed(42)
    Q = torch.randn(shape, dtype=torch.bfloat16)
    K = torch.randn(shape, dtype=torch.bfloat16)
    V = torch.randn(shape, dtype=torch.bfloat16)

    expected = _reference_sdpa(Q, K, V)  # fp32

    def to_dev(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    ttnn_out = scaled_dot_product_attention(to_dev(Q), to_dev(K), to_dev(V))
    got = ttnn.to_torch(ttnn_out).to(torch.float32)

    # --- metrics (reported, not all asserted) ---
    abs_err = (got - expected).abs()
    max_abs = abs_err.max().item()
    mean_abs = abs_err.mean().item()
    rel_rms = torch.sqrt(torch.mean((got - expected) ** 2)).item() / (expected.std().item() + 1e-12)

    _, allclose_msg = comp_allclose(expected, got, rtol=0.05, atol=0.05)

    print(
        f"\n[precision-baseline] shape={tuple(shape)} "
        f"max_abs={max_abs:.5f} mean_abs={mean_abs:.5f} rel_rms={rel_rms:.5f} "
        f"| {allclose_msg}"
    )

    assert_with_pcc(expected, got, pcc=PCC_THRESHOLD)
