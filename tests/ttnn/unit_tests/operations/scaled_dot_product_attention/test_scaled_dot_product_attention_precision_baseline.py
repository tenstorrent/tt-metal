# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for scaled_dot_product_attention (Phase 0).

Measures PCC, max abs error, mean abs error, and relative RMS error across a
small set of supported shapes (bf16, TILE, tile-aligned, MHA self-attention).
This is the numerical reference point the refinement loop measures against —
it is NOT a pass/fail acceptance gate beyond the loose bf16 PCC floor.

The metrics are printed (visible with -s) and the only hard assertion is the
bf16 PCC floor from the shared golden tolerance (0.995). Error growth with
sequence length / head count is the signal of interest: SDPA keeps its running
softmax statistics (m, l) and the running output accumulator O in bf16 circular
buffers (fp32 CBs hang this LLK — Issue #13364), so rounding compounds across
the online-softmax KV recurrence.
"""

from __future__ import annotations

import math

import pytest
import torch

import ttnn

from models.common.utility_functions import comp_allclose, comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

# bf16 PCC floor — matches the shared golden suite (helpers.TOLERANCES[bfloat16]).
BF16_PCC = 0.995


def _pytorch_sdpa(Q, K, V, *, scale=None):
    """fp32 reference: softmax(Q·Kᵀ·scale)·V, returned in input dtype."""
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    D = Qf.shape[-1]
    s = scale if scale is not None else 1.0 / math.sqrt(D)
    scores = torch.matmul(Qf, Kf.transpose(-2, -1)) * s
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, Vf).to(Q.dtype)


def _to_device(t, device):
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# (B, H, S, D): single-tile, multi-tile, medium multi-head, larger multi-head.
PRECISION_SHAPES = [
    (1, 1, 32, 32),
    (1, 1, 128, 64),
    (1, 4, 256, 64),
    (1, 8, 512, 64),
]


@pytest.mark.parametrize("shape", PRECISION_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_precision_baseline(shape, device):
    torch.manual_seed(0)
    B, H, S, D = shape
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16)

    expected = _pytorch_sdpa(Q, K, V).float()

    ttnn_out = scaled_dot_product_attention(_to_device(Q, device), _to_device(K, device), _to_device(V, device))
    actual = ttnn.to_torch(ttnn_out).float()

    assert list(actual.shape) == list(expected.shape)

    abs_err = (actual - expected).abs()
    max_abs = abs_err.max().item()
    mean_abs = abs_err.mean().item()
    rms = abs_err.pow(2).mean().sqrt().item()
    ref_rms = expected.pow(2).mean().sqrt().item()
    rel_rms = rms / ref_rms if ref_rms > 0 else float("nan")
    _, pcc_msg = comp_allclose(expected, actual)
    _, pcc_val = comp_pcc(expected, actual, BF16_PCC)

    print(
        f"\n[precision] shape={shape} pcc={pcc_val} "
        f"max_abs={max_abs:.5f} mean_abs={mean_abs:.5f} rel_rms={rel_rms:.5f} | {pcc_msg}"
    )

    assert_with_pcc(expected, actual, BF16_PCC)
