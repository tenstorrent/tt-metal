# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for scaled_dot_product_attention (Phase 0, bf16 / TILE).

Measures PCC, max/mean abs error, relative RMS error, and the got/true ratio
spread across 4 shapes (small → larger). The ratio-spread column is the
scale-bug detector: a tight cluster of `r = got/true` around a NON-1.0
constant is a uniform scale/structural bug; a spread centered on ~1.0 is
ordinary bf16 precision noise. Recorded in verification_report.md.

Not a pass/fail gate beyond the per-dtype PCC — this file documents the
measured baseline for the refinement loop.
"""

import math

import pytest
import torch

import ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import comp_allclose


def _reference(q, k, v, scale=None):
    B, H, Sq, D = q.shape
    Hkv = k.shape[1]
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    q, k, v = q.float(), k.float(), v.float()
    if Hkv != H:
        rep = H // Hkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


# (Q_shape, K_shape, V_shape): small single-tile, medium multi-tile,
# multi-head/batch, one larger long-ish self-attention.
BASELINE_SHAPES = {
    "small_32x32": ((1, 1, 32, 32), (1, 1, 32, 32), (1, 1, 32, 32)),
    "medium_128x64": ((1, 1, 128, 64), (1, 1, 128, 64), (1, 1, 128, 64)),
    "multihead_batch": ((2, 4, 256, 64), (2, 4, 256, 64), (2, 4, 256, 64)),
    "larger_1024x64": ((1, 2, 1024, 64), (1, 2, 1024, 64), (1, 2, 1024, 64)),
}


@pytest.mark.parametrize("shape_key", list(BASELINE_SHAPES.keys()))
def test_precision_baseline(device, shape_key):
    q_shape, k_shape, v_shape = BASELINE_SHAPES[shape_key]
    torch.manual_seed(42)
    q = torch.randn(q_shape)
    k = torch.randn(k_shape)
    v = torch.randn(v_shape)

    ref = _reference(q, k, v)

    tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    got = ttnn.to_torch(scaled_dot_product_attention(tq, tk, tv)).float()

    # Error metrics.
    err = (got - ref).abs()
    max_abs = err.max().item()
    mean_abs = err.mean().item()
    rel_rms = (torch.sqrt((got - ref).pow(2).mean()) / torch.sqrt(ref.pow(2).mean())).item()

    # got/true ratio spread (scale-bug detector) over finite non-zero refs.
    ef, gf = ref.flatten(), got.flatten()
    m = ef.abs() > 1e-4
    r = gf[m] / ef[m]
    r_med = r.median().item()
    p5 = r.kthvalue(max(1, int(0.05 * len(r)))).values.item()
    p95 = r.kthvalue(max(1, int(0.95 * len(r)))).values.item()

    _, pcc_msg = comp_allclose(ref, got)
    print(
        f"\n[precision-baseline {shape_key}] shape={q_shape}\n"
        f"  max_abs={max_abs:.5f} mean_abs={mean_abs:.5f} rel_rms={rel_rms:.5f}\n"
        f"  got/true ratio: median={r_med:.5f} p5={p5:.5f} p95={p95:.5f}\n"
        f"  {pcc_msg}"
    )

    # bf16 SUPPORTED gate (mirrors the golden suite bf16+fp32-DEST tolerance).
    assert_with_pcc(ref, got, 0.995)
