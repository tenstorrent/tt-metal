# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for qkv_proj_heads_rope (Phase 0 SUPPORTED: bf16, TILE, tile-aligned, L1 placement).

Per output (q, k, v): PCC, max / mean abs error, relative RMS, ULP p99 and the got/true ratio spread
(scale-bug detector: a tight cluster around a non-1.0 constant means a structural bug, not rounding).
Run with -s to see the table.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.qkv_proj_heads_rope import qkv_proj_heads_rope

from .test_qkv_proj_heads_rope import TOLERANCES, device_inputs, make_inputs, reference, relative_rms

PCC_MIN, RMS_MAX = TOLERANCES[ttnn.bfloat16]

# (B, S, S_tab): smallest grid class, a mid shape, the primary production shape, a wide-batch shape.
SHAPES = [(1, 32, 32), (4, 256, 512), (8, 512, 512), (32, 128, 128)]


def bf16_ulp_p99(actual, expected):
    """99th percentile of |actual - expected| in units of the bf16 ULP at |expected|."""
    ref = expected.double().abs().clamp_min(torch.finfo(torch.bfloat16).tiny)
    ulp = torch.pow(2.0, torch.floor(torch.log2(ref)) - 7)
    return torch.quantile(((actual.double() - expected.double()).abs() / ulp).flatten()[::7], 0.99).item()


def ratio_spread(actual, expected):
    """(median, p5, p95) of actual / expected over elements with |expected| above the bf16 noise floor."""
    exp = expected.double()
    mask = exp.abs() > 1e-2 * exp.abs().max()
    ratio = actual.double()[mask] / exp[mask]
    qs = torch.quantile(ratio[::7], torch.tensor([0.5, 0.05, 0.95], dtype=torch.float64))
    return tuple(q.item() for q in qs)


@pytest.mark.parametrize("batch, seq, seq_tab", SHAPES, ids=[f"{b}x{s}_tab{t}" for b, s, t in SHAPES])
def test_qkv_proj_heads_rope_precision_baseline(device, batch, seq, seq_tab):
    x, w, b, cos, sin = make_inputs(batch, seq, seq_tab)
    outputs = qkv_proj_heads_rope(*device_inputs(device, x, w, b, cos, sin))
    expected = reference(x, w, b, cos, sin)

    for name, out, ref in zip("qkv", outputs, expected):
        actual = ttnn.to_torch(out).double()
        _, pcc_msg = assert_with_pcc(ref, actual, PCC_MIN)
        _, allclose_msg = comp_allclose(actual, ref)
        mean_abs = (actual - ref).abs().mean().item()
        rel_rms = relative_rms(actual, ref)
        median, p5, p95 = ratio_spread(actual, ref)
        print(
            f"\n[{batch}x{seq}] {name}: {pcc_msg} | {allclose_msg} | mean_abs={mean_abs:.3e} "
            f"rel_rms={rel_rms:.3e} ulp_p99={bf16_ulp_p99(actual, ref):.2f} "
            f"ratio median={median:.5f} p5={p5:.5f} p95={p95:.5f}"
        )
        assert rel_rms <= RMS_MAX, f"{name}: rel_rms {rel_rms:.3e} > {RMS_MAX}"
        assert abs(median - 1.0) < 1e-3, f"{name}: got/true ratio median {median:.5f} != 1 (scale bug)"
