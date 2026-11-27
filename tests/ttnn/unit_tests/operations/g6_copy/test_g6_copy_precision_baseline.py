# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for g6_copy over the float32 production shapes.

A copy computes nothing, so the expected baseline is PCC 1, every error 0, 0 ULP and a got/true
ratio of exactly 1. The gate is bit equality (helpers.TOLERANCES[float32] = exact); the metrics
are printed for the verification report and to make any departure readable.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose, comp_pcc, comp_ulp
from ttnn.operations.g6_copy import g6_copy

SHAPES = [
    (1, 1),  # one partial tile
    (1, 12, 128, 128),  # 192 tiles, 1-2 per core
    (8, 12, 128, 128),  # 1536 tiles
    (16, 12, 128, 128),  # 3072 tiles: the perf-focus shape
]


def _to_device(t, device):
    return ttnn.from_torch(
        t,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        preserve_nan_values=True,
    )


def _ratio_spread(actual, expected):
    """Median and p5/p95 of actual / expected over non-zero reference elements: a tight cluster
    at a constant other than 1.0 is a scale bug, a spread around 1.0 is rounding noise."""
    mask = torch.isfinite(expected) & (expected != 0)
    r = (actual[mask].to(torch.float64) / expected[mask].to(torch.float64)).flatten()
    q = torch.quantile(r[: min(r.numel(), 1 << 24)], torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))
    return float(q[1]), float(q[0]), float(q[2])


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_g6_copy_precision_baseline(device, shape):
    torch.manual_seed(0)
    src_torch = torch.randn(shape, dtype=torch.float32)
    src = _to_device(src_torch, device)
    dst = _to_device(torch.randn(shape, dtype=torch.float32), device)
    expected = ttnn.to_torch(src)

    g6_copy(src, dst)
    actual = ttnn.to_torch(dst)

    _, pcc = comp_pcc(expected, actual, pcc=0.999999)
    _, allclose_msg = comp_allclose(expected, actual, rtol=0.0, atol=0.0)
    ulp_ok, ulp_msg = comp_ulp(expected, actual, ulp_threshold=0)
    err = (actual.to(torch.float64) - expected.to(torch.float64)).abs()
    mean_abs = float(err.mean())
    ref_std = float(expected.to(torch.float64).std()) if expected.numel() > 1 else 1.0
    rel_rms = float(err.pow(2).mean().sqrt()) / (ref_std if ref_std > 0 else 1.0)
    ratio_median, ratio_p5, ratio_p95 = _ratio_spread(actual, expected)
    bit_mismatches = int((actual.view(torch.int32) != expected.view(torch.int32)).sum())

    print(
        f"PRECISION {shape}: pcc={pcc:.9f} {allclose_msg} mean_abs={mean_abs:.3e} rel_rms={rel_rms:.3e} "
        f"ulp=({ulp_ok}, {ulp_msg}) ratio median={ratio_median:.9f} p5={ratio_p5:.9f} p95={ratio_p95:.9f} "
        f"bit_mismatches={bit_mismatches}"
    )

    assert bit_mismatches == 0, f"{bit_mismatches} logical elements differ bit for bit"
    assert pcc >= 0.999999
    assert rel_rms == 0.0
    assert ulp_ok, ulp_msg
    assert ratio_median == ratio_p5 == ratio_p95 == 1.0
