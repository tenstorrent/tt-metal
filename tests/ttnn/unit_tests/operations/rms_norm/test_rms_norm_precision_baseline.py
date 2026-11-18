# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precision baseline for ttnn.operations.rms_norm (verifier, Phase 0).

Measures, per (shape, dtype, layout) over the Phase 0 corner (fp32_dest_acc_en=True, HiFi4, γ RM):
PCC, max / mean absolute error, relative RMS error (RMS(err) / std(ref)), max / mean ULP (in the
output dtype), and the got/true ratio spread (median, p5, p95) — a tight cluster around a non-1.0
constant would flag a uniform scale bug rather than rounding noise.

Shapes span both regimes: LOCAL (small and tall) and W_SPLIT (one tile-row, W = 8192).
Print with `-s`; the asserts are the recommended tolerances in verification_report.md.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.comparison import ulp

from ttnn.operations.rms_norm import rms_norm

SHAPES = [
    (1, 1, 32, 64),  # LOCAL, single tile-row
    (2, 4, 128, 512),  # LOCAL / W_SPLIT by occupancy, many rows
    (1024, 1024),  # LOCAL, tall
    (1, 1, 32, 8192),  # W_SPLIT, cross-core combine
]

TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}
# Recommended tolerances (verification_report.md): PCC floor and relative-RMS ceiling.
PCC_FLOOR = {ttnn.float32: 0.9999, ttnn.bfloat16: 0.999}
REL_RMS_CEIL = {ttnn.float32: 0.002, ttnn.bfloat16: 0.02}
EPS = 1e-6


def _reference(x, gamma):
    xf = x.to(torch.float32)
    y = xf / torch.sqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + EPS)
    return y * gamma.to(torch.float32).reshape(-1)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_rms_norm_precision_baseline(device, shape, dtype, layout):
    torch.manual_seed(0)
    tdt = TORCH_DTYPE[dtype]
    x = torch.randn(shape, dtype=torch.float32).to(tdt)
    g = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32).to(tdt)
    tx = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tg = ttnn.from_torch(g, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    actual = ttnn.to_torch(rms_norm(tx, gamma=tg, epsilon=EPS)).to(torch.float32)
    expected32 = _reference(x, g)  # fp32 truth
    expected = expected32.to(tdt).to(torch.float32)  # truth rounded to the output dtype

    _, pcc_msg = assert_with_pcc(expected, actual, pcc=PCC_FLOOR[dtype])
    _, allclose_msg = comp_allclose(expected, actual)

    err = (actual - expected32).abs()
    rel_rms = float(torch.sqrt(torch.mean((actual - expected32) ** 2)) / expected32.std())
    ulp_out = ulp(expected.to(tdt)).to(torch.float32)
    ulps = (actual - expected).abs() / ulp_out

    nz = expected32.abs() > 1e-3
    ratio = actual[nz] / expected32[nz]
    q = torch.quantile(ratio[: 1 << 20].to(torch.float64), torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))

    print(
        f"\nPRECISION {shape} {dtype} {layout}: {pcc_msg} | max_abs={err.max():.4g} mean_abs={err.mean():.4g} "
        f"rel_rms={rel_rms:.4g} max_ulp={ulps.max():.1f} mean_ulp={ulps.mean():.3f} | "
        f"ratio p5/median/p95={q[0]:.5f}/{q[1]:.5f}/{q[2]:.5f} | {allclose_msg}"
    )
    assert rel_rms < REL_RMS_CEIL[dtype], f"relative RMS {rel_rms} >= {REL_RMS_CEIL[dtype]}"
    assert abs(float(q[1]) - 1.0) < 1e-2, f"got/true ratio median {float(q[1])} is off 1.0 (scale bug?)"
