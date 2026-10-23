# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for rms_norm at the Phase 0 corner (fp32_dest_acc_en=True, HiFi4).

Measures, per shape / dtype / layout:
  PCC, max abs error, mean abs error, relative RMS error (RMS(err) / RMS(ref)), ULP error (p50 / p99 / max, in units
  of the OUTPUT dtype's last place), and the got/true ratio spread (median, p5, p95) — the scale-bug detector: a tight
  cluster of actual/expected around a non-1.0 constant is a uniform scale bug, not rounding noise.

Shapes span both regimes of op_design.md §Regimes:
  R1 row_local (w_group_size == 1)            — (1,1,32,64), (1,1,128,512), (1,1,2048,1000) [W non-aligned, H tall]
  R2 w_split_root_combine (w_group_size >= 2) — (1,1,32,16384)

Run with `-s` to see the table rows (prefixed `PRECISION_BASELINE`).
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.rms_norm import rms_norm

TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}
MANTISSA_BITS = {ttnn.float32: 23, ttnn.bfloat16: 7}
PCC_FLOOR = {ttnn.float32: 0.9999, ttnn.bfloat16: 0.999}
# Max |median(actual/expected) - 1|. Measured Phase 0 bias (verification_report.md §Precision Baseline): fp32 ~-0.14 %,
# bf16 ~-0.08 % — the FPU operand registers hold at most TF32 and TRUNCATE fp32 operands (x, gamma, x*gamma, inv-RMS),
# reproduced by a CPU TF32-truncation emulation (median 0.99897). A structural scale bug (padding in the mean, wrong W,
# a wrong scaler) moves the median by >= 1 %, so 0.5 % separates the two.
RATIO_MEDIAN_TOL = {ttnn.float32: 5e-3, ttnn.bfloat16: 5e-3}

SHAPES = [
    pytest.param((1, 1, 32, 64), id="small_32x64_R1"),
    pytest.param((1, 1, 128, 512), id="medium_128x512_R1"),
    pytest.param((1, 1, 2048, 1000), id="large_2048x1000_Wnonaligned_R1"),
    pytest.param((1, 1, 32, 16384), id="wide_32x16384_R2"),
]


def _ulp_error(actual, expected, dtype):
    """|actual - expected| in units of the output dtype's ULP at |expected| (fp32 math)."""
    e = expected.to(torch.float32)
    a = actual.to(torch.float32)
    mag = e.abs().clamp_min(torch.finfo(torch.float32).tiny)
    exponent = torch.floor(torch.log2(mag))
    ulp = torch.pow(2.0, exponent - MANTISSA_BITS[dtype])
    return ((a - e).abs() / ulp).flatten()


def _ratio_spread(actual, expected):
    e = expected.to(torch.float32).flatten()
    a = actual.to(torch.float32).flatten()
    keep = torch.isfinite(e) & torch.isfinite(a) & (e.abs() > 1e-3 * e.abs().max())
    r = a[keep] / e[keep]
    q = torch.quantile(r[: min(r.numel(), 1 << 24)], torch.tensor([0.05, 0.5, 0.95]))
    return float(q[1]), float(q[0]), float(q[2])


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["TILE", "RM"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("shape", SHAPES)
def test_rms_norm_precision_baseline(device, shape, dtype, layout):
    torch.manual_seed(1234)
    W = shape[-1]
    torch_x = torch.randn(shape, dtype=torch.float32).to(TORCH_DTYPE[dtype])
    torch_gamma = torch.randn((1, 1, 1, W), dtype=torch.float32).to(TORCH_DTYPE[dtype])

    tt_x = ttnn.from_torch(torch_x, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_gamma = ttnn.from_torch(
        torch_gamma, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    actual = ttnn.to_torch(rms_norm(tt_x, gamma=tt_gamma, epsilon=1e-6))

    xf = torch_x.to(torch.float32)
    expected_f32 = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * torch_gamma.to(torch.float32)
    expected = expected_f32.to(TORCH_DTYPE[dtype])  # the best the output dtype can represent

    err = (actual.to(torch.float32) - expected_f32).flatten()
    max_abs = float(err.abs().max())
    mean_abs = float(err.abs().mean())
    rel_rms = float(err.pow(2).mean().sqrt() / expected_f32.pow(2).mean().sqrt())
    ulp = _ulp_error(actual, expected, dtype)
    ulp_p50, ulp_p99, ulp_max = (float(torch.quantile(ulp[: 1 << 24], q)) for q in (0.5, 0.99, 1.0))
    r_med, r_p5, r_p95 = _ratio_spread(actual, expected_f32)

    pcc_passed, pcc_msg = assert_with_pcc(expected_f32, actual.to(torch.float32), PCC_FLOOR[dtype])
    _, allclose_msg = comp_allclose(actual.to(torch.float32), expected_f32)

    print(
        f"\nPRECISION_BASELINE shape={shape} dtype={dtype} layout={layout} pcc={pcc_msg} "
        f"max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} rel_rms={rel_rms:.6g} "
        f"ulp_p50={ulp_p50:.3g} ulp_p99={ulp_p99:.3g} ulp_max={ulp_max:.3g} "
        f"ratio_median={r_med:.6f} ratio_p5={r_p5:.6f} ratio_p95={r_p95:.6f} | {allclose_msg}"
    )

    assert pcc_passed
    assert torch.isfinite(actual).all()
    # Scale-bug guard: a uniform scale error would move the ratio median away from 1.
    assert abs(r_med - 1.0) < RATIO_MEDIAN_TOL[dtype], f"ratio median {r_med} != 1: suspected scale bug"
