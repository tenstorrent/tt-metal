# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for rms_norm (Phase 0).

Measures, per (shape, dtype, layout):
  * PCC (via tests.ttnn.utils_for_testing.assert_with_pcc)
  * max / mean absolute error, relative RMS error
  * comp_allclose summary (models.common.utility_functions)
  * the got/true RATIO spread — the scale-bug detector: a tight cluster of
    r = actual / expected around a NON-1.0 constant is a uniform scale /
    structural bug (fix the kernel), whereas a broad spread centred on 1.0 is
    ordinary precision noise.  Printed always, because rms_norm's failure mode
    of interest (padding folded into the RMS denominator) is exactly a
    near-uniform scale error that PCC is largely blind to.

The numbers this file prints are the ones recorded in
ttnn/ttnn/operations/rms_norm/verification_report.md.  The asserts are loose
gates (the golden-suite thresholds); the value here is the measurement.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import (
    assert_with_pcc,
    check_with_pcc_without_tensor_printout,
    ulp_distance,
)
from ttnn.operations.rms_norm import rms_norm

# Same per-dtype thresholds as the golden suite (eval/golden_tests/rms_norm/helpers.py).
PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995}
RMS = {ttnn.float32: 0.02, ttnn.bfloat16: 0.04}
TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}

SHAPES = [
    pytest.param((1, 1, 32, 64), id="small_1x1x32x64"),
    pytest.param((1, 1, 128, 512), id="medium_1x1x128x512"),
    pytest.param((2, 1, 64, 4096), id="large_2x1x64x4096_stream"),
    pytest.param((1, 1, 50, 200), id="non_aligned_1x1x50x200"),
]


def torch_rms_norm(x, gamma, epsilon=1e-6):
    xf = x.to(torch.float32)
    rms = torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + epsilon)
    return (xf / rms) * gamma.to(torch.float32).reshape(-1)


def _compute_config():
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    return cfg


def _ratio_spread(actual, expected, eps=1e-8):
    """(median, p5, p95, p95/p5) of actual/expected over finite, non-tiny refs."""
    mask = torch.isfinite(actual) & torch.isfinite(expected) & (expected.abs() > eps)
    if mask.sum() < 8:
        return float("nan"), float("nan"), float("nan"), float("nan")
    r = (actual[mask] / expected[mask]).flatten().to(torch.float32)
    med = torch.median(r).item()
    p5 = torch.quantile(r, 0.05).item()
    p95 = torch.quantile(r, 0.95).item()
    return med, p5, p95, (p95 / p5 if p5 != 0 else float("nan"))


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
def test_rms_norm_precision_baseline(device, shape, dtype, layout):
    torch.manual_seed(0)
    torch_dtype = TORCH_DTYPE[dtype]
    W = shape[-1]

    torch_x = torch.randn(shape, dtype=torch_dtype)
    torch_gamma = torch.randn(W, dtype=torch_dtype)
    expected = torch_rms_norm(torch_x, torch_gamma)

    ttnn_x = ttnn.from_torch(torch_x, dtype=dtype, layout=layout, device=device)
    ttnn_gamma = ttnn.from_torch(
        torch_gamma.reshape(1, 1, 1, W), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    ttnn_out = rms_norm(ttnn_x, gamma=ttnn_gamma, compute_kernel_config=_compute_config())
    actual = ttnn.to_torch(ttnn_out).to(torch.float32)

    err = (actual - expected).abs()
    max_abs = err.max().item()
    mean_abs = err.mean().item()
    rel_rms = (torch.sqrt((err**2).mean()) / expected.std()).item()
    med, p5, p95, spread = _ratio_spread(actual, expected)

    _, pcc_msg = check_with_pcc_without_tensor_printout(expected, actual, pcc=PCC[dtype])
    _, allclose_msg = comp_allclose(expected, actual)
    # ULP measured in the OUTPUT dtype's own grid (both sides cast to it first),
    # so it reports "how many representable steps off" for that dtype.
    ulp = ulp_distance(expected.to(torch_dtype), actual.to(torch_dtype)).to(torch.float32)
    print(
        f"\n[precision] shape={tuple(shape)} dtype={dtype} layout={layout}\n"
        f"            pcc={pcc_msg}\n"
        f"            max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} rel_rms={rel_rms:.6g}\n"
        f"            ulp: max={ulp.max().item():.3f} mean={ulp.mean().item():.3f}\n"
        f"            ratio: median={med:.6f} p5={p5:.6f} p95={p95:.6f} p95/p5={spread:.6f}\n"
        f"            {allclose_msg}"
    )

    # A uniform-scale (structural) bug shows as a TIGHT ratio cluster around a
    # non-1.0 constant.  Catch it explicitly — PCC barely moves for a pure scale.
    if spread == spread and spread < 1.02:  # tight cluster
        assert abs(med - 1.0) < 0.01, (
            f"uniform scale error: got/true ratio clusters at {med:.6f} "
            f"(p5={p5:.6f}, p95={p95:.6f}) — structural bug, not precision"
        )

    assert rel_rms <= RMS[dtype], f"relative RMS {rel_rms:.6g} > {RMS[dtype]}"
    assert_with_pcc(expected, actual, pcc=PCC[dtype])
