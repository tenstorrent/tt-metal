# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for rms_norm.

Refinement 1 widened the axes from (shape, dtype, layout) to
(shape, dtype, layout, fp32_dest_acc_en) and added `bfloat8_b` to the dtype
list, plus a separate mixed input-dtype x gamma-dtype matrix.  Same file, same
metrics — one precision characterization per op, per the numeric-formats
contract.

Measures, per (shape, dtype, layout, fp32_dest_acc_en):
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
PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995, ttnn.bfloat8_b: 0.99}
RMS = {ttnn.float32: 0.02, ttnn.bfloat16: 0.04, ttnn.bfloat8_b: 0.10}
# torch has no native block-float, so bf8b is fed (and referenced) from bf16.
TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}

DTYPES = [
    pytest.param(ttnn.bfloat16, id="bf16"),
    pytest.param(ttnn.float32, id="fp32"),
    pytest.param(ttnn.bfloat8_b, id="bfp8"),
]

SHAPES = [
    pytest.param((1, 1, 32, 64), id="small_1x1x32x64"),
    pytest.param((1, 1, 128, 512), id="medium_1x1x128x512"),
    pytest.param((2, 1, 64, 4096), id="large_2x1x64x4096_stream"),
    pytest.param((1, 1, 50, 200), id="non_aligned_1x1x50x200"),
]


def _skip_unsupported(shape, dtype, layout, fp32_dest_acc_en, *, gamma_dtype=None):
    """Refuse the cells the op / the golden feature_spec declare out of universe.

    Mirrors rms_norm.EXCLUSIONS plus feature_spec.INVALID so this file never
    asserts on a cell the op is entitled to reject.
    """
    if dtype == ttnn.float32 and not fp32_dest_acc_en:
        pytest.skip("EXCLUSIONS: float32 activations without fp32 DEST accumulation")
    aligned = shape[-1] % 32 == 0 and shape[-2] % 32 == 0
    for dt in (dtype, gamma_dtype):
        if dt == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT:
            pytest.skip("feature_spec.INVALID: block-float has no ROW_MAJOR realisation")
    if dtype == ttnn.bfloat8_b and not aligned:
        pytest.skip("feature_spec.INVALID: bfloat8_b activations on a non-tile-aligned shape")


def torch_rms_norm(x, gamma, epsilon=1e-6):
    xf = x.to(torch.float32)
    rms = torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + epsilon)
    return (xf / rms) * gamma.to(torch.float32).reshape(-1)


def _compute_config(fp32_dest_acc_en=True, math_fidelity=ttnn.MathFidelity.HiFi4):
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = math_fidelity
    cfg.fp32_dest_acc_en = fp32_dest_acc_en
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


def _measure(device, shape, dtype, layout, fp32_dest_acc_en, gamma_dtype, gamma_layout, tag):
    """Run the op once and assert / print the whole metric set.

    ONE measurement body, shared by both matrices below — the dtype x
    fp32_dest_acc_en sweep and the mixed input/gamma dtype sweep — so a new axis
    never forks the metrics.
    """
    torch.manual_seed(0)
    torch_dtype = TORCH_DTYPE[dtype]
    W = shape[-1]

    torch_x = torch.randn(shape, dtype=torch_dtype)
    torch_gamma = torch.randn(W, dtype=TORCH_DTYPE[gamma_dtype])
    expected = torch_rms_norm(torch_x, torch_gamma)

    ttnn_x = ttnn.from_torch(torch_x, dtype=dtype, layout=layout, device=device)
    ttnn_gamma = ttnn.from_torch(torch_gamma.reshape(1, 1, 1, W), dtype=gamma_dtype, layout=gamma_layout, device=device)

    ttnn_out = rms_norm(
        ttnn_x, gamma=ttnn_gamma, compute_kernel_config=_compute_config(fp32_dest_acc_en=fp32_dest_acc_en)
    )
    actual = ttnn.to_torch(ttnn_out).to(torch.float32)

    # The output dtype (hence its precision floor) is the ACTIVATION dtype; a
    # coarser gamma only perturbs values, it does not coarsen the grid.
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
        f"\n[precision] {tag} shape={tuple(shape)} dtype={dtype} layout={layout}\n"
        f"            fp32_dest_acc_en={fp32_dest_acc_en} gamma_dtype={gamma_dtype} gamma_layout={gamma_layout}\n"
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

    # The looser of the two dtypes sets the floor: a bf8b gamma injects
    # block-float error into a bf16/fp32 activation path.
    pcc_gate = min(PCC[dtype], PCC[gamma_dtype])
    rms_gate = max(RMS[dtype], RMS[gamma_dtype])
    assert rel_rms <= rms_gate, f"relative RMS {rel_rms:.6g} > {rms_gate}"
    assert_with_pcc(expected, actual, pcc=pcc_gate)


# ---------------------------------------------------------------------------
# Matrix 1 — the precision surface: dtype x fp32_dest_acc_en x layout x shape.
# Gamma tracks the activation dtype (the single-precision case).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
@pytest.mark.parametrize("fp32_dest_acc_en", [True, False], ids=["fp32_acc", "bf16_acc"])
def test_rms_norm_precision_baseline(device, shape, dtype, layout, fp32_dest_acc_en):
    _skip_unsupported(shape, dtype, layout, fp32_dest_acc_en)
    # A block-float gamma has no ROW_MAJOR realisation; everything else keeps
    # Phase 0's ROW_MAJOR gamma so those rows stay comparable to the baseline.
    gamma_layout = ttnn.TILE_LAYOUT if dtype == ttnn.bfloat8_b else ttnn.ROW_MAJOR_LAYOUT
    _measure(device, shape, dtype, layout, fp32_dest_acc_en, dtype, gamma_layout, "surface")


# ---------------------------------------------------------------------------
# Matrix 2 — mixed activation dtype x gamma dtype (independent axes).
# Pins the corner op_design.md section 9.2 predicted would fail: a bf8b gamma on
# a non-tile-aligned W.  It does NOT fail — gamma's tile padding is ZERO, and a
# zero never raises a block-float block's shared exponent, so real weights in the
# straddling block are untouched.  Kept as a test so a future change that starts
# poisoning gamma padding is caught here rather than in the golden suite.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 128, 512), id="tile_aligned_1x1x128x512"),
        pytest.param((1, 1, 50, 200), id="non_aligned_1x1x50x200"),
    ],
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("gamma_dtype", DTYPES)
@pytest.mark.parametrize("fp32_dest_acc_en", [True, False], ids=["fp32_acc", "bf16_acc"])
def test_rms_norm_precision_mixed_gamma_dtype(device, shape, dtype, gamma_dtype, fp32_dest_acc_en):
    _skip_unsupported(shape, dtype, ttnn.TILE_LAYOUT, fp32_dest_acc_en, gamma_dtype=gamma_dtype)
    _measure(device, shape, dtype, ttnn.TILE_LAYOUT, fp32_dest_acc_en, gamma_dtype, ttnn.TILE_LAYOUT, "mixed_gamma")
