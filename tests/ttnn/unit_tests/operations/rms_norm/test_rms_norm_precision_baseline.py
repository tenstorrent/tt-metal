# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline + precision MATRIX for rms_norm.

Two tests live here:

  * `test_rms_norm_precision_baseline` — the Phase 0 accuracy record
    (shape x dtype x layout x gamma at the default compute config).
  * `test_rms_norm_precision_matrix` (+ `..._gamma_dtype`) — Refinement 1's
    numerical-configurability surface: dtype x fp32_dest_acc_en x math_fidelity
    x input distribution, and the independent gamma_dtype axis. This is the
    authoritative characterization of the precision axes the op EXPOSES.

The baseline measures, per (shape x dtype x layout x gamma) cell:
  * PCC (Pearson correlation) against an fp32 torch reference,
  * max / mean absolute error,
  * relative RMS error (RMS error / RMS of the reference),
  * the got/true RATIO SPREAD — median r and its p5/p95, where
    r = actual / expected over the finite, non-zero-reference elements.

The ratio spread is the scale-bug detector: a tight cluster of `r` around a
NON-1.0 constant is a uniform scale / structural bug (a mis-scaled reduction, a
padded denominator, a broadcast mistake), which PCC is largely blind to. A broad
spread centred on 1.0 is ordinary rounding noise. Both are printed for every
cell, so a future refinement can tell the two apart without re-deriving them.

The numbers this file prints are recorded in `verification_report.md`.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.rms_norm import rms_norm


EPS = 1e-6

# Baseline gates: loose enough to be a record rather than a tripwire, tight
# enough that a scale bug or a lost fp32 accumulation fails the run.
PCC_GATE = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995}

TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}

SHAPES = [
    pytest.param((1, 1, 32, 32), id="small_1x1x32x32"),
    pytest.param((1, 1, 128, 512), id="medium_1x1x128x512"),
    pytest.param((1, 1, 32, 4096), id="wide_1x1x32x4096"),
    pytest.param((2, 1, 1024, 1024), id="large_2x1x1024x1024"),
    pytest.param((1, 1, 40, 200), id="non_aligned_1x1x40x200"),
]


def _reference(x, gamma):
    x = x.float()
    out = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    if gamma is not None:
        out = out * gamma.float().reshape(-1)
    return out


def _metrics(expected, actual):
    err = (actual - expected).abs()
    ref_rms = expected.pow(2).mean().sqrt().item()
    rel_rms = (err.pow(2).mean().sqrt().item() / ref_rms) if ref_rms > 0 else float("nan")

    # got/true ratio over finite, non-negligible reference elements.
    mask = torch.isfinite(actual) & (expected.abs() > 1e-6 * max(ref_rms, 1e-30))
    if mask.any():
        r = (actual[mask] / expected[mask]).double()
        ratio_median = r.median().item()
        ratio_p5 = torch.quantile(r, 0.05).item()
        ratio_p95 = torch.quantile(r, 0.95).item()
    else:  # pragma: no cover - degenerate reference
        ratio_median = ratio_p5 = ratio_p95 = float("nan")

    return {
        "max_abs": err.max().item(),
        "mean_abs": err.mean().item(),
        "rel_rms": rel_rms,
        "ratio_median": ratio_median,
        "ratio_p5": ratio_p5,
        "ratio_p95": ratio_p95,
    }


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
@pytest.mark.parametrize("with_gamma", [True, False], ids=["gamma", "no_gamma"])
def test_rms_norm_precision_baseline(device, shape, dtype, layout, with_gamma):
    torch.manual_seed(0)
    torch_dtype = TORCH_DTYPE[dtype]
    W = shape[-1]

    torch_input = torch.randn(shape, dtype=torch.float32).to(torch_dtype)
    torch_gamma = (
        torch.randn((1,) * (len(shape) - 1) + (W,), dtype=torch.float32).to(torch_dtype) if with_gamma else None
    )

    expected = _reference(torch_input, torch_gamma)

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device)
    tt_gamma = ttnn.from_torch(torch_gamma, dtype=dtype, layout=layout, device=device) if with_gamma else None

    tt_out = rms_norm(tt_input, gamma=tt_gamma, epsilon=EPS)
    actual = ttnn.to_torch(tt_out).float()

    m = _metrics(expected, actual)
    allclose, allclose_msg = comp_allclose(expected, actual, rtol=0.05, atol=0.05)
    _, pcc_msg = assert_with_pcc(expected, actual, PCC_GATE[dtype])

    print(
        f"\n[precision] shape={tuple(shape)} dtype={dtype} layout={layout} gamma={with_gamma}\n"
        f"            {pcc_msg}\n"
        f"            max_abs={m['max_abs']:.6f} mean_abs={m['mean_abs']:.6f} rel_rms={m['rel_rms']:.6f}\n"
        f"            ratio median={m['ratio_median']:.6f} p5={m['ratio_p5']:.6f} p95={m['ratio_p95']:.6f}\n"
        f"            {allclose_msg} (allclose@rtol=atol=0.05: {allclose})"
    )

    # A uniform, non-1.0 got/true ratio is a scale/structural bug, not rounding:
    # PCC stays high while the whole tensor is mis-scaled. Gate on it explicitly.
    assert abs(m["ratio_median"] - 1.0) < 0.02, (
        f"got/true ratio median {m['ratio_median']:.6f} is not ~1.0 — uniform scale error "
        f"(p5={m['ratio_p5']:.6f}, p95={m['ratio_p95']:.6f})"
    )
    assert_with_pcc(expected, actual, PCC_GATE[dtype])


# ===========================================================================
# Refinement 1 — the precision MATRIX over the numerical-configurability axes
# ===========================================================================
#
# Refinement 1 opened three axis values: fp32_dest_acc_en=False (for bfloat16
# and bfloat8_b), dtype=bfloat8_b, and gamma_dtype=bfloat8_b. This matrix is
# their characterization of record.
#
# BOUNDED AXES, and why. The full cross-product this could span (8 shapes x 3
# dtypes x 2 acc x 4 fidelities x 2 distributions) is ~380 cells and ~190
# distinct JIT kernel builds, because shape/dtype drive the compile-time args
# and acc/fidelity drive the compute config. The axes below are trimmed to the
# values that carry signal:
#   * math_fidelity: HiFi4 (the op default) and HiFi2 (what EVERY perf,
#     resilience and pad_poison golden case pins). HiFi3 sits between them and
#     LoFi is spot-checked by the gamma-dtype matrix below.
#   * shapes: one per kernel path the precision axes interact with — a single
#     tile, a multi-block/multi-core shape, a wide-hidden shape whose Sum x^2
#     accumulates over many tiles (the one the DEST width actually bites on),
#     and the two non-aligned buckets (the mask-before-square path).
# Both distributions are kept: uniform inputs are all-positive, which is the
# monotonically-growing-sum regime a narrowed DEST accumulator degrades on.

# Gates mirror eval/golden_tests/rms_norm/helpers.py TOLERANCES (the external
# benchmark's own per-dtype thresholds). Stated locally rather than imported so
# the unit suite does not depend on the eval harness.
MATRIX_PCC_GATE = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995, ttnn.bfloat8_b: 0.99}

MATRIX_TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
    # No native torch bfloat8_b: the host-side tensor is bf16 and ttnn quantizes
    # on the way to device (same convention as the golden harness).
    ttnn.bfloat8_b: torch.bfloat16,
}

MATRIX_SHAPES = [
    pytest.param((1, 1, 32, 32), id="single_tile"),
    pytest.param((1, 1, 128, 512), id="multi_block"),
    pytest.param((1, 1, 32, 8192), id="wide_hidden"),
    pytest.param((1, 1, 32, 200), id="w_non_aligned"),
    pytest.param((1, 1, 40, 128), id="h_non_aligned"),
]

MATRIX_FIDELITIES = [
    pytest.param(ttnn.MathFidelity.HiFi4, id="HiFi4"),
    pytest.param(ttnn.MathFidelity.HiFi2, id="HiFi2"),
]


def _is_tile_aligned(shape):
    return shape[-1] % 32 == 0 and shape[-2] % 32 == 0


def _compute_config(math_fidelity, fp32_dest_acc_en):
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = math_fidelity
    cfg.fp32_dest_acc_en = fp32_dest_acc_en
    cfg.math_approx_mode = False
    return cfg


def _make_input(shape, torch_dtype, distribution):
    if distribution == "uniform":
        # All-positive: Sum x^2 grows monotonically, the regime a narrowed DEST
        # accumulator is worst on.
        return torch.rand(shape, dtype=torch.float32).to(torch_dtype)
    return torch.randn(shape, dtype=torch.float32).to(torch_dtype)


def _run_and_report(device, shape, dtype, gamma_dtype, cfg, distribution, label):
    """Dispatch one matrix cell, print every metric, gate on PCC + ratio median."""
    torch.manual_seed(0)
    W = shape[-1]
    torch_dtype = MATRIX_TORCH_DTYPE[dtype]

    torch_input = _make_input(shape, torch_dtype, distribution)
    torch_gamma = None
    tt_gamma = None
    if gamma_dtype is not None:
        gamma_torch_dtype = MATRIX_TORCH_DTYPE[gamma_dtype]
        torch_gamma = torch.randn((1,) * (len(shape) - 1) + (W,), dtype=torch.float32).to(gamma_torch_dtype)
        tt_gamma = ttnn.from_torch(torch_gamma, dtype=gamma_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    expected = _reference(torch_input, torch_gamma)

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = rms_norm(tt_input, gamma=tt_gamma, epsilon=EPS, compute_kernel_config=cfg)
    actual = ttnn.to_torch(tt_out).float()

    m = _metrics(expected, actual)
    _, pcc_msg = assert_with_pcc(expected, actual, MATRIX_PCC_GATE[dtype])
    print(
        f"\n[matrix] {label}\n"
        f"         {pcc_msg}\n"
        f"         max_abs={m['max_abs']:.6f} mean_abs={m['mean_abs']:.6f} rel_rms={m['rel_rms']:.6f}\n"
        f"         ratio median={m['ratio_median']:.6f} p5={m['ratio_p5']:.6f} p95={m['ratio_p95']:.6f}"
    )

    # Same scale-bug tripwire as the baseline: a uniform non-1.0 ratio is a
    # structural error (padded denominator, lost mask) that PCC is blind to.
    # Two configuration axes widen the HONEST band, so the band is derived from
    # them rather than fixed — otherwise the tripwire fires on expected hardware
    # behavior and stops reporting real scale bugs:
    #   * bfloat8_b's 16-datum shared exponent quantizes both operands.
    #   * LoFi truncates srcA/srcB to a 5-bit mantissa, and the FPU TRUNCATES
    #     rather than rounds — so each of the (x * rstd) and (* gamma) multiplies
    #     biases low, giving a MEASURED ~3.5% systematic shrink (ratio median
    #     0.965 at bf16, PCC still >= 0.9995). Same mechanism the Phase 0
    #     baseline recorded as a ~0.1% shrink at fp32/HiFi4.
    ratio_band = 0.02
    if dtype == ttnn.bfloat8_b or gamma_dtype == ttnn.bfloat8_b:
        ratio_band = max(ratio_band, 0.05)
    if cfg.math_fidelity == ttnn.MathFidelity.LoFi:
        ratio_band = max(ratio_band, 0.08)
    assert abs(m["ratio_median"] - 1.0) < ratio_band, (
        f"got/true ratio median {m['ratio_median']:.6f} is not ~1.0 — uniform scale error "
        f"(p5={m['ratio_p5']:.6f}, p95={m['ratio_p95']:.6f})"
    )
    assert_with_pcc(expected, actual, MATRIX_PCC_GATE[dtype])


@pytest.mark.parametrize("shape", MATRIX_SHAPES)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b], ids=["bf16", "fp32", "bfp8"])
@pytest.mark.parametrize("fp32_dest_acc_en", [True, False], ids=["fp32_acc", "bf16_acc"])
@pytest.mark.parametrize("math_fidelity", MATRIX_FIDELITIES)
@pytest.mark.parametrize("distribution", ["normal", "uniform"])
def test_rms_norm_precision_matrix(device, shape, dtype, fp32_dest_acc_en, math_fidelity, distribution):
    # {float32, fp32_dest_acc_en=False} is an op-side EXCLUSION: fp32 activations
    # through a 16-bit DEST is a precision configuration the op refuses.
    if dtype == ttnn.float32 and not fp32_dest_acc_en:
        pytest.skip("EXCLUSIONS: {dtype: float32, fp32_dest_acc_en: False}")
    # bfloat8_b on a non-tile-aligned shape: block-float quantizes in 16-datum
    # groups, so the masked/padded tail is out of scope (parked in the golden
    # suite's feature_spec.INVALID, hence skipped rather than xfailed).
    if dtype == ttnn.bfloat8_b and not _is_tile_aligned(shape):
        pytest.skip("bfloat8_b x non-tile-aligned is structurally out of scope (feature_spec.INVALID)")

    _run_and_report(
        device,
        shape,
        dtype,
        gamma_dtype=dtype,
        cfg=_compute_config(math_fidelity, fp32_dest_acc_en),
        distribution=distribution,
        label=(
            f"shape={tuple(shape)} dtype={dtype} acc={fp32_dest_acc_en} "
            f"fidelity={math_fidelity} dist={distribution}"
        ),
    )


# gamma_dtype is INDEPENDENT of the activation dtype (mixed-precision LLMs run
# bf16 activations against fp32 or bfloat8_b weights), so it gets its own sweep
# rather than riding the matrix diagonal above. LoFi is spot-checked here.
@pytest.mark.parametrize("shape", [pytest.param((1, 1, 128, 512), id="multi_block")])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bfp8"])
@pytest.mark.parametrize(
    "gamma_dtype",
    [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b, None],
    ids=["gamma_fp32", "gamma_bf16", "gamma_bfp8", "no_gamma"],
)
@pytest.mark.parametrize(
    "math_fidelity",
    [ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.LoFi],
    ids=["HiFi4", "LoFi"],
)
def test_rms_norm_precision_matrix_gamma_dtype(device, shape, dtype, gamma_dtype, math_fidelity):
    # Pinned to fp32_dest_acc_en=False — the setting every perf / resilience /
    # pad_poison golden case runs at, so this sweep exercises the narrowed DEST.
    _run_and_report(
        device,
        shape,
        dtype,
        gamma_dtype=gamma_dtype,
        cfg=_compute_config(math_fidelity, fp32_dest_acc_en=False),
        distribution="normal",
        label=f"shape={tuple(shape)} dtype={dtype} gamma_dtype={gamma_dtype} fidelity={math_fidelity}",
    )
