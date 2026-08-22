# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
ULP-based accuracy characterization for ttnn.sum (issue #51180 nightly recharter;
extends the ``test_mean_ulp.py`` pattern to sum).

Relationship to existing sum tests (do not duplicate; different goal):

- unit-tier ``test_reduction.py`` / ``test_sum.py``: equivalence vs torch via
  PCC / rtol / atol on modest shapes and many ``dim`` / ``keepdim`` combinations.
- ``test_sum_int.py`` (this directory): int32 exactness, extreme values, RM padding.
- ``test_reduction_ops.py`` (this directory): functional corner cases (empty tensors,
  preallocated outputs, dim parity) — not numeric sweeps.
- ``tests/ttnn/unit_tests/gtests/test_reduction.cpp`` (merge gate): one exact smoke
  cell per program factory.

This file is only for **bounded max-ULP** characterization (plus near-zero absolute
tolerance) over accumulation depth; PCC-style tolerances hide sum regressions
(pcc 0.95–0.999 passes results that are hundreds of ULP off).

BF16 golden: torch.sum(bf16_input.float(), dim=...).to(torch.bfloat16)
  - FP32 accumulation on host, result cast to BF16 — the best-practice policy the
    device path should follow with fp32_dest_acc_en=True.

FP32 golden: torch.sum(fp32_input, dim=...)
  - True IEEE 754 float32 accumulation. Unlike ttnn.mean / ttnn.max, ttnn.sum has
    NO fast_and_approximate_mode flag: fp32 sum always reduces on the FPU, which
    accumulates in TF32. The FP32 thresholds below therefore document the tf32
    accumulation error of the only available path, and a deep-accumulation section
    tracks its growth against an FP64 golden.

Distribution note: for zero-mean data the row sum itself is near zero, so most
outputs land in the near-zero absolute-tolerance regime of
``measure_ulp_with_near_zero_atol`` (ULP is meaningless when |expected| ~ 0).
``offset_uniform`` (uniform in [1, 2], all-positive) keeps |sum| ~ 1.5*N so the
ULP metric is meaningful; ``normal`` deliberately exercises the near-zero /
cancellation regime; ``wide_uniform`` mixes magnitudes.

Metrics are logged with loguru at INFO for every parametrized case (pass or fail).
"""

import pytest

pytestmark = pytest.mark.use_module_device

import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import measure_ulp_with_near_zero_atol
from tests.ttnn.nightly.unit_tests.operations.reduction.utility_functions import ttnn_sum


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sum_compute_kernel_config(device, fp32_dest_acc_en: bool):
    """Compute kernel config from device arch."""
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=True,
    )


def _golden_sum_bf16(input_bf16: torch.Tensor, dim, keepdim: bool) -> torch.Tensor:
    """FP32-accumulated sum, result cast back to BF16."""
    return torch.sum(input_bf16.float(), dim=dim, keepdim=keepdim).to(torch.bfloat16)


def _golden_sum_fp32(input_fp32: torch.Tensor, dim, keepdim: bool) -> torch.Tensor:
    """PyTorch IEEE 754 float32 sum; the tile engine accumulates in TF32 (no accurate-mode flag on sum)."""
    return torch.sum(input_fp32, dim=dim, keepdim=keepdim)


def _run_ttnn_sum(
    input_torch: torch.Tensor,
    ttnn_dtype,
    device,
    dim,
    keepdim: bool,
    compute_kernel_config=None,
) -> torch.Tensor:
    """Send tensor to device, run ttnn.sum (twice, via the determinism wrapper), return host torch tensor."""
    tt_input = ttnn.from_torch(input_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.fill_implicit_tile_padding(tt_input, 42)
    tt_out = ttnn_sum(tt_input, dim=dim, keepdim=keepdim, compute_kernel_config=compute_kernel_config)
    return ttnn.to_torch(tt_out)


def _sum_input(distribution: str, shape, seed: int = 42) -> torch.Tensor:
    """FP32 test input. Distributions stress different sum regimes (see module docstring)."""
    torch.manual_seed(seed)
    if distribution == "offset_uniform":  # |sum| ~ 1.5*N: the ULP-meaningful regime
        return torch.empty(shape, dtype=torch.float32).uniform_(1.0, 2.0)
    if distribution == "normal":  # zero-mean: cancellation, near-zero sums (atol regime)
        return torch.randn(shape, dtype=torch.float32)
    if distribution == "wide_uniform":  # mixed magnitudes
        return torch.empty(shape, dtype=torch.float32).uniform_(-1e3, 1e3)
    raise ValueError(f"unknown distribution {distribution}")


# ---------------------------------------------------------------------------
# Test parameters — same shape grid family as test_mean_ulp.py, trimmed:
# keepdim is fixed to True (keepdim does not change the accumulation; the mean
# suite already regression-tracks the keepdim plumbing for this op family).
# ---------------------------------------------------------------------------


def _build_sum_shapes_and_dims():
    """Build (shape, dim, id) cases: N×C grid for W and H reduction, plus HW and odd."""
    out = []

    _N_SIZES = [1, 8]  # small, large batch
    _C_SIZES = [1, 4]  # small, large channel
    _H_SIZES = [32, 512]  # small, large H
    _W_SIZES = [64, 512, 2048]  # small, medium, large W
    _H_FIXED = 32
    _W_FIXED = 64

    for n in _N_SIZES:
        for c in _C_SIZES:
            for w in _W_SIZES:
                out.append(((n, c, _H_FIXED, w), -1, f"W-{n}x{c}x{_H_FIXED}x{w}"))

    for n in _N_SIZES:
        for c in _C_SIZES:
            for h in _H_SIZES:
                out.append(((n, c, h, _W_FIXED), -2, f"H-{n}x{c}x{h}x{_W_FIXED}"))

    # 2D HW-reduction: one small and one large representative shape
    out.append(((1, 1, 64, 128), [-2, -1], "HW-small"))
    out.append(((4, 4, 256, 512), [-2, -1], "HW-large"))

    # One non-tile-aligned shape to catch padding edge cases
    out.append(((1, 1, 37, 41), -1, "W-odd"))

    return out


_SHAPES_AND_DIMS = _build_sum_shapes_and_dims()


# ---------------------------------------------------------------------------
# BF16 tests
# ---------------------------------------------------------------------------

# BF16 max-ULP cap vs FP32-accumulated torch golden (see measure_ulp_with_near_zero_atol).
# fp32_dest_acc_en=True (BF16 inputs → FP32 accumulation → BF16 out): observed peak 2 ULP
# on this grid (BH p100a) — the fp32 accumulator reproduces the golden policy up to the
# final-rounding step order.
# fp32_dest_acc_en=False (BF16 accumulation throughout): error grows with depth; observed
# peak 618 ULP (HW-large normal, 128k-element reduction) — the same peak test_mean_ulp.py
# documents for mean, as expected (same accumulator, no final divide). The gap documents
# that fp32 accumulation is essential for BF16 sum accuracy.
_BF16_ULP_THRESHOLD_FP32_DEST = 8  # 4x headroom over observed peak 2
_BF16_ULP_THRESHOLD_BF16_DEST = 2500  # ~4x headroom over observed peak 618 (matches mean)
_BF16_NEAR_ZERO_ATOL_FRACTION_FP32_DEST = 0.002  # tight; fp32 rounding error is tiny
_BF16_NEAR_ZERO_ATOL_FRACTION_BF16_DEST = 0.40  # loose; BF16 accum on near-zero sums


@pytest.mark.parametrize(
    "shape, dim, desc",
    _SHAPES_AND_DIMS,
    ids=[c[2] for c in _SHAPES_AND_DIMS],
)
@pytest.mark.parametrize("distribution", ["offset_uniform", "normal", "wide_uniform"])
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True], ids=["fp32_acc_off", "fp32_acc_on"])
def test_sum_ulp_bf16(device, shape, dim, desc, distribution, fp32_dest_acc_en):
    """Characterize BF16 sum ULP vs FP32-accumulated Torch golden.

    fp32_dest_acc_en=True reflects the recommended path (BF16 inputs accumulated in FP32).
    fp32_dest_acc_en=False documents the accuracy cost of BF16-only accumulation.
    """
    x = _sum_input(distribution, shape).to(torch.bfloat16)

    golden = _golden_sum_bf16(x, dim=dim, keepdim=True)
    ckc = _make_sum_compute_kernel_config(device, fp32_dest_acc_en)
    actual = _run_ttnn_sum(x, ttnn.bfloat16, device, dim=dim, keepdim=True, compute_kernel_config=ckc)

    ulp_threshold = _BF16_ULP_THRESHOLD_FP32_DEST if fp32_dest_acc_en else _BF16_ULP_THRESHOLD_BF16_DEST
    atol_fraction = (
        _BF16_NEAR_ZERO_ATOL_FRACTION_FP32_DEST if fp32_dest_acc_en else _BF16_NEAR_ZERO_ATOL_FRACTION_BF16_DEST
    )
    passed, max_ulp, max_atol_err, atol_tol, msg, ulp_stats = measure_ulp_with_near_zero_atol(
        golden, actual, ulp_threshold, atol_fraction
    )
    spec = f"{desc} {distribution} shape={shape} dim={dim} fp32_acc={fp32_dest_acc_en}"
    logger.info(
        f"ttnn.sum ULP (BF16) | {spec} | ulp mean={ulp_stats['mean']:.3g} p95={ulp_stats['p95']:.3g} p99={ulp_stats['p99']:.3g} max={max_ulp:.4g}/{ulp_threshold} atol {max_atol_err:.4g}/{atol_tol:.4g} | {'ok' if passed else 'FAIL'}"
    )
    if ulp_stats["worst"]:
        logger.info(f"  worst: {ulp_stats['worst']}")
    if not passed:
        logger.info(f"  {msg}")
    assert passed, f"[BF16 {desc} {distribution} fp32_acc={fp32_dest_acc_en}] {msg}"


# ---------------------------------------------------------------------------
# FP32 tests
# ---------------------------------------------------------------------------

# ttnn.sum has no accurate-SFPU flag (unlike ttnn.mean / ttnn.max): fp32 inputs reduce on
# the FPU, which quantizes operands to TF32 (11-bit mantissa). One tf32 rounding is
# ~2^12 fp32 ULP, so thresholds on this path are inherently in the 1e4–1e6 range — the
# value of this cap is pinning today's error level, not asserting fp32 accuracy.
# Observed peak on this grid: 3.5e5 ULP (BH p100a; wide_uniform, 128k-element HW case,
# where sign cancellation amplifies relative error). A future accurate-mode flag for sum
# (as added to mean) would let this drop to the ~512-ULP class of test_mean_ulp.py.
_FP32_ULP_THRESHOLD = 1_400_000  # ~4x headroom over observed peak 3.5e5
_FP32_NEAR_ZERO_ATOL_FRACTION = 0.00125  # observed peak 2.7% of the previous 0.0125 allowance


@pytest.mark.parametrize(
    "shape, dim, desc",
    _SHAPES_AND_DIMS,
    ids=[c[2] for c in _SHAPES_AND_DIMS],
)
@pytest.mark.parametrize("distribution", ["offset_uniform", "normal", "wide_uniform"])
def test_sum_ulp_fp32(device, shape, dim, desc, distribution):
    """Characterize FP32 sum ULP vs Torch FP32 golden (FPU/tf32 path; fp32_dest_acc_en=True)."""
    x = _sum_input(distribution, shape)

    golden = _golden_sum_fp32(x, dim=dim, keepdim=True)
    ckc = _make_sum_compute_kernel_config(device, fp32_dest_acc_en=True)
    actual = _run_ttnn_sum(x, ttnn.float32, device, dim=dim, keepdim=True, compute_kernel_config=ckc)

    passed, max_ulp, max_atol_err, atol_tol, msg, ulp_stats = measure_ulp_with_near_zero_atol(
        golden, actual, _FP32_ULP_THRESHOLD, _FP32_NEAR_ZERO_ATOL_FRACTION
    )
    spec = f"{desc} {distribution} shape={shape} dim={dim}"
    logger.info(
        f"ttnn.sum ULP (FP32, tf32 FPU path) | {spec} | ulp mean={ulp_stats['mean']:.3g} p95={ulp_stats['p95']:.3g} p99={ulp_stats['p99']:.3g} max={max_ulp:.4g}/{_FP32_ULP_THRESHOLD} atol {max_atol_err:.4g}/{atol_tol:.4g} | {'ok' if passed else 'FAIL'}"
    )
    if ulp_stats["worst"]:
        logger.info(f"  worst: {ulp_stats['worst']}")
    if not passed:
        logger.info(f"  {msg}")
    assert passed, f"[FP32 {desc} {distribution}] {msg}"


# ---------------------------------------------------------------------------
# FP32 deep accumulation vs FP64 golden
# ---------------------------------------------------------------------------

# Accumulation-depth tracking against an order-insensitive FP64 golden. There is no
# accurate-mode escape hatch for sum (unlike mean/max), so these caps pin the tf32 FPU
# path's error growth; a kernel change that widens accumulation error at depth shows up
# here first. Per-case caps are ~4x over the worst observed distribution (BH p100a):
# W512 2.1e4, W8192 4.2e4, H8192 3.1e5 (the single-stage H factory accumulates deepest),
# HW1M 1.3e4 (the two-stage W-then-H split keeps each stage shallow — note it beats H8192
# despite 16x more elements).
_FP32_DEEP_CASES = [
    # (shape, dim, id, ulp_cap)
    ((1, 1, 32, 512), -1, "W512", 84_000),
    ((1, 1, 32, 8192), -1, "W8192", 170_000),
    ((1, 1, 8192, 64), -2, "H8192", 1_250_000),
    ((1, 4, 1024, 1024), [-2, -1], "HW1M", 56_000),
]


@pytest.mark.parametrize(
    "shape, dim, desc, ulp_cap",
    _FP32_DEEP_CASES,
    ids=[c[2] for c in _FP32_DEEP_CASES],
)
@pytest.mark.parametrize("distribution", ["offset_uniform", "wide_uniform"])
def test_sum_fp32_deep_accumulation(device, shape, dim, desc, ulp_cap, distribution):
    """FP32 sum error growth with accumulation depth, vs an FP64 golden (tf32 FPU path)."""
    x = _sum_input(distribution, shape, seed=1234)
    golden = torch.sum(x.to(torch.float64), dim=dim, keepdim=True).to(torch.float32)
    ckc = _make_sum_compute_kernel_config(device, fp32_dest_acc_en=True)
    actual = _run_ttnn_sum(x, ttnn.float32, device, dim=dim, keepdim=True, compute_kernel_config=ckc)

    passed, max_ulp, max_atol_err, atol_tol, msg, ulp_stats = measure_ulp_with_near_zero_atol(
        golden, actual, ulp_cap, _FP32_NEAR_ZERO_ATOL_FRACTION
    )
    logger.info(
        f"ttnn.sum fp32 deep accumulation | {desc} {distribution} | ulp mean={ulp_stats['mean']:.3g} p95={ulp_stats['p95']:.3g} max={max_ulp:.4g}/{ulp_cap} atol {max_atol_err:.4g}/{atol_tol:.4g} | {'ok' if passed else 'FAIL'}"
    )
    if not passed:
        logger.info(f"  {msg}")
    assert passed, f"[FP32-deep {desc} {distribution}] {msg}"


# ---------------------------------------------------------------------------
# scalar= scaling must not add error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scalar", [0.5, 2.0], ids=["scalar_half", "scalar_two"])
def test_sum_scalar_scaling_exact(device, scalar):
    """sum(x, scalar=s) for a power-of-two s must be bit-identical to s * sum(x): the scale
    multiplies the accumulated result and a power of two is exact in both dtypes."""
    torch.manual_seed(7)
    shape, dim = (1, 2, 64, 512), -1
    x = _sum_input("offset_uniform", shape, seed=7)
    ckc = _make_sum_compute_kernel_config(device, fp32_dest_acc_en=True)

    tt_input = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.fill_implicit_tile_padding(tt_input, 42)
    base = ttnn.to_torch(ttnn_sum(tt_input, dim=dim, keepdim=True, compute_kernel_config=ckc))
    scaled = ttnn.to_torch(ttnn_sum(tt_input, dim=dim, keepdim=True, scalar=scalar, compute_kernel_config=ckc))
    assert torch.equal(scaled, base * scalar), "power-of-two scalar= scaling changed the accumulated value"
