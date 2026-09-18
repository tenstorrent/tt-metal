# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
ULP-based accuracy characterization for ttnn.std / ttnn.var (issue #51180 nightly
recharter; extends the ``test_mean_ulp.py`` pattern to the Welford ops).

Relationship to existing std/var tests (do not duplicate; different goal):

- ``test_reduction_ops.py`` (this directory): functional coverage of std/var over
  shapes / dims / correction with PCC-style tolerances, plus corner cases.
- ``tests/ttnn/unit_tests/gtests/test_reduction.cpp`` (merge gate): one exact smoke
  cell per Welford program factory (W / H / non-HW dim / multi-dim / fp32).

This file is only for **bounded max-ULP** characterization. The device implements
std/var with Welford's algorithm, whose defining property is robustness to a large
mean offset: a naive two-pass E[x^2] - E[x]^2 at offset 1e4 with sigma=1 cancels
~8 significant digits and produces garbage variance, while Welford's incremental
update stays well-conditioned. The ``offset`` distributions below are exactly that
probe — the thresholds on them are the Welford guarantee this suite pins down.

Golden policy:
- BF16: torch.std/var(bf16_input.float(), ...).to(torch.bfloat16) — FP32
  accumulation on host, result cast to BF16 (matches the fp32_dest_acc_en=True
  device policy, as in test_mean_ulp.py).
- FP32: torch.std/var(fp32_input.double(), ...).to(torch.float32) — an FP64,
  ordering-insensitive reference, because torch's own fp32 std/var carries
  accumulation-ordering noise of the same magnitude as the device's.

Offsets are dtype-aware: BF16 uses offset 8 (ULP(8) = 0.0625, so sigma=1 data is
still resolvable in the input dtype); FP32 uses offset 1e4 (the classic
catastrophic-cancellation regime for two-pass variance).

Metrics are logged with loguru at INFO for every parametrized case (pass or fail).
"""

import pytest

pytestmark = pytest.mark.use_module_device

import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import measure_ulp_with_near_zero_atol
from tests.ttnn.nightly.unit_tests.operations.reduction.utility_functions import ttnn_std, ttnn_var


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TTNN_OPS = {"std": ttnn_std, "var": ttnn_var}
_TORCH_OPS = {"std": torch.std, "var": torch.var}


def _make_compute_kernel_config(device, fp32_dest_acc_en: bool):
    """Compute kernel config from device arch."""
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=True,
    )


def _golden(op: str, x: torch.Tensor, dim, correction: bool, out_dtype: torch.dtype) -> torch.Tensor:
    """High-precision host golden (see module docstring), cast to the output dtype."""
    acc_dtype = torch.float32 if out_dtype == torch.bfloat16 else torch.float64
    return _TORCH_OPS[op](x.to(acc_dtype), dim=dim, keepdim=True, correction=1 if correction else 0).to(out_dtype)


def _run_ttnn(op: str, x: torch.Tensor, ttnn_dtype, device, dim, correction: bool, ckc) -> torch.Tensor:
    """Send tensor to device, run ttnn.std/var (twice, via the determinism wrapper), return torch tensor."""
    tt_input = ttnn.from_torch(x, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.fill_implicit_tile_padding(tt_input, 42)
    tt_out = _TTNN_OPS[op](tt_input, dim=dim, keepdim=True, correction=correction, compute_kernel_config=ckc)
    return ttnn.to_torch(tt_out)


def _std_var_input(distribution: str, shape, offset: float, seed: int = 42) -> torch.Tensor:
    """FP32 test input. Distributions stress different Welford regimes (see module docstring)."""
    torch.manual_seed(seed)
    if distribution == "centered":  # mean ~0, sigma 1: the benign regime
        return torch.randn(shape, dtype=torch.float32)
    if distribution == "offset":  # mean-shift cancellation probe: sigma 1 on a large offset
        return offset + torch.randn(shape, dtype=torch.float32)
    if distribution == "tiny_variance":  # sigma ~3e-3 on the offset: variance << mean^2
        return offset + torch.empty(shape, dtype=torch.float32).uniform_(0.0, 1e-2)
    raise ValueError(f"unknown distribution {distribution}")


# ---------------------------------------------------------------------------
# Shape/dim grid — one row per reduction geometry, sized for nightly budget
# ---------------------------------------------------------------------------

_SHAPES_AND_DIMS = [
    ((1, 1, 32, 512), -1, "W512"),
    ((2, 4, 32, 2048), -1, "W2048"),
    ((1, 1, 512, 64), -2, "H512"),
    ((2, 2, 256, 256), [-2, -1], "HW256"),
    ((1, 1, 37, 41), -1, "W-odd"),  # non-tile-aligned: padding must not enter the stats
]


# ---------------------------------------------------------------------------
# BF16 tests
# ---------------------------------------------------------------------------

# BF16 max-ULP caps vs FP32-accumulated torch golden. Observed peaks on BH p100a are
# strikingly small — 0 ULP (fp32_acc_on) and 1 ULP (fp32_acc_off) across the whole grid,
# including the offset distribution — because Welford's running mean keeps every
# accumulated quantity at O(1) magnitude, so bf16 accumulation barely costs anything
# (contrast test_sum_ulp.py: 618 ULP peak for raw bf16 sum accumulation on the same
# depth class). Caps leave margin over the tiny peaks for cross-arch variance while
# staying far below the naive-two-pass failure class this suite exists to catch.
_BF16_ULP_THRESHOLD_FP32_DEST = 16  # observed peak 0
_BF16_ULP_THRESHOLD_BF16_DEST = 32  # observed peak 1
_BF16_NEAR_ZERO_ATOL_FRACTION_FP32_DEST = 0.002
_BF16_NEAR_ZERO_ATOL_FRACTION_BF16_DEST = 0.40


@pytest.mark.parametrize(
    "shape, dim, desc",
    _SHAPES_AND_DIMS,
    ids=[c[2] for c in _SHAPES_AND_DIMS],
)
@pytest.mark.parametrize("distribution", ["centered", "offset"])
@pytest.mark.parametrize("op", ["std", "var"])
@pytest.mark.parametrize("correction", [True, False], ids=["bessel", "biased"])
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True], ids=["fp32_acc_off", "fp32_acc_on"])
def test_std_var_ulp_bf16(device, shape, dim, desc, distribution, op, correction, fp32_dest_acc_en):
    """Characterize BF16 std/var ULP vs FP32-accumulated Torch golden (offset 8: bf16-resolvable)."""
    x = _std_var_input(distribution, shape, offset=8.0).to(torch.bfloat16)

    golden = _golden(op, x, dim, correction, torch.bfloat16)
    ckc = _make_compute_kernel_config(device, fp32_dest_acc_en)
    actual = _run_ttnn(op, x, ttnn.bfloat16, device, dim, correction, ckc)

    ulp_threshold = _BF16_ULP_THRESHOLD_FP32_DEST if fp32_dest_acc_en else _BF16_ULP_THRESHOLD_BF16_DEST
    atol_fraction = (
        _BF16_NEAR_ZERO_ATOL_FRACTION_FP32_DEST if fp32_dest_acc_en else _BF16_NEAR_ZERO_ATOL_FRACTION_BF16_DEST
    )
    passed, max_ulp, max_atol_err, atol_tol, msg, ulp_stats = measure_ulp_with_near_zero_atol(
        golden, actual, ulp_threshold, atol_fraction
    )
    spec = f"{desc} {distribution} {op} correction={correction} shape={shape} dim={dim} fp32_acc={fp32_dest_acc_en}"
    logger.info(
        f"ttnn.{op} ULP (BF16) | {spec} | ulp mean={ulp_stats['mean']:.3g} p95={ulp_stats['p95']:.3g} p99={ulp_stats['p99']:.3g} max={max_ulp:.4g}/{ulp_threshold} atol {max_atol_err:.4g}/{atol_tol:.4g} | {'ok' if passed else 'FAIL'}"
    )
    if ulp_stats["worst"]:
        logger.info(f"  worst: {ulp_stats['worst']}")
    if not passed:
        logger.info(f"  {msg}")
    assert passed, f"[BF16 {op} {desc} {distribution} correction={correction} fp32_acc={fp32_dest_acc_en}] {msg}"


# ---------------------------------------------------------------------------
# FP32 tests
# ---------------------------------------------------------------------------

# FP32 max-ULP caps vs FP64 torch golden, fp32_dest_acc_en=True, per distribution
# (observed peaks on BH p100a, ~4x headroom):
# - centered: 29 ULP — Welford in fp32 is tight in the benign regime.
# - offset (1e4): 1.4e4 ULP ~ one tf32 rounding of relative error. This IS the Welford
#   guarantee: a two-pass fp32 variance at offset 1e4 cancels ~8 significant digits and
#   returns garbage; Welford holds tf32-level relative accuracy.
# - tiny_variance (sigma ~3e-3 on offset 1e4): 5.5e6 ULP — the documented tf32 floor.
#   The FPU quantizes operands to tf32, whose ULP at 1e4 is 8, so a variation of 1e-2
#   is simply invisible to it (sigma < mean * 2^-11). The cap pins this floor; an
#   accurate-mode (full-fp32 SFPU) flag for std/var, as exists for mean, would collapse it.
_FP32_ULP_THRESHOLDS = {
    "centered": 128,
    "offset": 60_000,
    "tiny_variance": 22_000_000,
}
_FP32_NEAR_ZERO_ATOL_FRACTION = 0.00125


@pytest.mark.parametrize(
    "shape, dim, desc",
    _SHAPES_AND_DIMS,
    ids=[c[2] for c in _SHAPES_AND_DIMS],
)
@pytest.mark.parametrize("distribution", ["centered", "offset", "tiny_variance"])
@pytest.mark.parametrize("op", ["std", "var"])
@pytest.mark.parametrize("correction", [True, False], ids=["bessel", "biased"])
def test_std_var_ulp_fp32(device, shape, dim, desc, distribution, op, correction):
    """Characterize FP32 std/var ULP vs FP64 Torch golden (offset 1e4: two-pass would cancel)."""
    x = _std_var_input(distribution, shape, offset=1.0e4)

    golden = _golden(op, x, dim, correction, torch.float32)
    ckc = _make_compute_kernel_config(device, fp32_dest_acc_en=True)
    actual = _run_ttnn(op, x, ttnn.float32, device, dim, correction, ckc)

    ulp_threshold = _FP32_ULP_THRESHOLDS[distribution]
    passed, max_ulp, max_atol_err, atol_tol, msg, ulp_stats = measure_ulp_with_near_zero_atol(
        golden, actual, ulp_threshold, _FP32_NEAR_ZERO_ATOL_FRACTION
    )
    spec = f"{desc} {distribution} {op} correction={correction} shape={shape} dim={dim}"
    logger.info(
        f"ttnn.{op} ULP (FP32) | {spec} | ulp mean={ulp_stats['mean']:.3g} p95={ulp_stats['p95']:.3g} p99={ulp_stats['p99']:.3g} max={max_ulp:.4g}/{ulp_threshold} atol {max_atol_err:.4g}/{atol_tol:.4g} | {'ok' if passed else 'FAIL'}"
    )
    if ulp_stats["worst"]:
        logger.info(f"  worst: {ulp_stats['worst']}")
    if not passed:
        logger.info(f"  {msg}")
    assert passed, f"[FP32 {op} {desc} {distribution} correction={correction}] {msg}"


# ---------------------------------------------------------------------------
# Zero-variance exactness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["std", "var"])
@pytest.mark.parametrize(
    "ttnn_dtype, torch_dtype", [(ttnn.bfloat16, torch.bfloat16), (ttnn.float32, torch.float32)], ids=["bf16", "fp32"]
)
def test_std_var_constant_input_exact_zero(device, op, ttnn_dtype, torch_dtype):
    """std/var of a constant tensor must be exactly 0.0: Welford's (x - mean) terms are exactly
    zero for constant input in either dtype, so any nonzero output is accumulated noise leaking
    through (or padding entering the stats)."""
    shape, dim = (1, 2, 64, 512), -1
    x = torch.full(shape, 3.140625, dtype=torch_dtype)  # exactly representable in bf16
    ckc = _make_compute_kernel_config(device, fp32_dest_acc_en=True)
    actual = _run_ttnn(op, x, ttnn_dtype, device, dim, correction=True, ckc=ckc)
    assert torch.equal(
        actual, torch.zeros_like(actual)
    ), f"{op} of a constant tensor returned nonzero values (max |out| = {actual.abs().max().item()})"
