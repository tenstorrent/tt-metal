# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
ULP-based accuracy characterization for ttnn.mean (issue #33740).

Relationship to existing mean tests (do not duplicate; different goal):

- ``test_reduction.py`` (e.g. ``test_mean_4d_tensor_dims``, ``test_mean_3d_tensor_dims``,
  ``test_mean_2d_tensor_dims``): equivalence vs torch via PCC / rtol / atol on modest
  shapes and many ``dim`` / ``keepdim`` combinations.
- ``test_reduction_mean.py``: batched 3D means (``dim`` -1 / -2) with tile padding
  stress, ``scalar=`` scaling, and sharded DRAM/block configs—still PCC-style metrics.
- ``tests/ttnn/nightly/unit_tests/operations/reduction/test_reduction_ops.py``:
  reduction corner cases (empty tensors, preallocated outputs, etc.) with ``op`` including
  ``mean``—not large numeric sweeps.

This file is only for **bounded max-ULP** characterization (plus near-zero absolute
tolerance) and **wide shape sweeps** to track regression in reduction length; it does
not replace functional, sharding, or corner-case coverage above.

BF16 golden: torch.mean(bf16_input.float(), dim=...).to(torch.bfloat16)
  - Accumulates in FP32 on the host, then casts the result to BF16.
  - This matches the "best practice" (FP32 accumulation) that the device
    path should also follow.

FP32 golden: torch.mean(fp32_input, dim=...)
  - PyTorch uses true IEEE 754 float32 accumulation.
  - Unless the API uses SFPU-based true float32 accumulation kernels, the
    tile engine accumulates in TF32, so accuracy may be lower / ULP higher.

ULP is measured in the output dtype (BF16 or FP32).  Elements where
|golden| is very small relative to the tensor's dynamic range are excluded
from ULP (where the metric breaks down due to division by a tiny ULP
quantum) and validated with a scaled absolute-tolerance check instead.

Metrics are logged with loguru at INFO for every parametrized case (pass or fail),
consistent with other tests under ``tests/ttnn`` (default sink: stderr).
"""

import pytest

pytestmark = pytest.mark.use_module_device

import torch
from loguru import logger

import ttnn
from tests.ttnn.unit_tests.operations.test_utils import get_compute_kernel_options
from tests.ttnn.utils_for_testing import measure_ulp_with_near_zero_atol


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mean_compute_kernel_config(device, fp32_dest_acc_en: bool):
    """Arch-aware compute kernel config (same pattern as test_softmax_ulp)."""
    try:
        return ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=True,
        )
    except TypeError:
        return get_compute_kernel_options(fp32_dest_acc_en)


def _golden_mean_bf16(input_bf16: torch.Tensor, dim, keepdim: bool) -> torch.Tensor:
    """FP32-accumulated mean, result cast back to BF16."""
    return torch.mean(input_bf16.float(), dim=dim, keepdim=keepdim).to(torch.bfloat16)


def _golden_mean_fp32(input_fp32: torch.Tensor, dim, keepdim: bool) -> torch.Tensor:
    """PyTorch IEEE 754 float32 mean; tile engine accumulates in TF32 unless SFPU kernels are used."""
    return torch.mean(input_fp32, dim=dim, keepdim=keepdim)


def _run_ttnn_mean(
    input_torch: torch.Tensor,
    ttnn_dtype,
    device,
    dim,
    keepdim: bool,
    compute_kernel_config=None,
) -> torch.Tensor:
    """Send tensor to device, run ttnn.mean, return host torch tensor."""
    tt_input = ttnn.from_torch(input_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.fill_implicit_tile_padding(tt_input, 42)
    tt_out = ttnn.mean(tt_input, dim=dim, keepdim=keepdim, compute_kernel_config=compute_kernel_config)
    return ttnn.to_torch(tt_out)


# ---------------------------------------------------------------------------
# Test parameters — 2×2×3×3 grid: small/large for N,C; small/medium/large for H,W
# ---------------------------------------------------------------------------

_N_SIZES = [1, 8]  # small, large batch
_C_SIZES = [1, 4]  # small, large channel
_H_SIZES = [32, 128, 512]  # small, medium, large H
_W_SIZES = [64, 512, 2048]  # small, medium, large W
_H_FIXED = 32  # non-reduction H for W-reduction shapes
_W_FIXED = 64  # non-reduction W for H-reduction shapes


def _build_mean_shapes_and_dims():
    """Build (shape, dim, id) cases: N×C grid for W and H reduction, plus HW and odd."""
    out = []

    # W-reduction: vary N, C, W; fix H
    for n in _N_SIZES:
        for c in _C_SIZES:
            for w in _W_SIZES:
                out.append(((n, c, _H_FIXED, w), -1, f"W-{n}x{c}x{_H_FIXED}x{w}"))

    # H-reduction: vary N, C, H; fix W
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


_SHAPES_AND_DIMS = _build_mean_shapes_and_dims()


# ---------------------------------------------------------------------------
# BF16 tests
# ---------------------------------------------------------------------------

# BF16 max-ULP cap vs FP32-accumulated torch golden (see measure_ulp_with_near_zero_atol).
# fp32_dest_acc_en=True (BF16 inputs → FP32 accumulation → BF16 out): peak ~8 ULP on this grid.
# fp32_dest_acc_en=False (BF16 accumulation throughout): much wider; long reductions over
# near-zero-mean tensors see ~77x more ULP error than the FP32-acc path.
# This demonstrates that fp32 accumulation is essential for BF16 mean accuracy.
_BF16_ULP_THRESHOLD_FP32_DEST = 30
_BF16_ULP_THRESHOLD_BF16_DEST = 2500  # ~4x headroom over observed peak 618 ULP
_BF16_NEAR_ZERO_ATOL_FRACTION_FP32_DEST = 0.002  # tight; fp32 rounding error is tiny
_BF16_NEAR_ZERO_ATOL_FRACTION_BF16_DEST = 0.40  # loose; BF16 accum can be ~35% of range on near-zero mean


@pytest.mark.parametrize(
    "shape, dim, desc",
    _SHAPES_AND_DIMS,
    ids=[c[2] for c in _SHAPES_AND_DIMS],
)
@pytest.mark.parametrize("distribution", ["normal", "wide_uniform"])
@pytest.mark.parametrize("keepdim", [False, True], ids=["keepdim_false", "keepdim_true"])
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True], ids=["fp32_acc_off", "fp32_acc_on"])
def test_mean_ulp_bf16(device, shape, dim, desc, distribution, keepdim, fp32_dest_acc_en):
    """Characterize BF16 mean ULP vs FP32-accumulated Torch golden.

    fp32_dest_acc_en=True reflects the recommended path (BF16 inputs accumulated in FP32).
    fp32_dest_acc_en=False documents the accuracy cost of BF16-only accumulation.
    """
    torch.manual_seed(42)
    if distribution == "normal":
        x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    else:
        x = torch.empty(shape, dtype=torch.float32).uniform_(-1e3, 1e3).to(torch.bfloat16)

    golden = _golden_mean_bf16(x, dim=dim, keepdim=keepdim)
    ckc = _make_mean_compute_kernel_config(device, fp32_dest_acc_en)
    actual = _run_ttnn_mean(x, ttnn.bfloat16, device, dim=dim, keepdim=keepdim, compute_kernel_config=ckc)

    ulp_threshold = _BF16_ULP_THRESHOLD_FP32_DEST if fp32_dest_acc_en else _BF16_ULP_THRESHOLD_BF16_DEST
    atol_fraction = (
        _BF16_NEAR_ZERO_ATOL_FRACTION_FP32_DEST if fp32_dest_acc_en else _BF16_NEAR_ZERO_ATOL_FRACTION_BF16_DEST
    )
    passed, max_ulp, max_atol_err, atol_tol, msg, ulp_stats = measure_ulp_with_near_zero_atol(
        golden, actual, ulp_threshold, atol_fraction
    )
    spec = f"{desc} {distribution} shape={shape} dim={dim} keepdim={keepdim} fp32_acc={fp32_dest_acc_en}"
    logger.info(
        f"ttnn.mean ULP (BF16) | {spec} | ulp mean={ulp_stats['mean']:.3g} p95={ulp_stats['p95']:.3g} p99={ulp_stats['p99']:.3g} max={max_ulp:.4g}/{ulp_threshold} atol {max_atol_err:.4g}/{atol_tol:.4g} | {'ok' if passed else 'FAIL'}"
    )
    if ulp_stats["worst"]:
        logger.info(f"  worst: {ulp_stats['worst']}")
    if not passed:
        logger.info(f"  {msg}")
    assert passed, f"[BF16 {desc} {distribution} keepdim={keepdim} fp32_acc={fp32_dest_acc_en}] {msg}"


# ---------------------------------------------------------------------------
# FP32 tests
# ---------------------------------------------------------------------------

# FP32: unless the API uses SFPU-based true float32 accumulation, the tile engine accumulates in
# TF32, so accuracy may be lower / ULP higher than the IEEE 754 float32 reference (PyTorch).
# Large HW reductions on BH reached ~6.3e5 ULP class.
# fp32_dest_acc_en=True is required for FP32 inputs (device enforces this).
_FP32_ULP_THRESHOLD = 800_000
_FP32_NEAR_ZERO_ATOL_FRACTION = 0.001


@pytest.mark.parametrize(
    "shape, dim, desc",
    _SHAPES_AND_DIMS,
    ids=[c[2] for c in _SHAPES_AND_DIMS],
)
@pytest.mark.parametrize("distribution", ["normal", "wide_uniform"])
@pytest.mark.parametrize("keepdim", [False, True], ids=["keepdim_false", "keepdim_true"])
def test_mean_ulp_fp32(device, shape, dim, desc, distribution, keepdim):
    """Characterize FP32 mean ULP vs Torch FP32 golden; fp32_dest_acc_en=True only."""
    torch.manual_seed(42)
    if distribution == "normal":
        x = torch.randn(shape, dtype=torch.float32)
    else:
        x = torch.empty(shape, dtype=torch.float32).uniform_(-1e3, 1e3)

    golden = _golden_mean_fp32(x, dim=dim, keepdim=keepdim)
    ckc = _make_mean_compute_kernel_config(device, fp32_dest_acc_en=True)
    actual = _run_ttnn_mean(x, ttnn.float32, device, dim=dim, keepdim=keepdim, compute_kernel_config=ckc)

    passed, max_ulp, max_atol_err, atol_tol, msg, ulp_stats = measure_ulp_with_near_zero_atol(
        golden, actual, _FP32_ULP_THRESHOLD, _FP32_NEAR_ZERO_ATOL_FRACTION
    )
    spec = f"{desc} {distribution} shape={shape} dim={dim} keepdim={keepdim}"
    logger.info(
        f"ttnn.mean ULP (FP32, fp32_dest_acc_en=True) | {spec} | ulp mean={ulp_stats['mean']:.3g} p95={ulp_stats['p95']:.3g} p99={ulp_stats['p99']:.3g} max={max_ulp:.4g}/{_FP32_ULP_THRESHOLD} atol {max_atol_err:.4g}/{atol_tol:.4g} | {'ok' if passed else 'FAIL'}"
    )
    if ulp_stats["worst"]:
        logger.info(f"  worst: {ulp_stats['worst']}")
    if not passed:
        logger.info(f"  {msg}")
    assert passed, f"[FP32 {desc} {distribution} keepdim={keepdim}] {msg}"
