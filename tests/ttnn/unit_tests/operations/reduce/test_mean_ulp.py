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
  - PyTorch FP32 accumulation (same precision as device).
  - Both device and PyTorch use FP32 arithmetic; differences arise from
    operation ordering (tile-based vs sequential).

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
from tests.ttnn.utils_for_testing import measure_ulp_with_near_zero_atol


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _golden_mean_bf16(input_bf16: torch.Tensor, dim, keepdim: bool) -> torch.Tensor:
    """FP32-accumulated mean, result cast back to BF16."""
    return torch.mean(input_bf16.float(), dim=dim, keepdim=keepdim).to(torch.bfloat16)


def _golden_mean_fp32(input_fp32: torch.Tensor, dim, keepdim: bool) -> torch.Tensor:
    """PyTorch FP32 mean — same accumulation width as device."""
    return torch.mean(input_fp32, dim=dim, keepdim=keepdim)


def _run_ttnn_mean(input_torch: torch.Tensor, ttnn_dtype, device, dim, keepdim: bool) -> torch.Tensor:
    """Send tensor to device, run ttnn.mean, return host torch tensor."""
    tt_input = ttnn.from_torch(input_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.fill_implicit_tile_padding(tt_input, 42)
    tt_out = ttnn.mean(tt_input, dim=dim, keepdim=keepdim)
    return ttnn.to_torch(tt_out)


# ---------------------------------------------------------------------------
# Test parameters (generated sweeps; adjust steps to trade coverage vs runtime)
# ---------------------------------------------------------------------------

# Reduction along W: shape (1,1,32,W); dim=-1
_MEAN_W_SIZES = sorted(set(range(256, 8193, 1024)) | {1024, 2048, 4096, 8192, 32768})
# Reduction along H: shape (1,1,H,32); dim=-2
_MEAN_H_SIZES = sorted(set(range(128, 4097, 1024)) | {512, 1024, 2048, 4096})
# 2D mean over HW: square and a few non-square tile-aligned pairs
_MEAN_HW_SQUARES = list(range(64, 513, 64))
_MEAN_HW_MIXED = [(96, 160), (128, 192), (192, 256)]
# Batch / channel sweeps for (B, C, 32, 32)
_MEAN_BATCH_SIZES = [2, 4, 8, 16, 32]  # 32 = one full tile row; tests tile-boundary batch reduction
_MEAN_CHANNEL_SIZES = [3, 5, 7, 16, 32]  # 32 = tile-boundary channel reduction


def _build_mean_shapes_and_dims():
    """Build (shape, dim, id) cases from numeric sweeps plus odd / corner cases."""
    out = []

    for w in _MEAN_W_SIZES:
        out.append(((1, 1, 32, w), -1, f"W-{w}"))

    for h in _MEAN_H_SIZES:
        out.append(((1, 1, h, 32), -2, f"H-{h}"))

    for side in _MEAN_HW_SQUARES:
        out.append(((1, 1, side, side), [-2, -1], f"HW-{side}x{side}"))
    for hh, ww in _MEAN_HW_MIXED:
        out.append(((1, 1, hh, ww), [-2, -1], f"HW-{hh}x{ww}"))

    out.extend(
        [
            ((1, 1, 37, 41), -1, "W-odd-41"),
            ((1, 1, 37, 41), -2, "H-odd-37"),
        ]
    )

    for b in _MEAN_BATCH_SIZES:
        out.append(((b, 1, 48, 64), -1, f"W-batch{b}-48x64"))

    for B in _MEAN_BATCH_SIZES:
        out.append(((B, 3, 32, 32), 0, f"batch-{B}"))

    for C in _MEAN_CHANNEL_SIZES:
        out.append(((2, C, 32, 32), 1, f"channel-{C}"))

    out.append(((4, 3, 32, 32), [0, 1], "batch+channel"))

    # W/H reductions with non-trivial outer dims: confirms long-reduction ULP profile
    # holds when there are multiple batch and channel slices (different code path vs N=C=1).
    for w in [1024, 4096]:
        out.append(((4, 3, 32, w), -1, f"W-4x3-{w}"))
    for h in [512, 2048]:
        out.append(((4, 3, h, 32), -2, f"H-4x3-{h}"))

    return out


_SHAPES_AND_DIMS = _build_mean_shapes_and_dims()


# ---------------------------------------------------------------------------
# BF16 tests
# ---------------------------------------------------------------------------

# BF16 max-ULP cap vs FP32-accumulated torch golden (see measure_ulp_with_near_zero_atol).
# Shape/distribution sweeps (long reductions, wide_uniform) widen tails vs tiny fixed shapes.
_BF16_ULP_THRESHOLD = 30
_BF16_NEAR_ZERO_ATOL_FRACTION = 0.002


@pytest.mark.parametrize(
    "shape, dim, desc",
    _SHAPES_AND_DIMS,
    ids=[c[2] for c in _SHAPES_AND_DIMS],
)
@pytest.mark.parametrize("distribution", ["normal", "wide_uniform"])
@pytest.mark.parametrize("keepdim", [False, True], ids=["keepdim_false", "keepdim_true"])
def test_mean_ulp_bf16(device, shape, dim, desc, distribution, keepdim):
    """Characterize BF16 mean ULP vs FP32-accumulated Torch golden."""
    torch.manual_seed(42)
    if distribution == "normal":
        x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    else:
        x = torch.empty(shape, dtype=torch.float32).uniform_(-1e3, 1e3).to(torch.bfloat16)

    golden = _golden_mean_bf16(x, dim=dim, keepdim=keepdim)
    actual = _run_ttnn_mean(x, ttnn.bfloat16, device, dim=dim, keepdim=keepdim)

    passed, max_ulp, max_atol_err, atol_tol, msg = measure_ulp_with_near_zero_atol(
        golden, actual, _BF16_ULP_THRESHOLD, _BF16_NEAR_ZERO_ATOL_FRACTION
    )
    spec = f"{desc} {distribution} shape={shape} dim={dim} keepdim={keepdim}"
    logger.info(
        f"ttnn.mean ULP (BF16) | {spec} | ulp {max_ulp:.4g}/{_BF16_ULP_THRESHOLD} atol {max_atol_err:.4g}/{atol_tol:.4g} | {'ok' if passed else 'FAIL'}"
    )
    if not passed:
        logger.info(f"  {msg}")
    assert passed, f"[BF16 {desc} {distribution} keepdim={keepdim}] {msg}"


# ---------------------------------------------------------------------------
# FP32 tests
# ---------------------------------------------------------------------------

# FP32: tile vs sequential ordering; sweeps add large 2D HW reductions—BH hit ~7.3e5 ULP class.
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
    """Characterize FP32 mean ULP vs Torch FP32 golden."""
    torch.manual_seed(42)
    if distribution == "normal":
        x = torch.randn(shape, dtype=torch.float32)
    else:
        x = torch.empty(shape, dtype=torch.float32).uniform_(-1e3, 1e3)

    golden = _golden_mean_fp32(x, dim=dim, keepdim=keepdim)
    actual = _run_ttnn_mean(x, ttnn.float32, device, dim=dim, keepdim=keepdim)

    passed, max_ulp, max_atol_err, atol_tol, msg = measure_ulp_with_near_zero_atol(
        golden, actual, _FP32_ULP_THRESHOLD, _FP32_NEAR_ZERO_ATOL_FRACTION
    )
    spec = f"{desc} {distribution} shape={shape} dim={dim} keepdim={keepdim}"
    logger.info(
        f"ttnn.mean ULP (FP32) | {spec} | ulp {max_ulp:.4g}/{_FP32_ULP_THRESHOLD} atol {max_atol_err:.4g}/{atol_tol:.4g} | {'ok' if passed else 'FAIL'}"
    )
    if not passed:
        logger.info(f"  {msg}")
    assert passed, f"[FP32 {desc} {distribution} keepdim={keepdim}] {msg}"
