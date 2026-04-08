# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
ULP-based accuracy characterization for ttnn.operations.moreh.logsoftmax (issue #33744).

BF16 golden: F.log_softmax(bf16_input.float(), dim=...).to(torch.bfloat16)
  Promotes to FP32 for the reduction, then casts back to BF16 -- matching the
  FP32-accumulated reference convention used across these ULP tests.

FP32 golden: F.log_softmax(fp32_input, dim=...)
  Same nominal precision as a full FP32 device path; differences vs hardware
  reflect operation ordering (tile-local vs fully sequential).

FP32 inputs require fp32_dest_acc_en=True in the compute kernel config;
the FP32 ULP tests only exercise that supported combination.

ULP is measured in the output dtype (BF16 or FP32).  Elements where
|golden| is very small relative to the tensor's dynamic range are excluded
from ULP and validated with a scaled absolute-tolerance check instead.

Metrics logged with loguru at INFO per parametrized case.
"""

import pytest

pytestmark = pytest.mark.use_module_device

import torch
import torch.nn.functional as F
from loguru import logger

import ttnn

from tests.ttnn.unit_tests.operations.test_utils import get_compute_kernel_options
from tests.ttnn.utils_for_testing import measure_ulp_with_near_zero_atol


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_logsoftmax_compute_kernel_config(device, fp32_dest_acc_en: bool):
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


def _golden_logsoftmax_bf16(input_bf16: torch.Tensor, dim: int) -> torch.Tensor:
    """Log-softmax with FP32 intermediate, result cast to BF16."""
    return F.log_softmax(input_bf16.float(), dim=dim).to(torch.bfloat16)


def _golden_logsoftmax_fp32(input_fp32: torch.Tensor, dim: int) -> torch.Tensor:
    """PyTorch FP32 log-softmax."""
    return F.log_softmax(input_fp32, dim=dim)


def _run_ttnn_logsoftmax(
    input_torch: torch.Tensor,
    ttnn_dtype,
    device,
    dim: int,
    strategy,
    fp32_dest_acc_en: bool,
) -> torch.Tensor:
    """Send tensor to device, run moreh logsoftmax, return host torch tensor."""
    tt_input = ttnn.from_torch(input_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.fill_implicit_tile_padding(tt_input, 42)
    compute_kernel_config = _make_logsoftmax_compute_kernel_config(device, fp32_dest_acc_en)
    tt_out = ttnn.operations.moreh.logsoftmax(
        tt_input,
        dim,
        compute_kernel_config=compute_kernel_config,
        strategy=strategy,
    )
    return ttnn.to_torch(tt_out).to(input_torch.dtype)


# ---------------------------------------------------------------------------
# Test parameters: strategy-consistent shape sweeps (multiples of 32 for tiles)
# ---------------------------------------------------------------------------

_SMALL_INNER = list(range(32, 385, 64))  # 32..352 step 64
# LARGE_W: dim is last; penultimate matches 32*4 tile-style layouts from callback tests
_LARGE_W_LAST = [160, 320, 480]  # (2,3,128,w), dim=3
_LARGE_H_PENULT = [128, 256, 384]  # dim 2 on (2,3,h,160)
_LARGE_C_DIMS = [7, 8, 15, 16, 23, 31, 32]  # channel count for (1,c,32,32) dim 1; includes tile-aligned 8, 16, 32


def _build_moreh_logsoftmax_strategy_cases():
    st = ttnn.operations.moreh.SoftmaxOpParallelizationStrategy
    cases = []

    for w in _SMALL_INNER:
        cases.append(((32, w), 1, st.SMALL_W, f"SMALL_W-32x{w}"))
    for h in _SMALL_INNER:
        cases.append(((h, 32), 0, st.SMALL_H, f"SMALL_H-{h}x32"))

    for w in _LARGE_W_LAST:
        cases.append(((2, 3, 128, w), 3, st.LARGE_W, f"LARGE_W-2x3x128x{w}"))
    cases.append(((1, 2, 64, 320), 3, st.LARGE_W, "LARGE_W-1x2x64x320"))
    for h in _LARGE_H_PENULT:
        cases.append(((2, 3, h, 160), 2, st.LARGE_H, f"LARGE_H-2x3x{h}x160"))

    for c in _LARGE_C_DIMS:
        cases.append(((1, c, 32, 32), 1, st.LARGE_C, f"LARGE_C-1x{c}x32x32"))

    return cases


_STRATEGY_CASES = _build_moreh_logsoftmax_strategy_cases()


# ---------------------------------------------------------------------------
# BF16 tests
# ---------------------------------------------------------------------------

# BF16 max-ULP caps: fp32_dest_acc_en=True uses the tighter threshold (12), False uses 24.
_BF16_ULP_THRESHOLD_FP32_DEST = 12
_BF16_ULP_THRESHOLD_BF16_DEST = 24
_BF16_NEAR_ZERO_ATOL_FRACTION = 0.001


@pytest.mark.parametrize("shape, dim, strategy, desc", _STRATEGY_CASES, ids=[c[3] for c in _STRATEGY_CASES])
@pytest.mark.parametrize("distribution", ["uniform_01", "normal", "wide_uniform"])
@pytest.mark.parametrize(
    "fp32_dest_acc_en",
    [False, True],
    ids=["fp32_dest_acc_en=False", "fp32_dest_acc_en=True"],
)
def test_moreh_logsoftmax_ulp_bf16(device, shape, dim, strategy, desc, distribution, fp32_dest_acc_en):
    """Characterize BF16 moreh logsoftmax ULP vs FP32-intermediate Torch golden."""
    torch.manual_seed(42)
    if distribution == "uniform_01":
        x = torch.empty(shape, dtype=torch.float32).uniform_(0.0, 1.0).to(torch.bfloat16)
    elif distribution == "normal":
        x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    else:
        x = torch.empty(shape, dtype=torch.float32).uniform_(-1e3, 1e3).to(torch.bfloat16)

    ulp_threshold = _BF16_ULP_THRESHOLD_FP32_DEST if fp32_dest_acc_en else _BF16_ULP_THRESHOLD_BF16_DEST

    golden = _golden_logsoftmax_bf16(x, dim=dim)
    actual = _run_ttnn_logsoftmax(x, ttnn.bfloat16, device, dim, strategy, fp32_dest_acc_en)

    passed, max_ulp, max_atol_err, atol_tol, msg, ulp_stats = measure_ulp_with_near_zero_atol(
        golden, actual, ulp_threshold, _BF16_NEAR_ZERO_ATOL_FRACTION
    )
    spec = f"{desc} {strategy} {distribution} shape={shape} dim={dim} fp32_acc={fp32_dest_acc_en}"
    logger.info(
        f"moreh.logsoftmax ULP (BF16) | {spec} | ulp mean={ulp_stats['mean']:.3g} p95={ulp_stats['p95']:.3g} p99={ulp_stats['p99']:.3g} max={max_ulp:.4g}/{ulp_threshold} atol {max_atol_err:.4g}/{atol_tol:.4g} | {'ok' if passed else 'FAIL'}"
    )
    if ulp_stats["worst"]:
        logger.info(f"  worst: {ulp_stats['worst']}")
    if not passed:
        logger.info(f"  {msg}")
    assert passed, f"[BF16 {desc} {distribution} fp32_dest_acc_en={fp32_dest_acc_en}] {msg}"


# ---------------------------------------------------------------------------
# FP32 tests
# ---------------------------------------------------------------------------

# FP32 fp32_dest_acc_en=True; BH max ~2.5e5 ULP (wide_uniform, large tensors).
_FP32_ULP_THRESHOLD = 400_000
_FP32_NEAR_ZERO_ATOL_FRACTION = 0.001


@pytest.mark.parametrize("shape, dim, strategy, desc", _STRATEGY_CASES, ids=[c[3] for c in _STRATEGY_CASES])
@pytest.mark.parametrize("distribution", ["uniform_01", "normal", "wide_uniform"])
def test_moreh_logsoftmax_ulp_fp32(device, shape, dim, strategy, desc, distribution):
    """Characterize FP32 moreh logsoftmax ULP vs Torch FP32 golden.

    fp32_dest_acc_en=True is required for FLOAT32 input (device enforces this).
    """
    torch.manual_seed(42)
    if distribution == "uniform_01":
        x = torch.empty(shape, dtype=torch.float32).uniform_(0.0, 1.0)
    elif distribution == "normal":
        x = torch.randn(shape, dtype=torch.float32)
    else:
        x = torch.empty(shape, dtype=torch.float32).uniform_(-1e3, 1e3)

    golden = _golden_logsoftmax_fp32(x, dim=dim)
    actual = _run_ttnn_logsoftmax(x, ttnn.float32, device, dim, strategy, fp32_dest_acc_en=True)

    passed, max_ulp, max_atol_err, atol_tol, msg, ulp_stats = measure_ulp_with_near_zero_atol(
        golden, actual, _FP32_ULP_THRESHOLD, _FP32_NEAR_ZERO_ATOL_FRACTION
    )
    spec = f"{desc} {strategy} {distribution} shape={shape} dim={dim}"
    logger.info(
        f"moreh.logsoftmax ULP (FP32, fp32_dest_acc_en=True) | {spec} | ulp mean={ulp_stats['mean']:.3g} p95={ulp_stats['p95']:.3g} p99={ulp_stats['p99']:.3g} max={max_ulp:.4g}/{_FP32_ULP_THRESHOLD} atol {max_atol_err:.4g}/{atol_tol:.4g} | {'ok' if passed else 'FAIL'}"
    )
    if ulp_stats["worst"]:
        logger.info(f"  worst: {ulp_stats['worst']}")
    if not passed:
        logger.info(f"  {msg}")
    assert passed, f"[FP32 {desc} {distribution}] {msg}"
