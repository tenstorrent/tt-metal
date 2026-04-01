# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
ULP-based accuracy characterization for ttnn.operations.moreh.logsoftmax (issue #33744).

Golden: torch.nn.functional.log_softmax(..., dim=...).

BF16 reference (recommended for comparing device BF16 output):
  F.log_softmax(bf16_input.float(), dim=...).to(torch.bfloat16)
  - Softmax reduction uses FP32 on the host, then the result is cast to BF16.
  - This isolates BF16 storage/rounding on the input from the numerics of the
    log-softmax reduction (analogous to the FP32-accumulated mean golden).

Alternative BF16-in-BF16 golden would be F.log_softmax(bf16_input, dim=...) with
PyTorch computing internally in wider precision for some steps; the float-input
form above is explicit and matches the “wider intermediate” intent.

FP32 reference:
  F.log_softmax(fp32_input, dim=...)
  - Same nominal precision as a full FP32 device path; differences vs hardware
    mainly reflect operation ordering (tile-local vs fully sequential).

ULP is measured in the output dtype (BF16 or FP32).  Elements where
|golden| is very small relative to the tensor's dynamic range are excluded
from ULP (where the metric breaks down due to division by a tiny ULP
quantum) and validated with a scaled absolute-tolerance check instead.

Metrics are logged at INFO for every parametrized case (pass or fail).  To
print them in the terminal, run pytest with e.g.:

  pytest .../test_moreh_logsoftmax_ulp.py --log-cli-level=INFO

or capture to a file:

  pytest .../test_moreh_logsoftmax_ulp.py --log-file=logsoftmax_ulp.log --log-file-level=INFO
"""

import logging

import pytest

pytestmark = pytest.mark.use_module_device

import torch
import torch.nn.functional as F

import ttnn
from models.common.utility_functions import ulp as compute_ulp

from tests.ttnn.unit_tests.operations.test_utils import get_compute_kernel_options

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Elements with |golden| below this fraction of max(|golden|) are excluded
# from the ULP check and verified with absolute tolerance instead.  This
# avoids the well-documented breakdown of ULP near zero.
_NEAR_ZERO_RELATIVE_FRACTION = 1e-2


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
    compute_kernel_config = get_compute_kernel_options(fp32_dest_acc_en)
    tt_out = ttnn.operations.moreh.logsoftmax(
        tt_input,
        dim,
        compute_kernel_config=compute_kernel_config,
        strategy=strategy,
    )
    return ttnn.to_torch(tt_out).to(input_torch.dtype)


def _measure_ulp_safe(golden, actual, ulp_threshold, near_zero_atol_fraction):
    """
    Measure max ULP while handling near-zero golden values safely.

    Elements where |golden| < _NEAR_ZERO_RELATIVE_FRACTION * max(|golden|)
    are excluded from the ULP check and validated with an absolute tolerance
    equal to near_zero_atol_fraction * max(|golden|).

    Returns (passed, max_ulp, max_atol_err, message).
    """
    if golden.dtype != actual.dtype:
        actual = actual.to(golden.dtype)

    abs_golden = torch.abs(golden.float())
    golden_max = abs_golden.max().item()
    if golden_max == 0:
        abs_err = torch.abs(actual.float()).max().item()
        return abs_err == 0, 0.0, abs_err, f"All-zero golden; max |actual|={abs_err:.6e}"

    dynamic_threshold = _NEAR_ZERO_RELATIVE_FRACTION * golden_max
    normal_mask = abs_golden >= dynamic_threshold
    near_zero_mask = ~normal_mask
    n_near_zero = near_zero_mask.sum().item()

    # Scaled absolute tolerance for near-zero elements: proportional to the
    # golden tensor's dynamic range so it naturally adapts to input magnitude.
    scaled_atol = near_zero_atol_fraction * golden_max

    # --- ULP check on elements above the dynamic threshold ---
    max_ulp = 0.0
    ulp_msg = ""
    if normal_mask.any():
        g = golden[normal_mask]
        a = actual[normal_mask]
        ulp_values = compute_ulp(g)
        ulp_diffs = torch.abs(a.float() - g.float()) / ulp_values.float()
        max_ulp = torch.max(ulp_diffs).item()
        if max_ulp > ulp_threshold:
            worst = torch.argmax(ulp_diffs)
            ulp_msg = (
                f"Max ULP: {max_ulp:.1f} "
                f"(golden={g[worst].item()}, actual={a[worst].item()}, "
                f"ulp@golden={ulp_values[worst].item()})"
            )

    # --- Absolute tolerance check on near-zero elements ---
    max_atol_err = 0.0
    atol_msg = ""
    if n_near_zero > 0:
        g_nz = golden[near_zero_mask].float()
        a_nz = actual[near_zero_mask].float()
        abs_diffs = torch.abs(a_nz - g_nz)
        max_atol_err = torch.max(abs_diffs).item()
        if max_atol_err > scaled_atol:
            worst = torch.argmax(abs_diffs)
            atol_msg = (
                f"Max atol err: {max_atol_err:.6e} > {scaled_atol:.6e} "
                f"(golden={g_nz[worst].item()}, actual={a_nz[worst].item()}) "
                f"[{n_near_zero} near-zero elems]"
            )

    ulp_ok = max_ulp <= ulp_threshold
    atol_ok = max_atol_err <= scaled_atol

    parts = [f"ULP: max={max_ulp:.1f} (threshold={ulp_threshold})"]
    if n_near_zero > 0:
        parts.append(
            f"Near-zero atol: max={max_atol_err:.6e} "
            f"(threshold={scaled_atol:.6e}, count={n_near_zero}/{golden.numel()})"
        )
    msg = "; ".join(parts)
    if not ulp_ok:
        msg += f" | FAIL ULP: {ulp_msg}"
    if not atol_ok:
        msg += f" | FAIL atol: {atol_msg}"

    return ulp_ok and atol_ok, max_ulp, max_atol_err, msg


# ---------------------------------------------------------------------------
# Test parameters (shapes / dims / strategies aligned with callback tests)
# ---------------------------------------------------------------------------

_STRATEGY_CASES = [
    ((32, 32), 1, ttnn.operations.moreh.SoftmaxOpParallelizationStrategy.SMALL_W, "SMALL_W"),
    ((32, 32), 0, ttnn.operations.moreh.SoftmaxOpParallelizationStrategy.SMALL_H, "SMALL_H"),
    ((2, 3, 32 * 4, 32 * 5), 3, ttnn.operations.moreh.SoftmaxOpParallelizationStrategy.LARGE_W, "LARGE_W"),
    ((2, 3, 32 * 4, 32 * 5), 2, ttnn.operations.moreh.SoftmaxOpParallelizationStrategy.LARGE_H, "LARGE_H"),
    ((1, 15, 32, 32), 1, ttnn.operations.moreh.SoftmaxOpParallelizationStrategy.LARGE_C, "LARGE_C"),
]


# ---------------------------------------------------------------------------
# BF16 tests
# ---------------------------------------------------------------------------

# ULP limits depend on dest accumulation: FP32 dest tightens the reduction.
_BF16_ULP_THRESHOLD_FP32_DEST = 16
_BF16_ULP_THRESHOLD_BF16_DEST = 32
_BF16_NEAR_ZERO_ATOL_FRACTION = 0.02


@pytest.mark.parametrize(
    "shape, dim, strategy, desc",
    _STRATEGY_CASES,
    ids=[c[3] for c in _STRATEGY_CASES],
)
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

    passed, max_ulp, max_atol_err, msg = _measure_ulp_safe(golden, actual, ulp_threshold, _BF16_NEAR_ZERO_ATOL_FRACTION)
    logger.info(
        "moreh.logsoftmax ULP dtype=BF16 desc=%r distribution=%r shape=%s dim=%s "
        "fp32_dest_acc_en=%s max_ulp=%s ulp_threshold=%s max_atol_err=%s passed=%s | %s",
        desc,
        distribution,
        shape,
        dim,
        fp32_dest_acc_en,
        max_ulp,
        ulp_threshold,
        max_atol_err,
        passed,
        msg,
    )
    assert passed, f"[BF16 {desc} {distribution} fp32_dest_acc_en={fp32_dest_acc_en}] {msg}"


# ---------------------------------------------------------------------------
# FP32 tests
# ---------------------------------------------------------------------------

# FP32 ULP threshold: device softmax uses tile-ordered reductions and possibly
# different intermediate staging vs a flat PyTorch reference, while exp/log
# amplify ordering differences.  This bound is intentionally conservative so
# CI is stable; tighten only after measuring per-strategy maxima.
_FP32_ULP_THRESHOLD = 8_000_000
_FP32_NEAR_ZERO_ATOL_FRACTION = 0.005


@pytest.mark.parametrize(
    "shape, dim, strategy, desc",
    _STRATEGY_CASES,
    ids=[c[3] for c in _STRATEGY_CASES],
)
@pytest.mark.parametrize("distribution", ["uniform_01", "normal", "wide_uniform"])
@pytest.mark.parametrize(
    "fp32_dest_acc_en",
    [False, True],
    ids=["fp32_dest_acc_en=False", "fp32_dest_acc_en=True"],
)
def test_moreh_logsoftmax_ulp_fp32(device, shape, dim, strategy, desc, distribution, fp32_dest_acc_en):
    """Characterize FP32 moreh logsoftmax ULP vs Torch FP32 golden."""
    torch.manual_seed(42)
    if distribution == "uniform_01":
        x = torch.empty(shape, dtype=torch.float32).uniform_(0.0, 1.0)
    elif distribution == "normal":
        x = torch.randn(shape, dtype=torch.float32)
    else:
        x = torch.empty(shape, dtype=torch.float32).uniform_(-1e3, 1e3)

    golden = _golden_logsoftmax_fp32(x, dim=dim)
    actual = _run_ttnn_logsoftmax(x, ttnn.float32, device, dim, strategy, fp32_dest_acc_en)

    passed, max_ulp, max_atol_err, msg = _measure_ulp_safe(
        golden, actual, _FP32_ULP_THRESHOLD, _FP32_NEAR_ZERO_ATOL_FRACTION
    )
    logger.info(
        "moreh.logsoftmax ULP dtype=FP32 desc=%r distribution=%r shape=%s dim=%s "
        "fp32_dest_acc_en=%s max_ulp=%s ulp_threshold=%s max_atol_err=%s passed=%s | %s",
        desc,
        distribution,
        shape,
        dim,
        fp32_dest_acc_en,
        max_ulp,
        _FP32_ULP_THRESHOLD,
        max_atol_err,
        passed,
        msg,
    )
    assert passed, f"[FP32 {desc} {distribution} fp32_dest_acc_en={fp32_dest_acc_en}] {msg}"
