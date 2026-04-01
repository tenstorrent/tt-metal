# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
ULP-based accuracy characterization for ttnn.softmax vs PyTorch (issue #33741).

BF16 golden: torch.ops.aten._softmax.default(x, dim, half_to_float=False)
  with BF16 input x.  Matches the reference used in test_softmax_accuracy and
  keeps the reduction in the same dtype family as the fused device path.

FP32 golden: the same ATen op with FP32 input x.
  Device softmax may reorder work (tiling) vs sequential Torch; FP32 ULP can
  be large, so thresholds are set conservatively (see constants below).

ULP is measured in the output dtype (BF16 or FP32).  Elements where |golden|
is very small relative to the tensor's dynamic range are excluded from ULP
(where the metric breaks down) and validated with a scaled absolute-tolerance
check instead.

Metrics are logged at INFO for every parametrized case (pass or fail).  To
print them in the terminal, run pytest with e.g.:

  pytest .../test_softmax_ulp.py --log-cli-level=INFO

or capture to a file:

  pytest .../test_softmax_ulp.py --log-file=softmax_ulp.log --log-file-level=INFO
"""

import logging

import pytest

pytestmark = pytest.mark.use_module_device

import torch

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


def _make_softmax_compute_kernel_config(device, fp32_dest_acc_en: bool):
    """
    Prefer arch-aware kernel config (matches test_softmax_accuracy).  Some
    bindings or architectures may not accept all kwargs; fall back to
    Wormhole-style config like test_utils.get_compute_kernel_options.
    """
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


def _golden_softmax_bf16(x_bf16: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.ops.aten._softmax.default(x_bf16, dim, half_to_float=False)


def _golden_softmax_fp32(x_fp32: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.ops.aten._softmax.default(x_fp32, dim, half_to_float=False)


def _run_ttnn_softmax(
    input_torch: torch.Tensor,
    ttnn_dtype,
    device,
    dim: int,
    compute_kernel_config,
    numeric_stable: bool,
) -> torch.Tensor:
    tt_input = ttnn.from_torch(input_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.fill_implicit_tile_padding(tt_input, 42)
    tt_out = ttnn.softmax(
        tt_input,
        dim=dim,
        compute_kernel_config=compute_kernel_config,
        numeric_stable=numeric_stable,
    )
    return ttnn.to_torch(tt_out)


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
# Test parameters
# ---------------------------------------------------------------------------

_SHAPES_AND_DIMS = [
    ((1, 1, 32, 1024), -1, "W-1024"),
    ((1, 1, 1024, 32), -2, "H-1024"),
    ((1, 1, 64, 128), -1, "W-128"),
    ((1, 1, 128, 64), -2, "H-128"),
    ((1, 1, 37, 41), -1, "W-odd-41"),
    ((1, 1, 37, 41), -2, "H-odd-37"),
    ((1, 1, 32, 32768), -1, "W-32768-large-reduction"),
    ((1, 1, 4096, 32), -2, "H-4096-large-reduction"),
]


# ---------------------------------------------------------------------------
# BF16 tests
# ---------------------------------------------------------------------------

# BF16: test_softmax_accuracy reports worst-case expected ULP up to 13
# (math_approx True, fp32_acc False, numeric_stable True).  Start slightly
# above that until CI measurements tighten the bound.
_BF16_ULP_THRESHOLD = 16
_BF16_NEAR_ZERO_ATOL_FRACTION = 0.02


@pytest.mark.parametrize(
    "shape, dim, desc",
    _SHAPES_AND_DIMS,
    ids=[c[2] for c in _SHAPES_AND_DIMS],
)
@pytest.mark.parametrize("distribution", ["normal", "wide_uniform"])
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True], ids=["fp32_acc_off", "fp32_acc_on"])
@pytest.mark.parametrize("numeric_stable", [False, True], ids=["numstab_off", "numstab_on"])
def test_softmax_ulp_bf16(device, shape, dim, desc, distribution, fp32_dest_acc_en, numeric_stable):
    """Characterize BF16 softmax ULP vs torch.ops.aten._softmax.default golden."""
    torch.manual_seed(42)
    if distribution == "normal":
        x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    else:
        x = torch.empty(shape, dtype=torch.float32).uniform_(-1e3, 1e3).to(torch.bfloat16)

    golden = _golden_softmax_bf16(x, dim)
    compute_kernel_config = _make_softmax_compute_kernel_config(device, fp32_dest_acc_en)
    actual = _run_ttnn_softmax(x, ttnn.bfloat16, device, dim, compute_kernel_config, numeric_stable)

    passed, max_ulp, max_atol_err, msg = _measure_ulp_safe(
        golden, actual, _BF16_ULP_THRESHOLD, _BF16_NEAR_ZERO_ATOL_FRACTION
    )
    logger.info(
        "ttnn.softmax ULP dtype=BF16 desc=%r distribution=%r shape=%s dim=%s "
        "fp32_dest_acc_en=%s numeric_stable=%s max_ulp=%s ulp_threshold=%s "
        "max_atol_err=%s passed=%s | %s",
        desc,
        distribution,
        shape,
        dim,
        fp32_dest_acc_en,
        numeric_stable,
        max_ulp,
        _BF16_ULP_THRESHOLD,
        max_atol_err,
        passed,
        msg,
    )
    assert passed, f"[BF16 {desc} {distribution} fp32_acc={fp32_dest_acc_en} numstab={numeric_stable}] {msg}"


# ---------------------------------------------------------------------------
# FP32 tests
# ---------------------------------------------------------------------------

# FP32: same ordering/tile vs sequential differences as ttnn.mean; measured
# mean ULP is ~1.6M in test_mean_ulp.  Use the same conservative cap until
# softmax-specific sweeps are recorded.
_FP32_ULP_THRESHOLD = 2_000_000
_FP32_NEAR_ZERO_ATOL_FRACTION = 0.005


@pytest.mark.parametrize(
    "shape, dim, desc",
    _SHAPES_AND_DIMS,
    ids=[c[2] for c in _SHAPES_AND_DIMS],
)
@pytest.mark.parametrize("distribution", ["normal", "wide_uniform"])
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True], ids=["fp32_acc_off", "fp32_acc_on"])
@pytest.mark.parametrize("numeric_stable", [False, True], ids=["numstab_off", "numstab_on"])
def test_softmax_ulp_fp32(device, shape, dim, desc, distribution, fp32_dest_acc_en, numeric_stable):
    """Characterize FP32 softmax ULP vs torch.ops.aten._softmax.default golden."""
    torch.manual_seed(42)
    if distribution == "normal":
        x = torch.randn(shape, dtype=torch.float32)
    else:
        x = torch.empty(shape, dtype=torch.float32).uniform_(-1e3, 1e3)

    golden = _golden_softmax_fp32(x, dim)
    compute_kernel_config = _make_softmax_compute_kernel_config(device, fp32_dest_acc_en)
    actual = _run_ttnn_softmax(x, ttnn.float32, device, dim, compute_kernel_config, numeric_stable)

    passed, max_ulp, max_atol_err, msg = _measure_ulp_safe(
        golden, actual, _FP32_ULP_THRESHOLD, _FP32_NEAR_ZERO_ATOL_FRACTION
    )
    logger.info(
        "ttnn.softmax ULP dtype=FP32 desc=%r distribution=%r shape=%s dim=%s "
        "fp32_dest_acc_en=%s numeric_stable=%s max_ulp=%s ulp_threshold=%s "
        "max_atol_err=%s passed=%s | %s",
        desc,
        distribution,
        shape,
        dim,
        fp32_dest_acc_en,
        numeric_stable,
        max_ulp,
        _FP32_ULP_THRESHOLD,
        max_atol_err,
        passed,
        msg,
    )
    assert passed, f"[FP32 {desc} {distribution} fp32_acc={fp32_dest_acc_en} numstab={numeric_stable}] {msg}"
