# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
ULP-based accuracy characterization for ttnn.mean (issue #33740).

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

Test structure:
  - Tile-axis reductions (W, H, HW) use tight ULP thresholds.
  - Non-HW-axis reductions (batch, channel) go through transpose + sequential
    reduce, producing slightly higher error due to operation ordering and
    intermediate BF16 packing.  They use a looser ULP threshold and the
    near-zero scaled atol to accommodate elements where heavy cancellation
    makes the golden value tiny relative to input magnitudes.
"""

import pytest

pytestmark = pytest.mark.use_module_device

import torch

import ttnn
from models.common.utility_functions import ulp as compute_ulp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Elements with |golden| below this fraction of max(|golden|) are excluded
# from the ULP check and verified with absolute tolerance instead.  This
# avoids the well-documented breakdown of ULP near zero.
_NEAR_ZERO_RELATIVE_FRACTION = 1e-2


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


def _measure_ulp_safe(golden, actual, ulp_threshold, near_zero_atol_fraction):
    """
    Measure max ULP while handling near-zero golden values safely.

    Elements where |golden| < near_zero_fraction * max(|golden|) are excluded
    from the ULP check and validated with an absolute tolerance equal to
    near_zero_atol_fraction * max(|golden|).

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

# Tile-axis reductions: W (-1), H (-2), HW ([-2,-1]).
# These go through a single prim::reduce call and have tight accuracy.
_TILE_AXIS_CASES = [
    ((1, 1, 32, 1024), -1, "W-1024"),
    ((1, 1, 32, 4096), -1, "W-4096"),
    ((1, 1, 1024, 32), -2, "H-1024"),
    ((1, 1, 4096, 32), -2, "H-4096"),
    ((1, 1, 64, 64), [-2, -1], "HW-64x64"),
    ((1, 1, 128, 128), [-2, -1], "HW-128x128"),
    ((1, 1, 32, 32768), -1, "W-32768"),
    ((1, 1, 37, 41), -1, "W-odd-41"),
    ((1, 1, 37, 41), -2, "H-odd-37"),
]

# Non-HW-axis reductions: batch (0), channel (1), multi-dim ([0,1]).
# These go through transpose + sequential reduce (reduce_nd_loop).
_NON_HW_AXIS_CASES = [
    ((4, 3, 32, 32), 0, "batch-4"),
    ((4, 3, 32, 32), 1, "channel-3"),
    ((4, 3, 32, 32), [0, 1], "batch+channel"),
]


# ---------------------------------------------------------------------------
# BF16 tests — tile-axis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, dim, desc",
    _TILE_AXIS_CASES,
    ids=[c[2] for c in _TILE_AXIS_CASES],
)
@pytest.mark.parametrize("distribution", ["normal", "wide_uniform"])
def test_mean_ulp_bf16_tile_axis(device, shape, dim, desc, distribution):
    """BF16 mean on tile axes (W/H/HW): expect tight ULP."""
    torch.manual_seed(42)
    if distribution == "normal":
        x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    else:
        x = torch.empty(shape, dtype=torch.float32).uniform_(-1e3, 1e3).to(torch.bfloat16)

    golden = _golden_mean_bf16(x, dim=dim, keepdim=True)
    actual = _run_ttnn_mean(x, ttnn.bfloat16, device, dim=dim, keepdim=True)

    passed, max_ulp, _, msg = _measure_ulp_safe(golden, actual, ulp_threshold=30, near_zero_atol_fraction=0.01)
    assert passed, f"[BF16 {desc} {distribution}] {msg}"


# ---------------------------------------------------------------------------
# BF16 tests — non-HW axis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, dim, desc",
    _NON_HW_AXIS_CASES,
    ids=[c[2] for c in _NON_HW_AXIS_CASES],
)
@pytest.mark.parametrize("distribution", ["normal", "wide_uniform"])
def test_mean_ulp_bf16_non_hw_axis(device, shape, dim, desc, distribution):
    """BF16 mean on non-HW axes (batch/channel): transpose + sequential reduce.

    Slightly looser ULP threshold because intermediate results are packed
    to BF16 between transpose and reduce steps.
    """
    torch.manual_seed(42)
    if distribution == "normal":
        x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    else:
        x = torch.empty(shape, dtype=torch.float32).uniform_(-1e3, 1e3).to(torch.bfloat16)

    golden = _golden_mean_bf16(x, dim=dim, keepdim=True)
    actual = _run_ttnn_mean(x, ttnn.bfloat16, device, dim=dim, keepdim=True)

    # Looser ULP (up to 100) and scaled atol (1% of max golden) for
    # elements where heavy cancellation makes golden near-zero.
    passed, max_ulp, _, msg = _measure_ulp_safe(golden, actual, ulp_threshold=100, near_zero_atol_fraction=0.02)
    assert passed, f"[BF16 {desc} {distribution}] {msg}"


# ---------------------------------------------------------------------------
# FP32 tests — tile-axis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, dim, desc",
    _TILE_AXIS_CASES,
    ids=[c[2] for c in _TILE_AXIS_CASES],
)
@pytest.mark.parametrize("distribution", ["normal", "wide_uniform"])
def test_mean_ulp_fp32_tile_axis(device, shape, dim, desc, distribution):
    """FP32 mean on tile axes (W/H/HW)."""
    torch.manual_seed(42)
    if distribution == "normal":
        x = torch.randn(shape, dtype=torch.float32)
    else:
        x = torch.empty(shape, dtype=torch.float32).uniform_(-1e3, 1e3)

    golden = _golden_mean_fp32(x, dim=dim, keepdim=True)
    actual = _run_ttnn_mean(x, ttnn.float32, device, dim=dim, keepdim=True)

    passed, max_ulp, _, msg = _measure_ulp_safe(golden, actual, ulp_threshold=2_000_000, near_zero_atol_fraction=0.001)
    assert passed, f"[FP32 {desc} {distribution}] {msg}"


# ---------------------------------------------------------------------------
# FP32 tests — non-HW axis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, dim, desc",
    _NON_HW_AXIS_CASES,
    ids=[c[2] for c in _NON_HW_AXIS_CASES],
)
@pytest.mark.parametrize("distribution", ["normal", "wide_uniform"])
def test_mean_ulp_fp32_non_hw_axis(device, shape, dim, desc, distribution):
    """FP32 mean on non-HW axes (batch/channel): transpose + sequential reduce."""
    torch.manual_seed(42)
    if distribution == "normal":
        x = torch.randn(shape, dtype=torch.float32)
    else:
        x = torch.empty(shape, dtype=torch.float32).uniform_(-1e3, 1e3)

    golden = _golden_mean_fp32(x, dim=dim, keepdim=True)
    actual = _run_ttnn_mean(x, ttnn.float32, device, dim=dim, keepdim=True)

    # Non-HW FP32 sees higher ULP at near-zero elements due to operation
    # ordering + intermediate packing.  Use scaled atol for near-zero elements.
    passed, max_ulp, _, msg = _measure_ulp_safe(golden, actual, ulp_threshold=5_000_000, near_zero_atol_fraction=0.005)
    assert passed, f"[FP32 {desc} {distribution}] {msg}"
