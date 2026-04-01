# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
ULP-based accuracy characterization for ttnn.layer_norm (issue #33749).

BF16 golden: torch.nn.functional.layer_norm on BF16 tensors (same dtype as device).

FP32 golden: torch.nn.functional.layer_norm in float32.

ULP is measured in the output dtype.  Near-zero golden elements use a scaled
absolute tolerance (same pattern as test_mean_ulp).

Log at INFO per parametrized case; e.g. pytest ... --log-cli-level=INFO
"""

import logging

import pytest

pytestmark = pytest.mark.use_module_device

import torch

import ttnn
from models.common.utility_functions import ulp as compute_ulp

logger = logging.getLogger(__name__)

# Poison value to ensure Welford's algorithm ignores padded elements (#31982)
PAD_VALUE = -42


def create_recip_tensor(device, w, use_welford):
    """Helper to create reciprocal tensor for non-sharded welford tests."""
    if not use_welford:
        return None
    grid = device.compute_with_storage_grid_size()
    core_range_set = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    return ttnn.create_layer_norm_reciprocals(device, core_range_set, w)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NEAR_ZERO_RELATIVE_FRACTION = 1e-2


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

    scaled_atol = near_zero_atol_fraction * golden_max

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


def _run_ttnn_layer_norm(
    torch_input_tensor: torch.Tensor,
    device,
    use_welford: bool,
    torch_weight: torch.Tensor | None = None,
    torch_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Same path as test_layer_norm: TILE, fill_implicit_tile_padding, optional weight/bias."""
    h, w = torch_input_tensor.shape[-2], torch_input_tensor.shape[-1]
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, PAD_VALUE)
    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    recip_tensor = create_recip_tensor(device, w, use_welford)
    if torch_weight is None and torch_bias is None:
        output_tensor = ttnn.layer_norm(input_tensor, program_config=program_config, recip_tensor=recip_tensor)
    else:
        weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)
        bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)
        output_tensor = ttnn.layer_norm(
            input_tensor,
            weight=weight,
            bias=bias,
            program_config=program_config,
            recip_tensor=recip_tensor,
        )
    output_tensor = ttnn.from_device(output_tensor)
    return ttnn.to_torch(output_tensor)


# ---------------------------------------------------------------------------
# BF16 tests
# ---------------------------------------------------------------------------

# Layer norm chains mean, variance, rsqrt, and elementwise ops; error compounds.
# Start conservative (256 ULP); tighten only if hardware + golden routinely measure lower.
_BF16_ULP_THRESHOLD = 256
_BF16_NEAR_ZERO_ATOL_FRACTION = 0.02

_SHAPES = [
    (32, 64, "32x64"),
    (37, 41, "37x41-odd"),
]


@pytest.mark.parametrize("h, w, desc", _SHAPES, ids=[c[2] for c in _SHAPES])
@pytest.mark.parametrize("use_welford", [True, False])
def test_layer_norm_ulp_bf16_no_weight_bias(device, h, w, desc, use_welford):
    """BF16 layer_norm ULP vs torch.nn.functional.layer_norm (no weight/bias)."""
    torch.manual_seed(0)
    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    golden = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w])
    actual = _run_ttnn_layer_norm(torch_input_tensor, device, use_welford)

    passed, max_ulp, max_atol_err, msg = _measure_ulp_safe(
        golden, actual, _BF16_ULP_THRESHOLD, _BF16_NEAR_ZERO_ATOL_FRACTION
    )
    logger.info(
        "ttnn.layer_norm ULP dtype=BF16 no_wb desc=%r shape=(%s,%s) use_welford=%s "
        "max_ulp=%s ulp_threshold=%s max_atol_err=%s passed=%s | %s",
        desc,
        h,
        w,
        use_welford,
        max_ulp,
        _BF16_ULP_THRESHOLD,
        max_atol_err,
        passed,
        msg,
    )
    assert passed, f"[BF16 no_wb {desc} use_welford={use_welford}] {msg}"


@pytest.mark.parametrize("h, w, desc", _SHAPES, ids=[c[2] for c in _SHAPES])
@pytest.mark.parametrize("use_welford", [True, False])
def test_layer_norm_ulp_bf16_with_weight_bias(device, h, w, desc, use_welford):
    """BF16 layer_norm ULP with weight and bias vs torch reference."""
    torch.manual_seed(0)
    dtype = torch.bfloat16
    torch_input_tensor = torch.rand((h, w), dtype=dtype)
    torch_weight = torch.rand((w,), dtype=dtype)
    torch_bias = torch.rand((w,), dtype=dtype)
    golden = torch.nn.functional.layer_norm(
        torch_input_tensor, normalized_shape=[w], weight=torch_weight, bias=torch_bias
    )
    actual = _run_ttnn_layer_norm(
        torch_input_tensor, device, use_welford, torch_weight=torch_weight, torch_bias=torch_bias
    )

    passed, max_ulp, max_atol_err, msg = _measure_ulp_safe(
        golden, actual, _BF16_ULP_THRESHOLD, _BF16_NEAR_ZERO_ATOL_FRACTION
    )
    logger.info(
        "ttnn.layer_norm ULP dtype=BF16 weight+bias desc=%r shape=(%s,%s) use_welford=%s "
        "max_ulp=%s ulp_threshold=%s max_atol_err=%s passed=%s | %s",
        desc,
        h,
        w,
        use_welford,
        max_ulp,
        _BF16_ULP_THRESHOLD,
        max_atol_err,
        passed,
        msg,
    )
    assert passed, f"[BF16 weight+bias {desc} use_welford={use_welford}] {msg}"


# ---------------------------------------------------------------------------
# FP32 tests
# ---------------------------------------------------------------------------

# Device vs torch both FP32 but ordering differs (tiling, fused stages).
# Threshold is intentionally very high until characterized on hardware.
_FP32_ULP_THRESHOLD = 50_000_000
_FP32_NEAR_ZERO_ATOL_FRACTION = 0.005


@pytest.mark.parametrize("h, w, desc", _SHAPES, ids=[c[2] for c in _SHAPES])
@pytest.mark.parametrize("use_welford", [True, False])
def test_layer_norm_ulp_fp32_no_weight_bias(device, h, w, desc, use_welford):
    """FP32 layer_norm ULP vs torch float32 golden (no weight/bias)."""
    torch.manual_seed(0)
    torch_input_tensor = torch.rand((h, w), dtype=torch.float32)
    golden = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w])
    actual = _run_ttnn_layer_norm(torch_input_tensor, device, use_welford)

    passed, max_ulp, max_atol_err, msg = _measure_ulp_safe(
        golden, actual, _FP32_ULP_THRESHOLD, _FP32_NEAR_ZERO_ATOL_FRACTION
    )
    logger.info(
        "ttnn.layer_norm ULP dtype=FP32 no_wb desc=%r shape=(%s,%s) use_welford=%s "
        "max_ulp=%s ulp_threshold=%s max_atol_err=%s passed=%s | %s",
        desc,
        h,
        w,
        use_welford,
        max_ulp,
        _FP32_ULP_THRESHOLD,
        max_atol_err,
        passed,
        msg,
    )
    assert passed, f"[FP32 no_wb {desc} use_welford={use_welford}] {msg}"
