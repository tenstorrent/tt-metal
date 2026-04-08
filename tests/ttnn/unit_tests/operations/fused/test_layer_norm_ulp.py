# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
ULP-based accuracy characterization for ttnn.layer_norm (issue #33749).

BF16 golden: torch.nn.functional.layer_norm on BF16 tensors (same dtype as device).

FP32 golden: torch.nn.functional.layer_norm in float32.

ULP is measured in the output dtype.  Near-zero golden elements use a scaled
absolute tolerance (same pattern as test_mean_ulp).

Loguru logs at INFO per parametrized case (same pattern as other ``tests/ttnn`` tests).
"""

import pytest

pytestmark = pytest.mark.use_module_device

import torch
from loguru import logger

import ttnn
from tests.ttnn.unit_tests.operations.test_utils import get_compute_kernel_options
from tests.ttnn.utils_for_testing import measure_ulp_with_near_zero_atol

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


def _make_ln_compute_kernel_config(device, fp32_dest_acc_en: bool):
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


def _make_ln_input(h: int, w: int, dtype: torch.dtype, distribution: str) -> torch.Tensor:
    """Generate a (h, w) layer-norm input for the requested distribution."""
    if distribution == "normal":
        return torch.randn((h, w), dtype=torch.float32).to(dtype)
    if distribution == "wide_uniform":
        return torch.empty((h, w), dtype=torch.float32).uniform_(-1e3, 1e3).to(dtype)
    # uniform_01 (default, matches original torch.rand behaviour)
    return torch.rand((h, w), dtype=torch.float32).to(dtype)


def _run_ttnn_layer_norm(
    torch_input_tensor: torch.Tensor,
    device,
    use_welford: bool,
    torch_weight: torch.Tensor | None = None,
    torch_bias: torch.Tensor | None = None,
    compute_kernel_config=None,
) -> torch.Tensor:
    """Same path as test_layer_norm: TILE, fill_implicit_tile_padding, optional weight/bias.

    Supports all four wb combinations: no_wb, weight_only, bias_only, and wb.
    Each non-None tensor is converted individually; None parameters are omitted
    from the ttnn.layer_norm call (not passed as explicit None) so the C++ binding
    uses its own default, matching the no-wb code path for missing parameters.
    """
    h, w = torch_input_tensor.shape[-2], torch_input_tensor.shape[-1]
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, PAD_VALUE)
    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    recip_tensor = create_recip_tensor(device, w, use_welford)
    ln_kwargs: dict = dict(program_config=program_config, recip_tensor=recip_tensor)
    if compute_kernel_config is not None:
        ln_kwargs["compute_kernel_config"] = compute_kernel_config
    if torch_weight is not None:
        ln_kwargs["weight"] = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)
    if torch_bias is not None:
        ln_kwargs["bias"] = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.layer_norm(input_tensor, **ln_kwargs)
    output_tensor = ttnn.from_device(output_tensor)
    return ttnn.to_torch(output_tensor)


# ---------------------------------------------------------------------------
# BF16 tests
# ---------------------------------------------------------------------------

# BF16 max-ULP cap vs torch golden.
# fp32_dest_acc_en=True (BF16 inputs → FP32 accumulation → BF16 out): tight tail ~26 ULP.
# fp32_dest_acc_en=False (BF16 accumulation throughout): wider tail, and near-zero normalized
# outputs see ~9x more absolute error due to BF16 variance/mean rounding.
# This demonstrates that fp32 accumulation is essential for BF16 layer_norm accuracy.
_BF16_ULP_THRESHOLD_FP32_DEST = 32
_BF16_ULP_THRESHOLD_BF16_DEST = 1000  # ~3x headroom over observed peak 330 ULP
_BF16_NEAR_ZERO_ATOL_FRACTION_FP32_DEST = 0.002  # tight; fp32 rounding error is tiny
_BF16_NEAR_ZERO_ATOL_FRACTION_BF16_DEST = 0.06  # loose; BF16 accum can be ~2% of range on near-zero output

_LN_W_SIZES = sorted(set(range(32, 513, 64)) | {33, 41, 512, 768, 1024, 2048})
_LN_H_SIZES = sorted(set(range(32, 257, 64)) | {17, 37, 128, 256})
_LN_SQUARES = list(range(64, 257, 64))
_LN_H_FIXED = 32
_LN_W_FIXED = 64


def _build_layer_norm_shapes():
    seen = set()
    rows = []

    def add(h, w, tag):
        key = (h, w)
        if key in seen:
            return
        seen.add(key)
        rows.append((h, w, f"{h}x{w}-{tag}"))

    for w in _LN_W_SIZES:
        add(_LN_H_FIXED, w, "Wsweep")
    for h in _LN_H_SIZES:
        add(h, _LN_W_FIXED, "Hsweep")
    for s in _LN_SQUARES:
        add(s, s, "square")
    add(37, 41, "odd")
    add(17, 33, "odd")
    add(1, 512, "vector")
    add(64, 256, "tall")
    add(256, 64, "wide")
    return rows


_SHAPES = _build_layer_norm_shapes()


@pytest.mark.parametrize("h, w, desc", _SHAPES, ids=[c[2] for c in _SHAPES])
@pytest.mark.parametrize("use_welford", [True, False])
@pytest.mark.parametrize("distribution", ["uniform_01", "normal", "wide_uniform"])
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True], ids=["fp32_acc_off", "fp32_acc_on"])
def test_layer_norm_ulp_bf16_no_weight_bias(device, h, w, desc, use_welford, distribution, fp32_dest_acc_en):
    """BF16 layer_norm ULP vs torch.nn.functional.layer_norm (no weight/bias).

    fp32_dest_acc_en=True reflects the recommended path (BF16 inputs accumulated in FP32).
    fp32_dest_acc_en=False documents the accuracy cost of BF16-only accumulation.
    """
    torch.manual_seed(0)
    torch_input_tensor = _make_ln_input(h, w, torch.bfloat16, distribution)
    golden = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w])
    ckc = _make_ln_compute_kernel_config(device, fp32_dest_acc_en)
    actual = _run_ttnn_layer_norm(torch_input_tensor, device, use_welford, compute_kernel_config=ckc)

    ulp_threshold = _BF16_ULP_THRESHOLD_FP32_DEST if fp32_dest_acc_en else _BF16_ULP_THRESHOLD_BF16_DEST
    atol_fraction = (
        _BF16_NEAR_ZERO_ATOL_FRACTION_FP32_DEST if fp32_dest_acc_en else _BF16_NEAR_ZERO_ATOL_FRACTION_BF16_DEST
    )
    passed, max_ulp, max_atol_err, atol_tol, msg, ulp_stats = measure_ulp_with_near_zero_atol(
        golden, actual, ulp_threshold, atol_fraction
    )
    spec = f"{desc} shape_hw=({h},{w}) welford={use_welford} dist={distribution} fp32_acc={fp32_dest_acc_en}"
    logger.info(
        f"ttnn.layer_norm ULP (BF16, no wb) | {spec} | ulp mean={ulp_stats['mean']:.3g} p95={ulp_stats['p95']:.3g} p99={ulp_stats['p99']:.3g} max={max_ulp:.4g}/{ulp_threshold} atol {max_atol_err:.4g}/{atol_tol:.4g} | {'ok' if passed else 'FAIL'}"
    )
    if ulp_stats["worst"]:
        logger.info(f"  worst: {ulp_stats['worst']}")
    if not passed:
        logger.info(f"  {msg}")
    assert (
        passed
    ), f"[BF16 no_wb {desc} use_welford={use_welford} dist={distribution} fp32_acc={fp32_dest_acc_en}] {msg}"


@pytest.mark.parametrize("h, w, desc", _SHAPES, ids=[c[2] for c in _SHAPES])
@pytest.mark.parametrize("use_welford", [True, False])
@pytest.mark.parametrize("distribution", ["uniform_01", "normal", "wide_uniform"])
@pytest.mark.parametrize("wb_mode", ["wb", "weight_only", "bias_only"])
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True], ids=["fp32_acc_off", "fp32_acc_on"])
def test_layer_norm_ulp_bf16_with_weight_bias(device, h, w, desc, use_welford, distribution, wb_mode, fp32_dest_acc_en):
    """BF16 layer_norm ULP with weight/bias variants vs torch reference.

    fp32_dest_acc_en=True reflects the recommended path (BF16 inputs accumulated in FP32).
    fp32_dest_acc_en=False documents the accuracy cost of BF16-only accumulation.

    wb_mode controls which affine parameters are active:
      "wb"          – both weight and bias (original coverage)
      "weight_only" – weight only, bias=None
      "bias_only"   – bias only, weight=None
    """
    torch.manual_seed(0)
    dtype = torch.bfloat16
    torch_input_tensor = _make_ln_input(h, w, dtype, distribution)
    torch_weight = torch.rand((w,), dtype=dtype) if wb_mode in ("wb", "weight_only") else None
    torch_bias = torch.rand((w,), dtype=dtype) if wb_mode in ("wb", "bias_only") else None
    golden = torch.nn.functional.layer_norm(
        torch_input_tensor, normalized_shape=[w], weight=torch_weight, bias=torch_bias
    )
    ckc = _make_ln_compute_kernel_config(device, fp32_dest_acc_en)
    actual = _run_ttnn_layer_norm(
        torch_input_tensor,
        device,
        use_welford,
        torch_weight=torch_weight,
        torch_bias=torch_bias,
        compute_kernel_config=ckc,
    )

    ulp_threshold = _BF16_ULP_THRESHOLD_FP32_DEST if fp32_dest_acc_en else _BF16_ULP_THRESHOLD_BF16_DEST
    atol_fraction = (
        _BF16_NEAR_ZERO_ATOL_FRACTION_FP32_DEST if fp32_dest_acc_en else _BF16_NEAR_ZERO_ATOL_FRACTION_BF16_DEST
    )
    passed, max_ulp, max_atol_err, atol_tol, msg, ulp_stats = measure_ulp_with_near_zero_atol(
        golden, actual, ulp_threshold, atol_fraction
    )
    spec = (
        f"{desc} shape_hw=({h},{w}) welford={use_welford} dist={distribution} wb={wb_mode} fp32_acc={fp32_dest_acc_en}"
    )
    logger.info(
        f"ttnn.layer_norm ULP (BF16, {wb_mode}) | {spec} | ulp mean={ulp_stats['mean']:.3g} p95={ulp_stats['p95']:.3g} p99={ulp_stats['p99']:.3g} max={max_ulp:.4g}/{ulp_threshold} atol {max_atol_err:.4g}/{atol_tol:.4g} | {'ok' if passed else 'FAIL'}"
    )
    if ulp_stats["worst"]:
        logger.info(f"  worst: {ulp_stats['worst']}")
    if not passed:
        logger.info(f"  {msg}")
    assert (
        passed
    ), f"[BF16 {wb_mode} {desc} use_welford={use_welford} dist={distribution} fp32_acc={fp32_dest_acc_en}] {msg}"


# ---------------------------------------------------------------------------
# FP32 tests
# ---------------------------------------------------------------------------

# FP32 max-ULP cap; no_wb peak ~1.92e6, wb peak ~2.31e6 (both on BH shape sweeps).
# Single threshold covers both variants with comfortable margin.
_FP32_ULP_THRESHOLD = 3_000_000
_FP32_NEAR_ZERO_ATOL_FRACTION = 0.004


@pytest.mark.parametrize("h, w, desc", _SHAPES, ids=[c[2] for c in _SHAPES])
@pytest.mark.parametrize("use_welford", [True, False])
@pytest.mark.parametrize("distribution", ["uniform_01", "normal", "wide_uniform"])
def test_layer_norm_ulp_fp32_no_weight_bias(device, h, w, desc, use_welford, distribution):
    """FP32 layer_norm ULP vs torch float32 golden (no weight/bias); fp32_dest_acc_en=True only."""
    torch.manual_seed(0)
    torch_input_tensor = _make_ln_input(h, w, torch.float32, distribution)
    golden = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w])
    ckc = _make_ln_compute_kernel_config(device, fp32_dest_acc_en=True)
    actual = _run_ttnn_layer_norm(torch_input_tensor, device, use_welford, compute_kernel_config=ckc)

    passed, max_ulp, max_atol_err, atol_tol, msg, ulp_stats = measure_ulp_with_near_zero_atol(
        golden, actual, _FP32_ULP_THRESHOLD, _FP32_NEAR_ZERO_ATOL_FRACTION
    )
    spec = f"{desc} shape_hw=({h},{w}) welford={use_welford} dist={distribution}"
    logger.info(
        f"ttnn.layer_norm ULP (FP32, no wb) | {spec} | ulp mean={ulp_stats['mean']:.3g} p95={ulp_stats['p95']:.3g} p99={ulp_stats['p99']:.3g} max={max_ulp:.4g}/{_FP32_ULP_THRESHOLD} atol {max_atol_err:.4g}/{atol_tol:.4g} | {'ok' if passed else 'FAIL'}"
    )
    if ulp_stats["worst"]:
        logger.info(f"  worst: {ulp_stats['worst']}")
    if not passed:
        logger.info(f"  {msg}")
    assert passed, f"[FP32 no_wb {desc} use_welford={use_welford} dist={distribution}] {msg}"


@pytest.mark.parametrize("h, w, desc", _SHAPES, ids=[c[2] for c in _SHAPES])
@pytest.mark.parametrize("use_welford", [True, False])
@pytest.mark.parametrize("distribution", ["uniform_01", "normal", "wide_uniform"])
@pytest.mark.parametrize("wb_mode", ["wb", "weight_only", "bias_only"])
def test_layer_norm_ulp_fp32_with_weight_bias(device, h, w, desc, use_welford, distribution, wb_mode):
    """FP32 layer_norm ULP with weight/bias variants vs torch float32 golden; fp32_dest_acc_en=True only.

    wb_mode controls which affine parameters are active:
      "wb"          – both weight and bias
      "weight_only" – weight only, bias=None
      "bias_only"   – bias only, weight=None
    """
    torch.manual_seed(0)
    torch_input_tensor = _make_ln_input(h, w, torch.float32, distribution)
    torch_weight = torch.rand((w,), dtype=torch.float32) if wb_mode in ("wb", "weight_only") else None
    torch_bias = torch.rand((w,), dtype=torch.float32) if wb_mode in ("wb", "bias_only") else None
    golden = torch.nn.functional.layer_norm(
        torch_input_tensor, normalized_shape=[w], weight=torch_weight, bias=torch_bias
    )
    ckc = _make_ln_compute_kernel_config(device, fp32_dest_acc_en=True)
    actual = _run_ttnn_layer_norm(
        torch_input_tensor,
        device,
        use_welford,
        torch_weight=torch_weight,
        torch_bias=torch_bias,
        compute_kernel_config=ckc,
    )

    passed, max_ulp, max_atol_err, atol_tol, msg, ulp_stats = measure_ulp_with_near_zero_atol(
        golden, actual, _FP32_ULP_THRESHOLD, _FP32_NEAR_ZERO_ATOL_FRACTION
    )
    spec = f"{desc} shape_hw=({h},{w}) welford={use_welford} dist={distribution} wb={wb_mode}"
    logger.info(
        f"ttnn.layer_norm ULP (FP32, {wb_mode}) | {spec} | ulp mean={ulp_stats['mean']:.3g} p95={ulp_stats['p95']:.3g} p99={ulp_stats['p99']:.3g} max={max_ulp:.4g}/{_FP32_ULP_THRESHOLD} atol {max_atol_err:.4g}/{atol_tol:.4g} | {'ok' if passed else 'FAIL'}"
    )
    if ulp_stats["worst"]:
        logger.info(f"  worst: {ulp_stats['worst']}")
    if not passed:
        logger.info(f"  {msg}")
    assert passed, f"[FP32 {wb_mode} {desc} use_welford={use_welford} dist={distribution}] {msg}"
