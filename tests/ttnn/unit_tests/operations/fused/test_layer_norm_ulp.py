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
from tests.ttnn.utils_for_testing import measure_ulp_with_near_zero_atol

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

# BH characterization stayed below ~14 ULP; keep modest margin.
_BF16_ULP_THRESHOLD = 24
_BF16_NEAR_ZERO_ATOL_FRACTION = 0.02

_SHAPES = [
    (32, 64, "32x64"),
    (37, 41, "37x41-odd"),
    (17, 33, "17x33-odd"),
    (128, 128, "128x128"),
    (256, 64, "256x64-wide"),
    (64, 256, "64x256-tall"),
    (1, 512, "1x512-vector"),
]


@pytest.mark.parametrize("h, w, desc", _SHAPES, ids=[c[2] for c in _SHAPES])
@pytest.mark.parametrize("use_welford", [True, False])
def test_layer_norm_ulp_bf16_no_weight_bias(device, h, w, desc, use_welford):
    """BF16 layer_norm ULP vs torch.nn.functional.layer_norm (no weight/bias)."""
    torch.manual_seed(0)
    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    golden = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w])
    actual = _run_ttnn_layer_norm(torch_input_tensor, device, use_welford)

    passed, max_ulp, max_atol_err, msg = measure_ulp_with_near_zero_atol(
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

    passed, max_ulp, max_atol_err, msg = measure_ulp_with_near_zero_atol(
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

# BH: ~1.28e6 on 32x64; larger mats + use_welford=False top ~1.73e6 (e.g. 64x256).
_FP32_ULP_THRESHOLD = 1_900_000
_FP32_NEAR_ZERO_ATOL_FRACTION = 0.005


@pytest.mark.parametrize("h, w, desc", _SHAPES, ids=[c[2] for c in _SHAPES])
@pytest.mark.parametrize("use_welford", [True, False])
def test_layer_norm_ulp_fp32_no_weight_bias(device, h, w, desc, use_welford):
    """FP32 layer_norm ULP vs torch float32 golden (no weight/bias)."""
    torch.manual_seed(0)
    torch_input_tensor = torch.rand((h, w), dtype=torch.float32)
    golden = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w])
    actual = _run_ttnn_layer_norm(torch_input_tensor, device, use_welford)

    passed, max_ulp, max_atol_err, msg = measure_ulp_with_near_zero_atol(
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
