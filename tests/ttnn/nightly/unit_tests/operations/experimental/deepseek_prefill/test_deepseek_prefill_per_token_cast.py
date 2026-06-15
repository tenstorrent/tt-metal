# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8 and per_token_cast_back.

Mirrors DeepEP's reference (deepseek-ai/DeepEP deep_ep/utils/math.py):
- per_token_cast_to_fp8: for each 128-element block of a token,
    scale = clamp(max(|x|), 1e-4) / 448, e4m3 = round(x / scale)
- per_token_cast_back: out = decode(e4m3) * scale
"""

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import comp_pcc, assert_equal

pytestmark = pytest.mark.use_module_device


BLOCK_W = 128
E4M3_MAX = 448.0

SHAPES = [
    (1, 1024),  # single row (partial tile-row)
    (30, 1152),  # partial tile-row + 9 scale blocks
    (640, 7168),
    (3200, 7168),
    (6400, 7168),
    (2, 3, 30, 1152),  # 4D + partial tile-row (M = 180)
]

ROUNDTRIP_SHAPES = [(32, 1024), (30, 1152), (4, 1, 128, 1024)]


# Blackhole only operation
@pytest.fixture(autouse=True)
def _require_blackhole():
    if not is_blackhole():
        pytest.skip("FP8_E4M3 path requires Blackhole")


# ---------------------------------------------------------------------------
# Tensor creation helpers.
# ---------------------------------------------------------------------------


def _scale_shape(shape):
    """Expected scale shape for an input shape: leading dims preserved, last dim -> H / 128."""
    return tuple(shape[:-1]) + (shape[-1] // BLOCK_W,)


def _make_e4m3_from_torch(x_fp8_torch, *, device):
    """ttnn.from_torch with dtype=fp8_e4m3 requires float32 input. Cast first."""
    return ttnn.from_torch(
        x_fp8_torch.float(),
        dtype=ttnn.fp8_e4m3,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# ---------------------------------------------------------------------------
# Reference math (mirrors DeepEP's per_token_cast_to_fp8 / per_token_cast_back).
# ---------------------------------------------------------------------------


def _ref_scale(x_fp32):
    """Per-token reference scale [.., H/128] = clamp(amax over each 128-wide block, 1e-4) / 448."""
    *leading, H = x_fp32.shape
    blocks = x_fp32.reshape(*leading, H // BLOCK_W, BLOCK_W)
    amax = blocks.abs().amax(dim=-1).clamp(min=1e-4)
    return amax / E4M3_MAX


# ---------------------------------------------------------------------------
# Quality assertion helper.
# ---------------------------------------------------------------------------


def assert_quality(result, ref, *, pcc_threshold, rtol, atol, label=""):
    """Assert PCC >= pcc_threshold and torch.allclose(result, ref, rtol, atol), with logging."""
    passed_pcc, pcc = comp_pcc(result, ref, pcc_threshold)
    close = torch.allclose(result, ref, rtol=rtol, atol=atol)
    status = "PASS" if passed_pcc and close else "FAIL"
    logger.info(f"[{status}] {label}: pcc={pcc:.6f} (min={pcc_threshold}), allclose={close} (rtol={rtol}, atol={atol})")
    assert passed_pcc, f"{label}: PCC {pcc:.6f} below {pcc_threshold}"
    assert close, f"{label}: values not allclose (rtol={rtol}, atol={atol})"


# ---------------------------------------------------------------------------
# per_token_cast_to_fp8: scale and quantization value tests.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_cast_to_fp8_scale(device, dtype):
    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)

    M = 1024
    # In this test, we want to verify that cast_to_fp8 scale is lossless if 128-consecutives values are the same
    # and if they can be represented exactly in fp8.
    # Build one row with one exactly representable value per 128-wide block, then repeat it
    # vertically across M rows. These values also produce exactly representable power-of-two scales.
    block_values = torch.tensor([-448, -224, -112, -56, 56, 112, 224, 448], dtype=torch_dtype)
    x_row = block_values.repeat_interleave(BLOCK_W)
    x = x_row.repeat([M, 1])
    x_tt = ttnn.from_torch(
        x, dtype=ttnn_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    output_e4m3_tt, scale_tt = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x_tt)
    scale = ttnn.to_torch(scale_tt).float()

    ref = _ref_scale(x.float())
    assert_equal(scale, ref)


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("shape", SHAPES)
def test_cast_to_fp8_scale_values(device, dtype, shape):
    torch.manual_seed(0)

    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)

    x = (torch.randn(*shape) * 5.0).to(torch_dtype)
    x_tt = ttnn.from_torch(
        x, dtype=ttnn_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    output_e4m3_tt, scale_tt = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x_tt)
    scale = ttnn.to_torch(scale_tt).float()

    ref = _ref_scale(x.float())

    assert tuple(x_tt.shape) == shape
    assert tuple(scale_tt.shape) == _scale_shape(shape)
    assert tuple(output_e4m3_tt.shape) == shape
    assert output_e4m3_tt.dtype == ttnn.fp8_e4m3
    assert x_tt.dtype == ttnn_dtype
    assert scale_tt.dtype == ttnn.float32
    assert x_tt.layout == ttnn.ROW_MAJOR_LAYOUT
    assert output_e4m3_tt.layout == ttnn.ROW_MAJOR_LAYOUT
    assert scale_tt.layout == ttnn.ROW_MAJOR_LAYOUT

    max_rel = ((scale - ref).abs() / ref.abs().clamp_min(1e-9)).max().item()
    logger.info(f"scale {dtype} shape={shape}: max_rel={max_rel:.4f}")
    assert_quality(scale, ref, pcc_threshold=0.999, rtol=1e-2, atol=1e-9, label=f"scale {dtype} shape={shape}")


# ---------------------------------------------------------------------------
# per_token_cast_back: dequantization value tests.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("out_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("shape", SHAPES)
def test_cast_back_dequant(device, out_dtype, shape):
    torch.manual_seed(0)
    torch_dtype = getattr(torch, out_dtype)
    ttnn_dtype = getattr(ttnn, out_dtype)

    input_e4m3 = (torch.randn(*shape) * 3.0).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    input_scale = (torch.rand(*_scale_shape(shape)) * 4.0 - 2.0).to(torch.float32)

    e4m3_tt = _make_e4m3_from_torch(input_e4m3, device=device)
    scale_tt = ttnn.from_torch(
        input_scale,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_tt = ttnn.experimental.deepseek_prefill.per_token_cast_back(e4m3_tt, scale_tt, output_dtype=ttnn_dtype)
    out = ttnn.to_torch(out_tt).float()

    golden = input_e4m3.float() * input_scale.repeat_interleave(BLOCK_W, dim=-1)
    if out_dtype == "bfloat16":
        golden = golden.to(torch_dtype).float()

    assert tuple(out_tt.shape) == shape
    assert out_tt.dtype == ttnn_dtype
    assert out_tt.layout == ttnn.ROW_MAJOR_LAYOUT

    # Restrict to normal e4m3 values where relative tolerance is meaningful.
    normal = input_e4m3.float().abs() > 2.0**-6
    assert_quality(
        out[normal],
        golden[normal],
        pcc_threshold=0.999,
        rtol=1e-2,
        atol=1e-3,
        label=f"dequant {out_dtype} shape={shape}",
    )


# ---------------------------------------------------------------------------
# Round-trip: per_token_cast_to_fp8 followed by per_token_cast_back.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("shape", ROUNDTRIP_SHAPES)
def test_round_trip_random(device, dtype, shape):
    torch.manual_seed(0)
    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)

    x = (torch.randn(*shape) * 5.0).to(torch_dtype)
    x_in = x.float()
    x_tt = ttnn.from_torch(
        x, dtype=ttnn_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    e4m3_tt, scale_tt = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x_tt)
    y_tt = ttnn.experimental.deepseek_prefill.per_token_cast_back(e4m3_tt, scale_tt, output_dtype=ttnn.float32)
    y = ttnn.to_torch(y_tt).float()

    # fp8 quantization (~12% worst-case relative error) bounds the reconstruction.
    assert_quality(y, x_in, pcc_threshold=0.999, rtol=0.1, atol=0.2, label=f"roundtrip {dtype} shape={shape}")
