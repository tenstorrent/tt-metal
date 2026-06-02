# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8 and per_token_cast_back.

Mirrors DeepEP's reference (deepseek-ai/DeepEP deep_ep/utils/math.py):
- per_token_cast_to_fp8: for each 128-element group of a token,
    scale = clamp(max(|x|), 1e-4) / 448, e4m3 = round(x / scale)
- per_token_cast_back: out = decode(e4m3) * scale

LLK notes reflected in the tolerances:
  * the e4m3 packer rounds toward zero (truncates the mantissa) vs torch's round-to-nearest, so the
    forward output is checked to be within one e4m3 ULP of the torch reference, not bit-equal;
  * the scale / divide run in fp32 on the FPU (operands truncated to ~19-bit), so scale and the
    dequant are checked with a small relative tolerance + PCC, not bit-equal.
Constraints: M % 32 == 0, H % 1024 == 0 (the LLK kernels use 1024-element column-blocks).
"""

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import comp_pcc, assert_equal


GROUP_SIZE = 128
E4M3_MAX = 448.0

# (M, H) shapes exercised by most tests.
# 7168 = 7 * 1024 = EMB_SIZE for both DeepSeek V3 and Kimi K2.6.
SHAPES = [
    (32, 1024),  # minimal
    (32, 2048),  # medium width
    (64, 1024),  # taller batch
    (32, 7168),  # DeepSeek V3 / Kimi K2.6 hidden dim
]

ROUNDTRIP_SHAPES = [(32, 1024), (64, 2048), (32, 7168)]


@pytest.fixture(autouse=True)
def _require_blackhole():
    if not is_blackhole():
        pytest.skip("FP8_E4M3 path requires Blackhole")


# ---------------------------------------------------------------------------
# Tensor creation helpers.
# ---------------------------------------------------------------------------


def _dtype_to_torch(ttnn_dtype):
    return {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}[ttnn_dtype]


def _make_zeros_input(device, *, M, H, dtype):
    x = torch.zeros(M, H, dtype=_dtype_to_torch(dtype))
    return ttnn.from_torch(
        x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _make_zeros_e4m3(device, *, M, H):
    return ttnn.from_torch(
        torch.zeros(M, H),
        dtype=ttnn.fp8_e4m3,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _make_ones_scale(device, *, M, H):
    return ttnn.from_torch(
        torch.ones(M, H // GROUP_SIZE),
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


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
    """Per-token reference scale [.., H/128] = clamp(amax over each 128-group, 1e-4) / 448."""
    *leading, H = x_fp32.shape
    grouped = x_fp32.reshape(*leading, H // GROUP_SIZE, GROUP_SIZE)
    amax = grouped.abs().amax(dim=-1).clamp(min=1e-4)
    return amax / E4M3_MAX


def _e4m3_ulp(ref):
    """Per-element e4m3 ULP (= binade / 8) for normal values."""
    return 2.0 ** (torch.floor(torch.log2(ref.abs().clamp_min(2.0**-9))) - 3)


def _decode_e4m3(t):
    """Decode an e4m3 device tensor to fp32 (ttnn.to_torch can't read fp8 on torch < 2.8)."""
    return ttnn.to_torch(ttnn.typecast(t, ttnn.float32)).float()


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
# Skeleton / dispatch tests: output shapes and dtypes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("M, H", SHAPES)
def test_cast_to_fp8_output_specs(device, dtype, M, H):
    x_tt = _make_zeros_input(device, M=M, H=H, dtype=dtype)
    e4m3_tt, scale_tt = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x_tt)

    assert tuple(e4m3_tt.shape) == (M, H)
    assert tuple(scale_tt.shape) == (M, H // GROUP_SIZE)
    assert e4m3_tt.dtype == ttnn.fp8_e4m3
    assert scale_tt.dtype == ttnn.float32
    assert e4m3_tt.layout == ttnn.ROW_MAJOR_LAYOUT
    assert scale_tt.layout == ttnn.ROW_MAJOR_LAYOUT


@pytest.mark.parametrize("out_dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("M, H", SHAPES)
def test_cast_back_output_specs(device, out_dtype, M, H):
    e4m3_tt = _make_zeros_e4m3(device, M=M, H=H)
    scale_tt = _make_ones_scale(device, M=M, H=H)
    out_tt = ttnn.experimental.deepseek_prefill.per_token_cast_back(e4m3_tt, scale_tt, output_dtype=out_dtype)

    assert tuple(out_tt.shape) == (M, H)
    assert out_tt.dtype == out_dtype
    assert out_tt.layout == ttnn.ROW_MAJOR_LAYOUT


def test_cast_back_default_dtype(device):
    e4m3_tt = _make_zeros_e4m3(device, M=32, H=1024)
    scale_tt = _make_ones_scale(device, M=32, H=1024)
    out_tt = ttnn.experimental.deepseek_prefill.per_token_cast_back(e4m3_tt, scale_tt)
    assert out_tt.dtype == ttnn.bfloat16, "default output_dtype should be BFLOAT16"


# ---------------------------------------------------------------------------
# per_token_cast_to_fp8: scale and quantization value tests.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_cast_to_fp8_scale(device, dtype):
    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)

    H = 1024
    M = 128 * 16
    # In this test, we want to verify that cast_to_fp8 scale is lossless if 128-consecutives values are the same
    # and if they can be represented exactly in fp8.
    # This generate values [-M/4, -M/4 + 0.25, ..., M/4 - 0.25] (centered around 0)
    # And for each value, repeat 128 times horizontally; and 1024 times vertically.
    STEP = 0.25
    MIN_VAL = -M / 2 * 0.25
    x = torch.tensor([(MIN_VAL + i * STEP) for i in range(M // 128)], dtype=torch_dtype).repeat([H, M])
    x_tt = ttnn.from_torch(
        x, dtype=ttnn_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    _, scale_tt = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x_tt)
    scale = ttnn.to_torch(scale_tt).float()

    ref = _ref_scale(x.float())
    assert_equal(scale, ref)


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("M, H", SHAPES)
def test_cast_to_fp8_scale_values(device, dtype, M, H):
    torch.manual_seed(0)

    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)

    x = (torch.randn(M, H) * 5.0).to(torch_dtype)
    x_tt = ttnn.from_torch(
        x, dtype=ttnn_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    _, scale_tt = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x_tt)
    scale = ttnn.to_torch(scale_tt).float()

    ref = _ref_scale(x.float())
    max_rel = ((scale - ref).abs() / ref.abs().clamp_min(1e-9)).max().item()
    logger.info(f"scale {dtype} M={M} H={H}: max_rel={max_rel:.4f}")
    assert_quality(scale, ref, pcc_threshold=0.999, rtol=1e-2, atol=1e-9, label=f"scale {dtype} M={M} H={H}")


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("M, H", SHAPES)
def test_cast_to_fp8_quantize_within_ulp(device, dtype, M, H):
    torch.manual_seed(0)
    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)

    x = (torch.randn(M, H) * 5.0).to(torch_dtype)
    x_in = x.float()
    x_tt = ttnn.from_torch(
        x, dtype=ttnn_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    e4m3_tt, scale_tt = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x_tt)
    y = _decode_e4m3(e4m3_tt)
    scale = ttnn.to_torch(scale_tt).float()

    ref_scale = _ref_scale(x_in)
    scale_exp = ref_scale.repeat_interleave(GROUP_SIZE, dim=1)
    ref_fp8 = (x_in / scale_exp).to(torch.float8_e4m3fn).float()

    normal = ref_fp8.abs() > 2.0**-6
    within = (y - ref_fp8).abs() <= _e4m3_ulp(ref_fp8) + 1e-6
    frac_within = within[normal].float().mean().item()
    logger.info(f"quant {dtype} M={M} H={H}: within_1ulp={frac_within:.4f}")
    assert frac_within >= 0.995, f"only {frac_within:.4f} of e4m3 values within one ULP of RNE ref"

    # Round-trip quality: reconstruction via actual device scale and decoded e4m3.
    # e4m3 has 3 mantissa bits (~12% worst-case relative error); use loose allclose.
    recon = y * scale.repeat_interleave(GROUP_SIZE, dim=1)
    assert_quality(recon, x_in, pcc_threshold=0.999, rtol=0.1, atol=0.2, label=f"quant roundtrip {dtype} M={M} H={H}")


# ---------------------------------------------------------------------------
# per_token_cast_back: dequantization value tests.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("out_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("M, H", SHAPES)
def test_cast_back_dequant(device, out_dtype, M, H):
    torch.manual_seed(0)
    torch_dtype = getattr(torch, out_dtype)
    ttnn_dtype = getattr(ttnn, out_dtype)

    e4m3 = (torch.randn(M, H) * 3.0).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    scale = (torch.rand(M, H // GROUP_SIZE) * 4.0 - 2.0).to(torch.float32)

    e4m3_tt = _make_e4m3_from_torch(e4m3, device=device)
    scale_tt = ttnn.from_torch(
        scale, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out_tt = ttnn.experimental.deepseek_prefill.per_token_cast_back(e4m3_tt, scale_tt, output_dtype=ttnn_dtype)
    out = ttnn.to_torch(out_tt).float()

    ref = e4m3.float() * scale.repeat_interleave(GROUP_SIZE, dim=1)
    if out_dtype == "bfloat16":
        ref = ref.to(torch_dtype).float()

    # Restrict to normal e4m3 values where relative tolerance is meaningful.
    normal = e4m3.float().abs() > 2.0**-6
    assert_quality(
        out[normal], ref[normal], pcc_threshold=0.999, rtol=1e-2, atol=1e-3, label=f"dequant {out_dtype} M={M} H={H}"
    )


# ---------------------------------------------------------------------------
# Round-trip: per_token_cast_to_fp8 followed by per_token_cast_back.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("M, H", ROUNDTRIP_SHAPES)
def test_round_trip_random(device, dtype, M, H):
    torch.manual_seed(0)
    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)

    x = (torch.randn(M, H) * 5.0).to(torch_dtype)
    x_in = x.float()
    x_tt = ttnn.from_torch(
        x, dtype=ttnn_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    e4m3_tt, scale_tt = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x_tt)
    y_tt = ttnn.experimental.deepseek_prefill.per_token_cast_back(e4m3_tt, scale_tt, output_dtype=ttnn.float32)
    y = ttnn.to_torch(y_tt).float()

    # fp8 quantization (~12% worst-case relative error) bounds the reconstruction.
    assert_quality(y, x_in, pcc_threshold=0.999, rtol=0.1, atol=0.2, label=f"roundtrip {dtype} M={M} H={H}")
