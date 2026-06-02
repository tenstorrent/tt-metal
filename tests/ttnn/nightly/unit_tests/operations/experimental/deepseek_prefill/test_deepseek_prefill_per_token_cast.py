# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8 and per_token_cast_back.

Mirrors DeepEP's reference (deepseek-ai/DeepEP math.py): for each 128-element group of a token,
scale = clamp(max(|x|), 1e-4) / 448 and e4m3 = round(x / scale); per_token_cast_back recovers
out = decode(e4m3) * scale.

LLK notes reflected in the tolerances:
  * the e4m3 packer rounds toward zero (truncates the mantissa) vs torch's round-to-nearest, so the
    forward output is checked to be within one e4m3 ULP of the torch reference, not bit-equal;
  * the scale / divide run in fp32 on the FPU (operands truncated to ~19-bit), so scale and the
    dequant are checked with a small relative tolerance + cosine, not bit-equal.
Constraints: M % 32 == 0, H % 1024 == 0 (the LLK kernels use 1024-element column-blocks).
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole

GROUP_SIZE = 128
E4M3_MAX = 448.0
SHAPES = [(32, 1024), (32, 2048), (64, 1024)]


@pytest.fixture(autouse=True)
def _require_blackhole():
    if not is_blackhole():
        pytest.skip("FP8_E4M3 path requires Blackhole")


def _dtype_to_torch(ttnn_dtype):
    return {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}[ttnn_dtype]


def _fp8_from_torch_via_fp32(x_fp8_torch, *, device):
    """ttnn.from_torch with dtype=fp8_e4m3 requires float32 input. Cast first."""
    return ttnn.from_torch(x_fp8_torch.float(), dtype=ttnn.fp8_e4m3, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def _read_e4m3(t):
    """Decode an e4m3 device tensor to fp32 (ttnn.to_torch can't read fp8 on torch < 2.8)."""
    return ttnn.to_torch(ttnn.typecast(t, ttnn.float32)).float()


def _ref_scale(x_fp32):
    """Per-token reference scale [.., H/128] = clamp(amax over each 128-group, 1e-4) / 448."""
    *leading, H = x_fp32.shape
    grouped = x_fp32.reshape(*leading, H // GROUP_SIZE, GROUP_SIZE)
    amax = grouped.abs().amax(dim=-1).clamp(min=1e-4)
    return amax / E4M3_MAX


def _e4m3_ulp(ref):
    """Per-element e4m3 ULP (= binade / 8) for normal values."""
    return 2.0 ** (torch.floor(torch.log2(ref.abs().clamp_min(2.0**-9))) - 3)


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()


# ---------- Skeleton dispatch tests (shape / dtype / layout) ----------


class TestSkeletonDispatch:
    @pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
    @pytest.mark.parametrize("M, H", SHAPES)
    def test_per_token_cast_to_fp8_specs(self, device, dtype, M, H):
        x = torch.zeros(M, H, dtype=_dtype_to_torch(dtype))
        x_tt = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        e4m3_tt, scale_tt = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x_tt)

        assert tuple(e4m3_tt.shape) == (M, H)
        assert tuple(scale_tt.shape) == (M, H // GROUP_SIZE)
        assert e4m3_tt.dtype == ttnn.fp8_e4m3
        assert scale_tt.dtype == ttnn.float32
        assert e4m3_tt.layout == ttnn.ROW_MAJOR_LAYOUT
        assert scale_tt.layout == ttnn.ROW_MAJOR_LAYOUT

    @pytest.mark.parametrize("out_dtype", [ttnn.bfloat16, ttnn.float32])
    @pytest.mark.parametrize("M, H", SHAPES)
    def test_per_token_cast_back_specs(self, device, out_dtype, M, H):
        e4m3 = torch.zeros(M, H, dtype=torch.float8_e4m3fn)
        scale = torch.ones(M, H // GROUP_SIZE, dtype=torch.float32)
        e4m3_tt = _fp8_from_torch_via_fp32(e4m3, device=device)
        scale_tt = ttnn.from_torch(scale, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        out_tt = ttnn.experimental.deepseek_prefill.per_token_cast_back(e4m3_tt, scale_tt, output_dtype=out_dtype)

        assert tuple(out_tt.shape) == (M, H)
        assert out_tt.dtype == out_dtype
        assert out_tt.layout == ttnn.ROW_MAJOR_LAYOUT

    def test_per_token_cast_back_default_dtype(self, device):
        e4m3 = torch.zeros(32, 1024, dtype=torch.float8_e4m3fn)
        scale = torch.ones(32, 1024 // GROUP_SIZE, dtype=torch.float32)
        e4m3_tt = _fp8_from_torch_via_fp32(e4m3, device=device)
        scale_tt = ttnn.from_torch(scale, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        out_tt = ttnn.experimental.deepseek_prefill.per_token_cast_back(e4m3_tt, scale_tt)
        assert out_tt.dtype == ttnn.bfloat16, "default output_dtype should be BFLOAT16"


# ---------- per_token_cast_to_fp8 value tests ----------


class TestPerTokenCastToFp8:
    @pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
    @pytest.mark.parametrize("M, H", SHAPES)
    def test_scale_values(self, device, dtype, M, H):
        torch.manual_seed(0)
        x = (torch.randn(M, H) * 5.0).to(_dtype_to_torch(dtype))
        x_in = x.float()  # the device sees the dtype-rounded input; bf16 rounding already applied above
        x_tt = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        _, scale_tt = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x_tt)
        scale = ttnn.to_torch(scale_tt).float()
        ref = _ref_scale(x_in)
        max_rel = ((scale - ref).abs() / ref.abs().clamp_min(1e-9)).max().item()
        print(f"\n[scale {dtype} M={M} H={H}] max_rel={max_rel:.4f}")
        assert torch.allclose(scale, ref, rtol=2e-2, atol=1e-9), f"scale mismatch (max_rel={max_rel:.4f})"

    @pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
    @pytest.mark.parametrize("M, H", SHAPES)
    def test_quantize_within_ulp(self, device, dtype, M, H):
        torch.manual_seed(0)
        x = (torch.randn(M, H) * 5.0).to(_dtype_to_torch(dtype))
        x_in = x.float()
        x_tt = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        e4m3_tt, scale_tt = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x_tt)
        y = _read_e4m3(e4m3_tt)
        scale = ttnn.to_torch(scale_tt).float()

        ref_scale = _ref_scale(x_in)
        scale_exp = ref_scale.repeat_interleave(GROUP_SIZE, dim=1)
        ref_fp8 = (x_in / scale_exp).to(torch.float8_e4m3fn).float()

        normal = ref_fp8.abs() > 2.0**-6
        within = (y - ref_fp8).abs() <= _e4m3_ulp(ref_fp8) + 1e-6
        frac_within = within[normal].float().mean().item()
        recon = y * scale.repeat_interleave(GROUP_SIZE, dim=1)
        rt_cos = _cos(recon, x_in)
        print(f"\n[quant {dtype} M={M} H={H}] within_1ulp={frac_within:.4f}, roundtrip_cos={rt_cos:.5f}")
        assert frac_within >= 0.995, f"only {frac_within:.4f} of e4m3 values within one ULP of RNE ref"
        assert rt_cos > 0.999, f"round-trip cosine {rt_cos:.5f} below 0.999"


# ---------- per_token_cast_back value tests ----------


class TestPerTokenCastBack:
    @pytest.mark.parametrize("out_dtype", [ttnn.bfloat16, ttnn.float32])
    @pytest.mark.parametrize("M, H", SHAPES)
    def test_dequant(self, device, out_dtype, M, H):
        torch.manual_seed(0)
        e4m3 = (torch.randn(M, H) * 3.0).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        scale = (torch.rand(M, H // GROUP_SIZE) * 4.0 - 2.0).to(torch.float32)

        e4m3_tt = _fp8_from_torch_via_fp32(e4m3, device=device)
        scale_tt = ttnn.from_torch(scale, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        out_tt = ttnn.experimental.deepseek_prefill.per_token_cast_back(e4m3_tt, scale_tt, output_dtype=out_dtype)
        out = ttnn.to_torch(out_tt).float()

        ref = e4m3.float() * scale.repeat_interleave(GROUP_SIZE, dim=1)
        if out_dtype == ttnn.bfloat16:
            ref = ref.to(torch.bfloat16).float()
        normal = e4m3.float().abs() > 2.0**-6
        cos = _cos(out, ref)
        close = torch.allclose(out[normal], ref[normal], rtol=1e-2, atol=1e-3)
        print(f"\n[dequant {out_dtype} M={M} H={H}] cos={cos:.6f}, allclose={close}")
        assert cos > 0.999, f"dequant cosine {cos:.6f} below 0.999"
        assert close, "dequant mismatch on normal values"


class TestRoundTrip:
    @pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
    @pytest.mark.parametrize("M, H", [(32, 1024), (64, 2048)])
    def test_round_trip_random(self, device, dtype, M, H):
        torch.manual_seed(0)
        x = (torch.randn(M, H) * 5.0).to(_dtype_to_torch(dtype))
        x_in = x.float()
        x_tt = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        e4m3_tt, scale_tt = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x_tt)
        y_tt = ttnn.experimental.deepseek_prefill.per_token_cast_back(e4m3_tt, scale_tt, output_dtype=ttnn.float32)
        y = ttnn.to_torch(y_tt).float()
        cos = _cos(y, x_in)
        print(f"\n[roundtrip {dtype} M={M} H={H}] cos={cos:.6f}")
        # fp8 quantization (~6% per-element steps) bounds the reconstruction; cosine stays very high.
        assert cos > 0.998, f"round-trip cosine {cos:.6f} below 0.998"
