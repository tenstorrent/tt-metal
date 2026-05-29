# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8 and per_token_cast_back.

Mirrors DeepEP's reference implementation (deepseek-ai/DeepEP math.py:31).

v0 simplifications:
  * per_token_cast_to_fp8 emits scale = 1.0 (no real amax).
  * per_token_cast_back ignores the scale tensor and performs a pure typecast.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import is_blackhole


E4M3_MAX = 448.0
GROUP_SIZE = 128

# ttnn.to_torch for FP8_E4M3 needs torch >= 2.8 (dlpack code 10 support).
_TORCH_HAS_FP8_DLPACK = tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 8)


@pytest.fixture(autouse=True)
def _require_blackhole():
    if not is_blackhole():
        pytest.skip("FP8_E4M3 path requires Blackhole")


def _skip_if_no_fp8_dlpack():
    if not _TORCH_HAS_FP8_DLPACK:
        pytest.skip("ttnn.to_torch of FP8_E4M3 requires torch >= 2.8")


def ref_to_fp8_v0(x: torch.Tensor):
    """v0: scale tensor is all 1.0; e4m3 is pure typecast of input."""
    e4m3 = x.float().to(torch.float8_e4m3fn)
    *leading, H = x.shape
    scale = torch.ones(*leading, H // GROUP_SIZE, dtype=torch.float32)
    return e4m3, scale


def ref_back_v0(e4m3: torch.Tensor, out_dtype: torch.dtype):
    """v0: ignore scale, just typecast."""
    return e4m3.float().to(out_dtype)


def _dtype_to_torch(ttnn_dtype):
    return {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}[ttnn_dtype]


def _fp8_from_torch_via_fp32(x_fp8_torch, *, device):
    """ttnn.from_torch with dtype=fp8_e4m3 requires float32 input. Cast first."""
    return ttnn.from_torch(x_fp8_torch.float(), dtype=ttnn.fp8_e4m3, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


# ---------- Skeleton dispatch tests (Milestone 2: value-free) ----------


class TestSkeletonDispatch:
    """Verify ops dispatch with correct output shape/dtype/layout. No value checks."""

    @pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
    @pytest.mark.parametrize("M, H", [(32, 128), (32, 256), (64, 7168)])
    def test_per_token_cast_to_fp8_specs(self, device, dtype, M, H):
        if dtype == ttnn.float32 and H >= 7168:
            pytest.skip("fp32 + large H exceeds L1 budget in v0")
        x = torch.zeros(M, H, dtype=_dtype_to_torch(dtype))
        x_tt = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        e4m3_tt, scale_tt = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x_tt)

        assert tuple(e4m3_tt.shape) == (M, H), f"e4m3 shape {tuple(e4m3_tt.shape)} != ({M}, {H})"
        assert tuple(scale_tt.shape) == (
            M,
            H // GROUP_SIZE,
        ), f"scale shape {tuple(scale_tt.shape)} != ({M}, {H // GROUP_SIZE})"
        assert e4m3_tt.dtype == ttnn.fp8_e4m3, f"e4m3 dtype is {e4m3_tt.dtype}"
        assert scale_tt.dtype == ttnn.float32, f"scale dtype is {scale_tt.dtype}"
        assert e4m3_tt.layout == ttnn.ROW_MAJOR_LAYOUT
        assert scale_tt.layout == ttnn.ROW_MAJOR_LAYOUT

    @pytest.mark.parametrize("out_dtype", [ttnn.bfloat16, ttnn.float32])
    @pytest.mark.parametrize("M, H", [(32, 128), (32, 256), (64, 7168)])
    def test_per_token_cast_back_specs(self, device, out_dtype, M, H):
        if out_dtype == ttnn.float32 and H >= 7168:
            pytest.skip("fp32 + large H exceeds L1 budget in v0")
        e4m3 = torch.zeros(M, H, dtype=torch.float8_e4m3fn)
        scale = torch.ones(M, H // GROUP_SIZE, dtype=torch.float32)
        e4m3_tt = _fp8_from_torch_via_fp32(e4m3, device=device)
        scale_tt = ttnn.from_torch(scale, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        out_tt = ttnn.experimental.deepseek_prefill.per_token_cast_back(e4m3_tt, scale_tt, output_dtype=out_dtype)

        assert tuple(out_tt.shape) == (M, H)
        assert out_tt.dtype == out_dtype
        assert out_tt.layout == ttnn.ROW_MAJOR_LAYOUT

    def test_per_token_cast_back_default_dtype(self, device):
        e4m3 = torch.zeros(32, 128, dtype=torch.float8_e4m3fn)
        scale = torch.ones(32, 1, dtype=torch.float32)
        e4m3_tt = _fp8_from_torch_via_fp32(e4m3, device=device)
        scale_tt = ttnn.from_torch(scale, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        out_tt = ttnn.experimental.deepseek_prefill.per_token_cast_back(e4m3_tt, scale_tt)
        assert out_tt.dtype == ttnn.bfloat16, "default output_dtype should be BFLOAT16"


# ---------- Value tests (Milestones 3 & 5+) ----------


class TestPerTokenCastToFp8:
    @pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
    @pytest.mark.parametrize("M, H", [(32, 128), (32, 256), (64, 7168)])
    def test_v0_scale_is_one(self, device, dtype, M, H):
        if dtype == ttnn.float32 and H >= 7168:
            pytest.skip("fp32 + large H exceeds L1 budget in v0 (kernel needs chunked tilize+pack_untilize)")
        torch.manual_seed(0)
        x = torch.randn(M, H, dtype=_dtype_to_torch(dtype))
        x_tt = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        _, scale_tt = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x_tt)
        scale_back = ttnn.to_torch(scale_tt)
        assert torch.all(scale_back == 1.0), "v0 scale tensor must be all 1.0"

    @pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
    def test_v0_e4m3_representable_inputs(self, device, dtype):
        _skip_if_no_fp8_dlpack()
        # Inputs that are exactly representable in e4m3 (powers of 2 in the normal range).
        values = torch.tensor([0.5, 1.0, 1.5, 2.0, -0.5, -1.0, 4.0, 8.0], dtype=_dtype_to_torch(dtype))
        x = values.repeat(32, 16)  # (32, 128)
        x_tt = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        e4m3_tt, _ = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x_tt)
        e4m3_back = ttnn.to_torch(e4m3_tt).float()
        e4m3_ref, _ = ref_to_fp8_v0(x)
        assert torch.equal(e4m3_back, e4m3_ref.float()), "e4m3 output mismatch on exactly-representable inputs"

    @pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
    @pytest.mark.parametrize("M, H", [(32, 128), (64, 256)])
    def test_v0_random_pcc(self, device, dtype, M, H):
        _skip_if_no_fp8_dlpack()
        torch.manual_seed(0)
        # Keep values within e4m3 normal range to avoid clipping noise.
        x = torch.randn(M, H, dtype=_dtype_to_torch(dtype)).clamp(-200.0, 200.0)
        x_tt = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        e4m3_tt, _ = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x_tt)
        e4m3_back = ttnn.to_torch(e4m3_tt).float()
        e4m3_ref, _ = ref_to_fp8_v0(x)
        assert_with_pcc(e4m3_back, e4m3_ref.float(), pcc=0.999)


class TestPerTokenCastBack:
    @pytest.mark.parametrize("out_dtype", [ttnn.bfloat16, ttnn.float32])
    def test_v0_e4m3_representable(self, device, out_dtype):
        _skip_if_no_fp8_dlpack()
        values = torch.tensor([0.5, 1.0, 1.5, 2.0, -0.5, -1.0, 4.0, 8.0], dtype=torch.float32)
        e4m3 = values.repeat(32, 16).to(torch.float8_e4m3fn)
        scale = torch.ones(32, 1, dtype=torch.float32)
        e4m3_tt = _fp8_from_torch_via_fp32(e4m3, device=device)
        scale_tt = ttnn.from_torch(scale, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        out_tt = ttnn.experimental.deepseek_prefill.per_token_cast_back(e4m3_tt, scale_tt, output_dtype=out_dtype)
        out_back = ttnn.to_torch(out_tt)
        out_ref = ref_back_v0(e4m3, _dtype_to_torch(out_dtype))
        assert torch.equal(out_back, out_ref)


class TestRoundTrip:
    @pytest.mark.parametrize("M, H", [(32, 128), (64, 256), (128, 7168)])
    def test_round_trip_representable(self, device, M, H):
        values = torch.tensor([0.5, 1.0, 1.5, 2.0, -0.5, -1.0, 4.0, 8.0], dtype=torch.bfloat16)
        n_repeat_h = H // len(values)
        x = values.repeat(M, n_repeat_h)
        x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        e4m3_tt, scale_tt = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x_tt)
        y_tt = ttnn.experimental.deepseek_prefill.per_token_cast_back(e4m3_tt, scale_tt)
        y = ttnn.to_torch(y_tt)
        y_ref = x.float().to(torch.float8_e4m3fn).float().to(torch.bfloat16)
        assert torch.equal(y, y_ref), "round-trip mismatch for representable values"

    @pytest.mark.parametrize("M, H", [(32, 128), (64, 256)])
    def test_round_trip_random(self, device, M, H):
        torch.manual_seed(0)
        x = torch.randn(M, H, dtype=torch.bfloat16).clamp(-100.0, 100.0)
        x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        e4m3_tt, scale_tt = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x_tt)
        y_tt = ttnn.experimental.deepseek_prefill.per_token_cast_back(e4m3_tt, scale_tt)
        y = ttnn.to_torch(y_tt).float()
        y_ref = x.float().to(torch.float8_e4m3fn).float()
        assert_with_pcc(y, y_ref, pcc=0.999)
