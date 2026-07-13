# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
    SparseKVCache,
    SparseKVCacheFormat,
    reconstruct_scaled_fp8_kv_cache,
    unpack_scaled_fp8_kv_cache,
)


def _tensor(shape, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    return SimpleNamespace(shape=shape, dtype=dtype, layout=layout)


def test_bf16_sparse_kv_contract_retains_combined_cache():
    kvpe = _tensor((78, 1, 1024, 576), ttnn.bfloat16)
    cache = SparseKVCache(SparseKVCacheFormat.BF16, tensor=kvpe)

    assert cache.tensor is kvpe


def test_scaled_fp8_sparse_kv_contract_owns_one_mixed_format_cache():
    packed = _tensor((78, 1, 1024, 656), ttnn.fp8_e4m3)
    cache = SparseKVCache(SparseKVCacheFormat.SCALED_FP8, tensor=packed)

    assert cache.tensor is packed
    assert SparseKVCache.PACKED_ROW_BYTES == 656
    assert SparseKVCache.PACKED_ROW_BYTES < 0.59 * (SparseKVCache.LATENT_DIM + SparseKVCache.ROPE_DIM) * 2


def test_scaled_fp8_packed_host_codec_preserves_mixed_fields():
    latent = torch.tensor([[0.5, -1.0, 2.0, -4.0]], dtype=torch.float8_e4m3fn).repeat(1, 128)
    scales = torch.tensor([[0.25, 0.5, 1.0, 2.0]], dtype=torch.float32)
    rope = torch.arange(64, dtype=torch.float32).to(torch.bfloat16).unsqueeze(0)
    raw = torch.cat(
        (latent.view(torch.uint8), scales.view(torch.uint8), rope.view(torch.uint8)),
        dim=-1,
    )
    packed = raw.view(torch.float8_e4m3fn)

    decoded_latent, decoded_scales, decoded_rope = unpack_scaled_fp8_kv_cache(packed)
    torch.testing.assert_close(decoded_latent, latent.float(), rtol=0, atol=0)
    torch.testing.assert_close(decoded_scales, scales, rtol=0, atol=0)
    torch.testing.assert_close(decoded_rope, rope, rtol=0, atol=0)

    expected_latent = latent.float() * scales.repeat_interleave(128, dim=-1)
    expected = torch.cat((expected_latent.to(torch.bfloat16), rope), dim=-1)
    torch.testing.assert_close(reconstruct_scaled_fp8_kv_cache(packed), expected, rtol=0, atol=0)


def test_scaled_fp8_packing_preserves_logical_kvpe_debug_intermediate(monkeypatch):
    latent = object()
    rope = object()
    latent_rm = object()
    rope_rm = object()
    latent_fp8 = object()
    scales = object()
    reconstructed_latent = object()
    logical_kvpe = object()
    packed = _tensor((1, 1, 32, SparseKVCache.PACKED_ROW_BYTES), ttnn.fp8_e4m3)
    deallocated = []

    def to_layout(tensor, layout):
        assert layout == ttnn.ROW_MAJOR_LAYOUT
        return latent_rm if tensor is latent else rope_rm

    def cast_to_fp8(tensor, *, round_scale_to_power_of_two):
        assert tensor is latent_rm
        assert round_scale_to_power_of_two
        return latent_fp8, scales

    def cast_back(fp8, fp8_scales, *, output_dtype):
        assert (fp8, fp8_scales, output_dtype) == (latent_fp8, scales, ttnn.bfloat16)
        return reconstructed_latent

    def pack(fp8, fp8_scales, packed_rope):
        assert (fp8, fp8_scales, packed_rope) == (latent_fp8, scales, rope_rm)
        return packed

    def concat(tensors, *, dim):
        assert tensors == [reconstructed_latent, rope_rm]
        assert dim == -1
        return logical_kvpe

    monkeypatch.setattr(ttnn, "to_layout", to_layout)
    monkeypatch.setattr(ttnn, "deallocate", deallocated.append)
    monkeypatch.setattr(ttnn, "clone", lambda tensor: ("clone", tensor))
    monkeypatch.setattr(ttnn, "concat", concat)
    monkeypatch.setattr(ttnn.experimental.deepseek_prefill, "per_token_cast_to_fp8", cast_to_fp8)
    monkeypatch.setattr(ttnn.experimental.deepseek_prefill, "per_token_cast_back", cast_back)
    monkeypatch.setattr(ttnn.experimental.deepseek_prefill, "pack_scaled_fp8_kv_cache", pack)

    intermediates = {}
    cache = ttMLA._pack_scaled_kvpe(None, latent, rope, intermediates)

    assert cache == SparseKVCache(SparseKVCacheFormat.SCALED_FP8, packed)
    assert intermediates["tt_kvpe"] is logical_kvpe
    assert intermediates["tt_kvpe_latent"] == ("clone", latent_fp8)
    assert intermediates["tt_kvpe_scales"] == ("clone", scales)
    assert intermediates["tt_kvpe_rope"] == ("clone", rope_rm)
    assert intermediates["tt_kvpe_packed"] == ("clone", packed)
    assert reconstructed_latent in deallocated


@pytest.mark.parametrize(
    "packed,match",
    [
        (
            _tensor((1, 1, 32, 656), ttnn.bfloat16),
            "packed cache must use",
        ),
        (
            _tensor((1, 1, 32, 512), ttnn.fp8_e4m3),
            "packed cache width must be 656",
        ),
        (
            _tensor((1, 1, 32, 656), ttnn.fp8_e4m3, ttnn.TILE_LAYOUT),
            "row-major",
        ),
    ],
)
def test_scaled_fp8_sparse_kv_contract_rejects_invalid_packed_storage(packed, match, expect_error):
    with expect_error(ValueError, match):
        SparseKVCache(SparseKVCacheFormat.SCALED_FP8, tensor=packed)


def test_index_cache_contract_remains_tiled_bfloat8_b():
    index_cache = _tensor((78, 1, 1024, 128), ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
    assert index_cache.dtype == ttnn.bfloat8_b
    assert index_cache.layout == ttnn.TILE_LAYOUT
