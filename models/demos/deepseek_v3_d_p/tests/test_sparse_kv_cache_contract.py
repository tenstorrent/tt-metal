# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
    MlaKvCache,
    MlaKvCacheFormat,
    reconstruct_scaled_fp8_kv_cache,
    unpack_scaled_fp8_kv_cache,
)


def _tensor(shape, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    return SimpleNamespace(shape=shape, dtype=dtype, layout=layout)


def test_bfp8_tile_kv_contract_retains_physical_storage():
    storage = _tensor((78, 1, 1024, 576), ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
    cache = MlaKvCache(MlaKvCacheFormat.BFP8_TILE, storage=storage)

    assert cache.storage is storage


def test_homogeneous_pack_is_conversion_free_when_storage_format_matches(monkeypatch):
    storage = _tensor((78, 1, 1024, 576), ttnn.bfloat16)
    logical = _tensor((1, 1, 32, 576), ttnn.bfloat16)
    cache = MlaKvCache(MlaKvCacheFormat.BF16_RM, storage=storage)
    monkeypatch.setattr(ttnn, "concat", lambda tensors, dim: logical)
    monkeypatch.setattr(ttnn, "to_layout", lambda *_: pytest.fail("unexpected layout conversion"))
    monkeypatch.setattr(ttnn, "typecast", lambda *_: pytest.fail("unexpected dtype conversion"))

    assert cache.pack(object(), object()) is logical


def test_bf16_sparse_kv_contract_retains_combined_cache():
    kvpe = _tensor((78, 1, 1024, 576), ttnn.bfloat16)
    cache = MlaKvCache(MlaKvCacheFormat.BF16_RM, storage=kvpe)

    assert cache.storage is kvpe


def test_scaled_fp8_sparse_kv_contract_owns_one_mixed_format_cache():
    packed = _tensor((78, 1, 1024, 656), ttnn.fp8_e4m3)
    cache = MlaKvCache(MlaKvCacheFormat.SCALED_FP8, storage=packed)

    assert cache.storage is packed
    assert MlaKvCache.PACKED_ROW_BYTES == 656
    assert MlaKvCache.PACKED_ROW_BYTES < 0.59 * (MlaKvCache.LATENT_DIM + MlaKvCache.ROPE_DIM) * 2


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
    packed = _tensor((1, 1, 32, MlaKvCache.PACKED_ROW_BYTES), ttnn.fp8_e4m3)
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
    cache = MlaKvCache(MlaKvCacheFormat.SCALED_FP8, packed)
    result = cache.pack(latent, rope, intermediates=intermediates)

    assert result is packed
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
            "cache must use",
        ),
        (
            _tensor((1, 1, 32, 512), ttnn.fp8_e4m3),
            "cache width must be 656",
        ),
        (
            _tensor((1, 1, 32, 656), ttnn.fp8_e4m3, ttnn.TILE_LAYOUT),
            "cache must use",
        ),
    ],
)
def test_scaled_fp8_sparse_kv_contract_rejects_invalid_packed_storage(packed, match, expect_error):
    with expect_error(ValueError, match):
        MlaKvCache(MlaKvCacheFormat.SCALED_FP8, storage=packed)


def test_index_cache_contract_remains_tiled_bfloat8_b():
    index_cache = _tensor((78, 1, 1024, 128), ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
    assert index_cache.dtype == ttnn.bfloat8_b
    assert index_cache.layout == ttnn.TILE_LAYOUT
