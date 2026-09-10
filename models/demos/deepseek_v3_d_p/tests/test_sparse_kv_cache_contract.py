# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.utils import kv_cache_utils
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
    MlaKvCache,
    MlaKvCacheFormat,
    MlaKvCacheGeometry,
    init_mla_kv_cache,
    reconstruct_scaled_fp8_kv_cache,
    unpack_scaled_fp8_kv_cache,
)


def _tensor(shape, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    return SimpleNamespace(shape=shape, dtype=dtype, layout=layout)


PRODUCTION_GEOMETRY = MlaKvCacheGeometry(latent_dim=512, rope_dim=64)


def test_mla_kv_cache_geometry_is_derived_from_config():
    config = SimpleNamespace(kv_lora_rank=1024, qk_rope_head_dim=32)
    geometry = MlaKvCacheGeometry.from_config(config)

    assert geometry.logical_width == 1056
    assert geometry.num_scales == 8
    assert geometry.rope_offset_bytes == 1056
    assert geometry.packed_row_bytes == 1120

    storage = _tensor((1, 1, 32, geometry.packed_row_bytes), ttnn.fp8_e4m3)
    cache = MlaKvCache(MlaKvCacheFormat.SCALED_FP8, storage=storage, geometry=geometry)
    assert cache.geometry is geometry


def test_scaled_mla_kv_cache_geometry_rejects_unaligned_rope_offset(expect_error):
    geometry = MlaKvCacheGeometry(latent_dim=256, rope_dim=32)
    storage = _tensor((1, 1, 32, geometry.packed_row_bytes), ttnn.fp8_e4m3)

    with expect_error(ValueError, "16-byte aligned"):
        MlaKvCache(MlaKvCacheFormat.SCALED_FP8, storage=storage, geometry=geometry)


def test_mla_kv_cache_allocation_uses_config_geometry(monkeypatch):
    config = SimpleNamespace(kv_lora_rank=1024, qk_rope_head_dim=32)

    def allocate(*, kvpe_cache_head_dim, dtype, layout, **_):
        return _tensor((1, 1, 32, kvpe_cache_head_dim), dtype, layout)

    monkeypatch.setattr(kv_cache_utils, "init_kvpe_cache", allocate)
    cache = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.SCALED_FP8,
        hf_config=config,
        mesh_device=object(),
        seq_len=32,
        mesh_shape=[1, 1],
        sp_axis=0,
        num_kvpe_cache_layers=1,
    )

    assert cache.geometry == MlaKvCacheGeometry.from_config(config)
    assert cache.storage.shape[-1] == 1120


def test_synthetic_scaled_fp8_host_codec_uses_geometry_offsets():
    geometry = MlaKvCacheGeometry(latent_dim=1024, rope_dim=32)
    latent = torch.ones((1, geometry.latent_dim), dtype=torch.float8_e4m3fn)
    scales = torch.arange(1, geometry.num_scales + 1, dtype=torch.float32).unsqueeze(0)
    rope = torch.arange(geometry.rope_dim, dtype=torch.float32).to(torch.bfloat16).unsqueeze(0)
    packed = torch.cat((latent.view(torch.uint8), scales.view(torch.uint8), rope.view(torch.uint8)), dim=-1).view(
        torch.float8_e4m3fn
    )

    decoded = reconstruct_scaled_fp8_kv_cache(packed, geometry)
    expected_latent = scales.repeat_interleave(geometry.SCALE_BLOCK_SIZE, dim=-1).to(torch.bfloat16)
    torch.testing.assert_close(decoded, torch.cat((expected_latent, rope), dim=-1), rtol=0, atol=0)


def test_mla_kv_cache_contract_covers_dense_and_sparse_physical_formats():
    dense = MlaKvCache(
        MlaKvCacheFormat.BFP8_TILE,
        storage=_tensor((1, 1, 32, 576), ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
        geometry=PRODUCTION_GEOMETRY,
    )
    sparse = MlaKvCache(
        MlaKvCacheFormat.BF16_RM,
        storage=_tensor((1, 1, 32, 576), ttnn.bfloat16),
        geometry=PRODUCTION_GEOMETRY,
    )

    assert dense.storage.layout == ttnn.TILE_LAYOUT
    assert sparse.storage.layout == ttnn.ROW_MAJOR_LAYOUT


def test_homogeneous_mla_kv_cache_does_not_require_scaled_geometry():
    geometry = MlaKvCacheGeometry(latent_dim=96, rope_dim=32)
    storage = _tensor((1, 1, 32, geometry.logical_width), ttnn.bfloat16)

    cache = MlaKvCache(MlaKvCacheFormat.BF16_RM, storage=storage, geometry=geometry)

    assert cache.storage is storage


def test_bf16_sparse_kv_contract_retains_combined_cache():
    kvpe = _tensor((78, 1, 1024, 576), ttnn.bfloat16)
    cache = MlaKvCache(MlaKvCacheFormat.BF16_RM, storage=kvpe, geometry=PRODUCTION_GEOMETRY)

    assert cache.storage is kvpe


def test_mla_kv_cache_pack_rejects_inputs_that_disagree_with_config(expect_error):
    storage = _tensor((1, 1, 32, PRODUCTION_GEOMETRY.logical_width), ttnn.bfloat16)
    cache = MlaKvCache(MlaKvCacheFormat.BF16_RM, storage=storage, geometry=PRODUCTION_GEOMETRY)
    latent = SimpleNamespace(shape=(1, 1, 32, 256))
    rope = SimpleNamespace(shape=(1, 1, 32, 64))

    with expect_error(ValueError, "latent width 512"):
        cache.pack(latent, rope)


def test_scaled_fp8_sparse_kv_contract_owns_one_mixed_format_cache():
    packed = _tensor((78, 1, 1024, 656), ttnn.fp8_e4m3)
    cache = MlaKvCache(MlaKvCacheFormat.SCALED_FP8, storage=packed, geometry=PRODUCTION_GEOMETRY)

    assert cache.storage is packed
    assert cache.geometry.packed_row_bytes == 656
    assert cache.geometry.packed_row_bytes < 0.59 * cache.geometry.logical_width * 2


def test_scaled_fp8_packed_host_codec_preserves_mixed_fields():
    latent = torch.tensor([[0.5, -1.0, 2.0, -4.0]], dtype=torch.float8_e4m3fn).repeat(1, 128)
    scales = torch.tensor([[0.25, 0.5, 1.0, 2.0]], dtype=torch.float32)
    rope = torch.arange(64, dtype=torch.float32).to(torch.bfloat16).unsqueeze(0)
    raw = torch.cat(
        (latent.view(torch.uint8), scales.view(torch.uint8), rope.view(torch.uint8)),
        dim=-1,
    )
    packed = raw.view(torch.float8_e4m3fn)

    decoded_latent, decoded_scales, decoded_rope = unpack_scaled_fp8_kv_cache(packed, PRODUCTION_GEOMETRY)
    torch.testing.assert_close(decoded_latent, latent.float(), rtol=0, atol=0)
    torch.testing.assert_close(decoded_scales, scales, rtol=0, atol=0)
    torch.testing.assert_close(decoded_rope, rope, rtol=0, atol=0)

    expected_latent = latent.float() * scales.repeat_interleave(128, dim=-1)
    expected = torch.cat((expected_latent.to(torch.bfloat16), rope), dim=-1)
    torch.testing.assert_close(reconstruct_scaled_fp8_kv_cache(packed, PRODUCTION_GEOMETRY), expected, rtol=0, atol=0)


def test_scaled_fp8_packing_preserves_logical_kvpe_debug_intermediate(monkeypatch):
    latent = SimpleNamespace(shape=(1, 1, 32, PRODUCTION_GEOMETRY.latent_dim))
    rope = SimpleNamespace(shape=(1, 1, 32, PRODUCTION_GEOMETRY.rope_dim))
    latent_rm = object()
    rope_rm = object()
    latent_fp8 = object()
    scales = object()
    reconstructed_latent = object()
    logical_kvpe = object()
    packed = _tensor((1, 1, 32, PRODUCTION_GEOMETRY.packed_row_bytes), ttnn.fp8_e4m3)
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
    cache = MlaKvCache(MlaKvCacheFormat.SCALED_FP8, storage=packed, geometry=PRODUCTION_GEOMETRY)
    encoded = cache.pack(latent, rope, intermediates=intermediates)

    assert encoded is packed
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
            "ROW_MAJOR",
        ),
    ],
)
def test_scaled_fp8_sparse_kv_contract_rejects_invalid_packed_storage(packed, match, expect_error):
    with expect_error(ValueError, match):
        MlaKvCache(MlaKvCacheFormat.SCALED_FP8, storage=packed, geometry=PRODUCTION_GEOMETRY)


def test_index_cache_contract_remains_tiled_bfloat8_b():
    index_cache = _tensor((78, 1, 1024, 128), ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
    assert index_cache.dtype == ttnn.bfloat8_b
    assert index_cache.layout == ttnn.TILE_LAYOUT
