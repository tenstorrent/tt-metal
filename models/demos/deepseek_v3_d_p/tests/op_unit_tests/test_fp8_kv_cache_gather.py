# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Focused coverage for the FP8 sparse-MLA cache gather pipeline.

The chunked path writes each chip's 1/tp window into its SP*TP-deduped persistent cache, then selects
one slot and reassembles the whole sequence with a single full-mesh snake all-gather before sparse_sdpa.
This test validates that exact dtype/layout and communication sequence independently of model weights.

It follows ttMLA deliberately: it allocates through the real adapter and then hand-rolls the same
write + gather the model performs, so a change to the model's cache layout shows up here. (It did: the
SP-only gather this used to mirror was deleted when the sparse path became deduped-only.)
"""

from dataclasses import replace

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.adapter import PrefillRunParams
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import GLM51Config, glm_hf_config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_1 import GLM51Adapter
from models.demos.deepseek_v3_d_p.tt.runners.kv_chunk_table import (
    _dram_chunk_size_bytes,
    build_and_serialize_kv_chunk_table,
)
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import PREFILL_CHUNK_TOKENS, MlaKvCacheFormat


@pytest.mark.parametrize(
    "device_params",
    [fabric2d_device_params(model_config=GLM51Config)],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["2x4"], indirect=True)
@pytest.mark.timeout(0)
def test_fp8_row_major_kv_cache_all_gather(mesh_device, tmp_path):
    if not is_blackhole():
        pytest.skip("FP8_E4M3 is Blackhole-only")

    sp_axis, tp_axis = 0, 1
    mesh_shape = tuple(mesh_device.shape)
    seq_len = PREFILL_CHUNK_TOKENS
    config = glm_hf_config(max_seq=seq_len)
    head_dim = config.kv_lora_rank + config.qk_rope_head_dim
    params = PrefillRunParams(
        mesh_shape=mesh_shape,
        num_layers=1,
        first_layer_idx=0,
        is_first_rank=True,
        is_last_rank=True,
        max_seq_len=seq_len,
        chunk_size=seq_len,
        num_users=1,
        capacity_factor=1,
        num_links=1,
        gate_mode_name="HOST_ALL",
        kv_only_last_layer=False,
        weight_cache_path=None,
        sparse_kv_cache_format=MlaKvCacheFormat.SCALED_FP8,
    )
    adapter = GLM51Adapter()
    assert adapter.default_sparse_kv_cache_format == MlaKvCacheFormat.SCALED_FP8
    assert adapter.resolve_sparse_kv_cache_format(None) == MlaKvCacheFormat.SCALED_FP8
    assert adapter.resolve_sparse_kv_cache_format(MlaKvCacheFormat.BF16_RM) == MlaKvCacheFormat.BF16_RM
    caches = adapter.allocate_kv_cache(mesh_device=mesh_device, hf_config=config, params=params)
    cache = caches.kvpe
    index_cache = caches.index
    assert index_cache is not None
    assert cache.format == MlaKvCacheFormat.SCALED_FP8
    assert cache.storage.dtype == ttnn.fp8_e4m3
    assert cache.storage.layout == ttnn.ROW_MAJOR_LAYOUT
    assert cache.storage.shape[-1] == 656
    assert index_cache.dtype == ttnn.bfloat8_b
    assert index_cache.layout == ttnn.TILE_LAYOUT
    assert _dram_chunk_size_bytes(cache.storage) == 32 * cache.storage.buffer_aligned_page_size()

    table_path = tmp_path / "scaled_fp8_kv_table.pb"
    build_and_serialize_kv_chunk_table(
        mesh_device=mesh_device,
        kvpe_cache=cache,
        index_kv_cache=index_cache,
        seq_len=seq_len,
        num_layers=1,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_users=1,
        chunk_size_global=PREFILL_CHUNK_TOKENS,
        path=str(table_path),
    )
    table = ttnn.experimental.disaggregation.import_from_protobuf_file(str(table_path))
    assert table.num_configs() == 2  # packed KVPE, tiled index

    bf16_params = replace(params, sparse_kv_cache_format=MlaKvCacheFormat.BF16_RM)
    bf16_caches = adapter.allocate_kv_cache(mesh_device=mesh_device, hf_config=config, params=bf16_params)
    bf16_cache = bf16_caches.kvpe
    bf16_index_cache = bf16_caches.index
    assert bf16_index_cache is not None
    assert bf16_cache.format == MlaKvCacheFormat.BF16_RM
    assert bf16_cache.storage.dtype == ttnn.bfloat16
    assert cache.storage.buffer_aligned_page_size() < bf16_cache.storage.buffer_aligned_page_size()
    ttnn.deallocate(bf16_cache.storage, force=True)
    ttnn.deallocate(bf16_index_cache, force=True)

    torch.manual_seed(0)
    source = torch.randn(1, 1, seq_len, head_dim, dtype=torch.bfloat16)
    source_bf16 = ttnn.from_torch(
        source,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=(2, None)),
    )
    # The whole per-chip SP shard: source is dim-2 sharded over the SP axis, so each chip holds
    # seq_len / sp rows. (Was hardcoded seq_len // 2, which silently assumed sp == 2.)
    seq_len_local = seq_len // mesh_shape[sp_axis]
    latent_bf16 = ttnn.slice(source_bf16, [0, 0, 0, 0], [1, 1, seq_len_local, config.kv_lora_rank])
    source_rope = ttnn.slice(
        source_bf16,
        [0, 0, 0, config.kv_lora_rank],
        [1, 1, seq_len_local, head_dim],
    )
    source_packed = cache.pack(latent_bf16, source_rope)
    # tp_axis, matching ttMLA's write: the sparse path's cache is striped across SP*TP (the adapter
    # allocates it that way and there is no TP-replicated variant), so each chip owns seq_len/(sp*tp)
    # rows and must write only its own 1/tp window of the TP-replicated source. Without tp_axis the op
    # computes written_seq = input_seq / 1 and rejects the cache as too short for it.
    ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
        cache.storage,
        source_packed,
        slot_idx=0,
        layer_idx=0,
        num_layers=1,
        kv_actual_global=0,
        cluster_axis=sp_axis,
        tp_axis=tp_axis,
    )

    # This mirrors ttMLA._gather_kvpe_prefix, which is now ONE full-mesh snake gather
    # (cluster_axis=None) reading the ND-sharded cache directly into a replicated worst-case scratch.
    # It replaced an SP-only all_gather_async over an interleaved copy when the SP-only gather was
    # deleted: the deduped cache is striped over BOTH axes, so a single-axis gather could only reach
    # 1/tp of the rows. Row-major over the mesh IS the sp*tp linearization, so the snake reassembles
    # the slab in natural order. Requires a 2D fabric, which this test already opens.
    gather_out = ttnn.from_torch(
        torch.zeros(1, 1, seq_len, cache.storage.shape[-1]),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=cache.storage.dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    gathered = ttnn.experimental.high_bw_all_gather(
        cache.storage,
        dim=2,
        output_tensor=gather_out,
        num_links=1,
        cluster_axis=None,
        input_batch_index=0,
        gathered_dim_size=seq_len,
    )
    ttnn.synchronize_device(mesh_device)

    assert gathered.dtype == ttnn.fp8_e4m3
    assert gathered.layout == ttnn.ROW_MAJOR_LAYOUT

    # Every output device holds the full gathered sequence. Compose the replicated result as one tensor so
    # FP8 host export remains supported, then compare raw mixed-format bytes.
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_shape)
    expected = ttnn.to_torch(source_packed, mesh_composer=composer).contiguous().view(torch.uint8)[:, :1]
    actual_all = ttnn.to_torch(gathered, mesh_composer=composer).contiguous().view(torch.uint8)
    for sp_rank in range(mesh_shape[0]):
        for tp_rank in range(mesh_shape[1]):
            actual = actual_all[:, tp_rank : tp_rank + 1, sp_rank * seq_len : (sp_rank + 1) * seq_len]
            assert torch.equal(actual, expected)
