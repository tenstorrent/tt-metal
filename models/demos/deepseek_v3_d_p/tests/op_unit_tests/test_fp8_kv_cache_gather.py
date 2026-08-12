# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Focused coverage for the FP8 sparse-MLA cache gather pipeline.

The chunked path converts its ND-sharded persistent cache to interleaved DRAM, selects one slot, and
all-gathers sequence rows over SP before sparse_sdpa. This test validates that exact dtype/layout and
communication sequence independently of model weights.
"""

from dataclasses import replace

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.adapter import PrefillRunParams
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import glm_hf_config
from models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_1 import GLM51Adapter
from models.demos.deepseek_v3_d_p.tt.runners.kv_chunk_table import (
    _dram_chunk_size_bytes,
    build_and_serialize_kv_chunk_table,
)
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import PREFILL_CHUNK_OUTPUT_TOKENS, MlaKvCacheFormat


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["2x4"], indirect=True)
@pytest.mark.timeout(0)
def test_fp8_row_major_kv_cache_all_gather(mesh_device, tmp_path):
    if not is_blackhole():
        pytest.skip("FP8_E4M3 is Blackhole-only")

    sp_axis = 0
    mesh_shape = tuple(mesh_device.shape)
    seq_len = PREFILL_CHUNK_OUTPUT_TOKENS
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
        chunk_size_global=PREFILL_CHUNK_OUTPUT_TOKENS,
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
    latent_bf16 = ttnn.slice(source_bf16, [0, 0, 0, 0], [1, 1, seq_len // 2, config.kv_lora_rank])
    source_rope = ttnn.slice(
        source_bf16,
        [0, 0, 0, config.kv_lora_rank],
        [1, 1, seq_len // 2, head_dim],
    )
    source_packed = cache.pack(latent_bf16, source_rope)
    ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
        cache.storage,
        source_packed,
        slot_idx=0,
        layer_idx=0,
        num_layers=1,
        kv_actual_global=0,
        cluster_axis=sp_axis,
    )

    # This is the exact prefix-gather preparation in ttMLA._gather_kvpe_prefix.
    cache_interleaved = ttnn.to_memory_config(cache.storage, ttnn.DRAM_MEMORY_CONFIG)

    compute_grid = mesh_device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1))}
    )
    gather_semaphores = [ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(2)]
    barrier_semaphore = ttnn.create_global_semaphore(mesh_device, cores, 0)
    gathered = ttnn.experimental.all_gather_async(
        cache_interleaved,
        dim=2,
        multi_device_global_semaphore=gather_semaphores,
        barrier_semaphore=barrier_semaphore,
        num_links=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear,
        cluster_axis=sp_axis,
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
