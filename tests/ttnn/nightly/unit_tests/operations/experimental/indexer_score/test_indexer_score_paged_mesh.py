# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.nightly.unit_tests.operations.experimental.indexer_score.test_indexer_score import (
    _msa_scale_w,
    _nd_sharded_dram_config,
    assert_grouped_match,
    assert_indexer_match,
    indexer_score_dsa_ref,
    indexer_score_msa_ref,
    make_inputs,
)


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [(2, 1)], indirect=True)
@pytest.mark.parametrize("extra_page", [0, 1], ids=["even_pages", "odd_pages"])
@pytest.mark.parametrize("mode", ["dsa", "msa"])
def test_indexer_allocator_sp_slots(mesh_device, extra_page, mode):
    sp, heads, dim, sq, length, page_size = 2, 4, 128, 64, 256, 32
    pages, layers, layer = length // page_size, 2, 1
    torch.manual_seed(3834)
    table = torch.zeros(3, sp * pages + extra_page, dtype=torch.int64)
    pools, queries, weights, references = [], [], [], {0: [], 2: []}
    for rank in range(sp):
        q, _, w = make_inputs(heads, dim, sq, length, seed=343 + rank)
        queries.append(q)
        weights.append(w)
        pool = torch.randn(3 * pages * layers, 1, page_size, dim, dtype=torch.bfloat16)
        for slot in (0, 2):
            ids = torch.randperm(pages) + slot * pages
            k = torch.randn(1, 1, length, dim, dtype=torch.bfloat16)
            table[slot, rank : sp * pages : sp] = ids
            pool[ids * layers + layer, 0] = k.reshape(pages, page_size, dim)
            if mode == "dsa":
                ref = indexer_score_dsa_ref(q, k, w, 128 + rank * sq)
            else:
                ref = indexer_score_msa_ref(
                    q, k, _msa_scale_w(heads, sq, dim**-0.5), 128 + rank * sq, num_groups=heads, block_size=0
                )
            references[slot].append(ref)
        pools.append(pool)
    seq_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, 1), dims=(2, None))
    pool_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, 1), dims=(0, None))

    def upload(value, mapper, memory=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.from_torch(
            value,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory,
            mesh_mapper=mapper,
        )

    tt_q = upload(torch.cat(queries, dim=2), seq_mapper)
    tt_w = upload(torch.cat(weights, dim=2), seq_mapper)
    tt_k = upload(torch.cat(pools), pool_mapper, _nd_sharded_dram_config(mesh_device, rows_per_shard=page_size))
    tt_table = ttnn.from_torch(
        table,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=64, head_group_size=0)
    kwargs = dict(
        chunk_start_idx=128,
        kv_len=length,
        program_config=cfg,
        page_bundle_indices=tt_table,
        kv_cache_page_size=page_size,
        kv_cache_num_layers=layers,
        kv_cache_layer_idx=layer,
        kv_cache_sp_axis=0,
    )
    entries = None
    for slot in (2, 0, 2):
        if mode == "dsa":
            out = ttnn.experimental.indexer_score_dsa(tt_q, tt_k, tt_w, kv_cache_slot_idx=slot, **kwargs)
        else:
            out = ttnn.experimental.indexer_score_msa(tt_q, tt_k, num_groups=heads, kv_cache_slot_idx=slot, **kwargs)
        shards = ttnn.get_device_tensors(out)
        for rank, shard in enumerate(shards):
            actual = ttnn.to_torch(shard)[..., :length]
            if mode == "dsa":
                assert_indexer_match(actual, references[slot][rank], sq, length, check_neg=True)
            else:
                assert_grouped_match(actual, references[slot][rank], heads, sq, length)
        if entries is None:
            entries = mesh_device.num_program_cache_entries()
        else:
            assert mesh_device.num_program_cache_entries() == entries
