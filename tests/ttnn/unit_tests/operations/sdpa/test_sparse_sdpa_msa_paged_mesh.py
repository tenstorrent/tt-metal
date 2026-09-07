# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.sdpa.test_sparse_sdpa_msa import (
    _make_paged_msa_pools,
    _paged_msa_memory_config,
    make_msa_inputs,
    pcc,
    sparse_attention_ref_msa,
)


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [(2, 1)], indirect=True)
@pytest.mark.parametrize("extra_page", [0, 1], ids=["even_pages", "odd_pages"])
def test_msa_paged_sp_columns_and_slots(mesh_device, expect_error, extra_page):
    """Replicated allocator rows select per-SP, per-slot K/V pools with layers, heads and sub-page shards."""
    sp, dim, page_size, seq_len, block_size = 2, 64, 64, 512, 32
    heads, kv_heads, local_q, layers, layer = 32, 2, 80, 2, 1
    q, _, _, indices = make_msa_inputs(heads, kv_heads, local_q * sp, seq_len, 16, dim, blk_kv=block_size, seed=841)
    table = torch.zeros((3, sp * seq_len // page_size + extra_page), dtype=torch.int64)
    k_pools, v_pools, references = [], [], {0: [], 2: []}
    for rank in range(sp):
        local_k_pools, local_v_pools = [], []
        bundle_base = 0
        for slot in (0, 2):
            _, k, v, _ = make_msa_inputs(
                heads, kv_heads, local_q, seq_len, 16, dim, blk_kv=block_size, seed=842 + rank * 3 + slot
            )
            kp, vp, ids = _make_paged_msa_pools(k, v, page_size, layers, layer, seed=852 + rank * 3 + slot)
            table[slot, rank : sp * seq_len // page_size : sp] = ids[0] + bundle_base
            local_k_pools.append(kp)
            local_v_pools.append(vp)
            bundle_base += kp.shape[0] // (layers * kv_heads)
            references[slot].append(
                sparse_attention_ref_msa(
                    q[:, :, rank * local_q : (rank + 1) * local_q],
                    k,
                    v,
                    indices[:, :, rank * local_q : (rank + 1) * local_q],
                    dim**-0.5,
                    blk_kv=block_size,
                )
            )
        k_pools.append(torch.cat(local_k_pools))
        v_pools.append(torch.cat(local_v_pools))
    seq_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, 1), dims=(2, None))
    pool_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, 1), dims=(0, None))

    def upload(value, dtype, mapper, *, layout=ttnn.ROW_MAJOR_LAYOUT, memory=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.from_torch(
            value, device=mesh_device, dtype=dtype, layout=layout, memory_config=memory, mesh_mapper=mapper
        )

    tt_q = upload(q.to(torch.bfloat16), ttnn.bfloat16, seq_mapper)
    tt_idx = upload(indices, ttnn.uint32, seq_mapper)
    pool_mem = _paged_msa_memory_config(mesh_device, page_size, dim, shard_height=32)
    tt_k = upload(
        torch.cat(k_pools).to(torch.bfloat16), ttnn.bfloat16, pool_mapper, layout=ttnn.TILE_LAYOUT, memory=pool_mem
    )
    tt_v = upload(
        torch.cat(v_pools).to(torch.bfloat16), ttnn.bfloat16, pool_mapper, layout=ttnn.TILE_LAYOUT, memory=pool_mem
    )
    tt_table = upload(table, ttnn.uint32, ttnn.ReplicateTensorToMesh(mesh_device))
    mesh_device.clear_program_cache()
    args = dict(
        block_size=block_size,
        kv_cache_page_size=page_size,
        kv_cache_num_layers=layers,
        kv_cache_layer_idx=layer,
        page_bundle_indices=tt_table,
        kv_cache_sp_axis=0,
    )
    entries = None
    for slot in (2, 0, 2):
        out = ttnn.transformer.sparse_sdpa_msa(tt_q, tt_k, tt_v, tt_idx, kv_cache_slot_idx=slot, **args)
        actual = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2))
        score = pcc(actual, torch.cat(references[slot], dim=2))
        assert score >= 0.99, f"SP-interleaved slot {slot} PCC {score:.5f}"
        if entries is None:
            entries = mesh_device.num_program_cache_entries()
        else:
            assert mesh_device.num_program_cache_entries() == entries, "slot change rebuilt mesh programs"
    with expect_error(RuntimeError, "kv_cache_sp_axis"):
        ttnn.transformer.sparse_sdpa_msa(tt_q, tt_k, tt_v, tt_idx, **{**args, "kv_cache_sp_axis": 2})
