# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.sdpa.test_sparse_sdpa import (
    BF16_KV,
    _make_paged_kv,
    _nd_sharded_dram_config,
    golden,
    make_inputs,
    pcc,
)


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [(2, 1)], indirect=True)
def test_sparse_sdpa_paged_sp_columns_and_slots(mesh_device, expect_error):
    """A replicated table selects distinct local pools on each SP and repatches its slot on cache hits."""
    sp, dim, page_size, seq_len = 2, 32, 32, 128
    q, _, indices = make_inputs(32, 32 * sp, seq_len, 32, dim, lambda s: 32, seed=811)
    table = torch.zeros((3, sp * seq_len // page_size), dtype=torch.int64)
    pools, references = [], {0: [], 2: []}
    for rank in range(sp):
        local_pools = []
        bundle_base = 0
        for slot in (0, 2):
            gen = torch.Generator().manual_seed(812 + rank * 3 + slot)
            kv = torch.randn((1, 1, seq_len, dim), generator=gen, dtype=torch.bfloat16)
            pool, ids = _make_paged_kv(kv, page_size, num_layers=1, layer_idx=0, seed=821 + rank * 3 + slot)
            table[slot, rank::sp] = ids[0] + bundle_base
            local_pools.append(pool)
            bundle_base += pool.shape[0]
            references[slot].append(
                golden(
                    q[:, :, rank * 32 : (rank + 1) * 32],
                    kv.float(),
                    indices[:, :, rank * 32 : (rank + 1) * 32],
                    dim**-0.5,
                    dim,
                )
            )
        pools.append(torch.cat(local_pools))
    seq_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, 1), dims=(2, None))
    pool_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, 1), dims=(0, None))

    def upload(value, dtype, mapper, memory=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.from_torch(
            value,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=memory,
            mesh_mapper=mapper,
        )

    tt_q = upload(q.to(torch.bfloat16), ttnn.bfloat16, seq_mapper)
    tt_idx = upload(indices.to(torch.int32), ttnn.uint32, seq_mapper)
    tt_pool = upload(torch.cat(pools), ttnn.bfloat16, pool_mapper, _nd_sharded_dram_config(mesh_device, page_size, dim))
    tt_table = upload(table, ttnn.uint32, ttnn.ReplicateTensorToMesh(mesh_device))
    mesh_device.clear_program_cache()
    entries = None
    for slot in (2, 0, 2):
        out = ttnn.transformer.sparse_sdpa(
            tt_q,
            tt_pool,
            tt_idx,
            dim,
            kv_format=BF16_KV,
            k_chunk_size=32,
            page_bundle_indices=tt_table,
            kv_cache_slot_idx=slot,
            kv_cache_sp_axis=0,
        )
        actual = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2))
        expected = torch.cat(references[slot], dim=2)
        score = pcc(actual, expected)
        assert score >= 0.99, f"SP-interleaved slot {slot} PCC {score:.5f}"
        if entries is None:
            entries = mesh_device.num_program_cache_entries()
        else:
            assert mesh_device.num_program_cache_entries() == entries, "slot change rebuilt mesh programs"
    with expect_error(RuntimeError, "kv_cache_sp_axis"):
        ttnn.transformer.sparse_sdpa(
            tt_q, tt_pool, tt_idx, dim, kv_format=BF16_KV, page_bundle_indices=tt_table, kv_cache_sp_axis=2
        )
