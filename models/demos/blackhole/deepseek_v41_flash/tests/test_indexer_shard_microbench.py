# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Feasibility micro-benchmark for column-sharding the PREFILL indexer (tt/prefill_sparse.py select_dyn): per prefill chunk the replicated path scores C query rows against all L index keys on every
column ([C, L] scores, a [C, L] visibility mask, a top-512 per row). The sharded path scores against L / 8 keys per column, takes a local top-512, gathers the 8 x 512 candidates over the columns and
takes the final top-512 of the 4096 candidates. Prints IS lines (ms per call, mean over a few synced calls): replicated vs sharded for U = 1, C = 512 and several L, and the buffer bytes."""

import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_sparse import IDIM, IHEADS, TOPK
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

C = 512


def timed_ms(md, fn, n=3):
    fn()
    ttnn.synchronize_device(md)
    t = time.perf_counter()
    for _ in range(n):
        fn()
    ttnn.synchronize_device(md)
    return (time.perf_counter() - t) / n * 1e3


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 100_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_indexer_shard(mesh_device):
    md = mesh_device
    cfg_m, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    rep = ttnn.ReplicateTensorToMesh(md)
    mk = lambda t, layout=ttnn.TILE_LAYOUT, dt=ttnn.bfloat16: ttnn.from_torch(
        t, device=md, dtype=dt, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep
    )
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=128, head_group_size=0)
    q = mk(torch.randn(1, IHEADS, C, IDIM).to(torch.bfloat16))
    w = mk(torch.rand(1, 1, C, IHEADS).to(torch.bfloat16))

    def local(keys, vis, L):
        sc = ttnn.experimental.indexer_score_dsa(
            q, keys, w, chunk_start_idx=L - 32, kv_len=L, program_config=cfg, cache_batch_idx=0
        )
        scm = ttnn.to_layout(
            ttnn.where(vis, ttnn.to_layout(sc, ttnn.TILE_LAYOUT), float("-inf")), ttnn.ROW_MAJOR_LAYOUT
        )
        ids = ttnn.experimental.topk_large_indices(scm, k=TOPK, valid_length=L)
        return scm, ids

    for L in (65536, 131072, 262144):
        keys = mk(torch.randn(1, 1, L, IDIM).to(torch.bfloat16))
        vis = mk((torch.rand(1, 1, C, L) > 0.01).to(torch.bfloat16))
        try:
            ms_rep = timed_ms(md, lambda: local(keys, vis, L))
            print(
                f"IS L={L}: replicated select {ms_rep:.2f} ms; score+masked buffers {2 * C * L * 2 / 2**20:.0f} MiB, vis {C * L * 2 / 2**20:.0f} MiB, keys {L * IDIM * 2 / 2**20:.0f} MiB",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"IS L={L}: replicated ERR {str(e)[:160]}", flush=True)
        ttnn.deallocate(keys)
        ttnn.deallocate(vis)
        L8 = L // 8
        keys8 = mk(torch.randn(1, 1, L8, IDIM).to(torch.bfloat16))
        vis8 = mk((torch.rand(1, 1, C, L8) > 0.01).to(torch.bfloat16))

        def sharded():
            scm, ids = local(keys8, vis8, L8)
            vals = ttnn.gather(
                ttnn.to_layout(scm, ttnn.TILE_LAYOUT),
                3,
                index=ttnn.to_layout(ttnn.typecast(ids, ttnn.uint32), ttnn.TILE_LAYOUT),
            )  # scores at the local top-512
            cand_v = cfg_m.allgather(ttnn.to_layout(vals, ttnn.ROW_MAJOR_LAYOUT), ccl, axis=1, dim=3)  # [1,1,C,8*512]
            cand_i = cfg_m.allgather(ttnn.typecast(ids, ttnn.uint32), ccl, axis=1, dim=3)
            pos = ttnn.experimental.topk_large_indices(ttnn.to_layout(cand_v, ttnn.ROW_MAJOR_LAYOUT), k=TOPK)
            return ttnn.gather(
                ttnn.to_layout(cand_i, ttnn.TILE_LAYOUT),
                3,
                index=ttnn.to_layout(ttnn.typecast(pos, ttnn.uint32), ttnn.TILE_LAYOUT),
            )

        try:
            ms_sh = timed_ms(md, sharded)
            print(
                f"IS L={L}: sharded select (L/8 local + 8x512 merge) {ms_sh:.2f} ms; per-column buffers {2 * C * L8 * 2 / 2**20:.0f} MiB scores, {C * L8 * 2 / 2**20:.0f} MiB vis, {L8 * IDIM * 2 / 2**20:.0f} MiB keys",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"IS L={L}: sharded ERR {str(e)[:300]}", flush=True)
        ttnn.deallocate(keys8)
        ttnn.deallocate(vis8)
