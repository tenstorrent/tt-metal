# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vsa_ring_sdpa vs (two all_gather_async + vsa_sdpa) at the 15 s / 768p per-device shape on the 4x8 galaxy.

Per device: H=14 heads, 226 query tiles, 226-block K/V shard (T_local 14464), ring of 8 -> 1808 global blocks,
k=179 listed blocks per row with a spatially decaying selection over the 8 shards (like the real coarse stage:
own shard ~30 %, neighbors ~15-18 %, far shards ~5-8 %). Wall clock over `iters` warm iterations per variant.
VSA_RING_PERF_ITERS (default 6)."""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.utils.test import ring_params_8k_req_exact_devices, skip_if_unsupported_num_links

BLOCK = 64
DIM = 128
SENTINEL = 0xFFFFFFFF


def _selection(heads, rows_local, blocks_per_shard, sp, my_shard, k_sel, exempt_ids, gen):
    """Global block ids per (head, row): top-k of a smooth score field decaying with shard distance."""
    n_blocks = blocks_per_shard * sp
    idx = torch.full((1, heads, rows_local, ((n_blocks + 15) // 16) * 16), SENTINEL, dtype=torch.int64)
    shard_of = torch.arange(n_blocks) // blocks_per_shard
    dist = torch.minimum((shard_of - my_shard) % sp, (my_shard - shard_of) % sp).float()
    for h in range(heads):
        field = torch.randn(n_blocks, generator=gen)
        for r in range(rows_local):
            scores = field - 1.1 * dist + 0.6 * torch.randn(n_blocks, generator=gen)
            scores[exempt_ids] = float("-inf")  # exempt ids are added by the kernel, not listed
            idx[0, h, r, :k_sel] = scores.topk(k_sel).indices
    return idx


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params"),
    [
        pytest.param(
            (4, 8),
            1,
            0,
            2,
            {**ring_params_8k_req_exact_devices, "trace_region_size": 50_000_000, "l1_small_size": 65536},
            id="4x8sp1tp0nl2",
        )
    ],
    indirect=["mesh_device", "device_params"],
)
def test_vsa_ring_sdpa_perf_15s(mesh_device, sp_axis, tp_axis, num_links, reset_seeds):
    skip_if_unsupported_num_links(mesh_device, num_links)
    sp = tuple(mesh_device.shape)[sp_axis]
    tp = tuple(mesh_device.shape)[tp_axis]
    heads_local, blocks_per_shard, k_sel = 14, 226, 179
    heads_total = heads_local * tp
    n_blocks = blocks_per_shard * sp
    t_local = blocks_per_shard * BLOCK
    t_total = n_blocks * BLOCK
    exempt_ids = [0, 1]
    iters = int(os.environ.get("VSA_RING_PERF_ITERS", "6"))
    mesh_shape = tuple(mesh_device.shape)
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1)

    shard_qkv = [None, None]
    shard_qkv[tp_axis] = 1
    shard_qkv[sp_axis] = 2
    shard_heads = [None, None]
    shard_heads[tp_axis] = 1
    replicate = [None, None]
    to_dev = lambda t, dims, layout, dtype: ttnn.from_torch(
        t,
        device=mesh_device,
        layout=layout,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims),
    )
    q = torch.randn(1, heads_total, t_total, DIM, dtype=torch.bfloat16)
    k = torch.randn(1, heads_total, t_total, DIM, dtype=torch.bfloat16)
    v = torch.randn(1, heads_total, t_total, DIM, dtype=torch.bfloat16)
    # per-shard selections concatenated along the row dim so the sequence-sharded upload lands each shard's rows
    idx = torch.cat(
        [_selection(heads_total, blocks_per_shard, blocks_per_shard, sp, s, k_sel, exempt_ids, gen) for s in range(sp)],
        dim=2,
    )
    counts = torch.full((1, 1, 1, idx.shape[3]), BLOCK, dtype=torch.int32)
    counts[..., n_blocks:] = 0
    tt_q = to_dev(q, shard_qkv, ttnn.TILE_LAYOUT, ttnn.bfloat16)
    tt_k = to_dev(k, shard_qkv, ttnn.TILE_LAYOUT, ttnn.bfloat16)
    tt_v = to_dev(v, shard_qkv, ttnn.TILE_LAYOUT, ttnn.bfloat16)
    tt_idx = to_dev(idx.to(torch.uint32).view(torch.int32), shard_qkv, ttnn.ROW_MAJOR_LAYOUT, ttnn.uint32)
    tt_counts = to_dev(counts, replicate, ttnn.ROW_MAJOR_LAYOUT, ttnn.uint32)
    tt_gk = to_dev(
        torch.zeros(1, heads_total, t_total, DIM, dtype=torch.bfloat16), shard_heads, ttnn.TILE_LAYOUT, ttnn.bfloat16
    )
    tt_gv = to_dev(
        torch.zeros(1, heads_total, t_total, DIM, dtype=torch.bfloat16), shard_heads, ttnn.TILE_LAYOUT, ttnn.bfloat16
    )
    kv = torch.cat(  # per-TP-shard [k_local, v_local] so the dim-1 sharding hands each device its own K then V
        [
            torch.cat([k[:, t * heads_local : (t + 1) * heads_local], v[:, t * heads_local : (t + 1) * heads_local]], 1)
            for t in range(tp)
        ],
        dim=1,
    )
    tt_kv = to_dev(kv, shard_qkv, ttnn.TILE_LAYOUT, ttnn.bfloat16)
    tt_gkv = to_dev(
        torch.zeros(1, 2 * heads_total, t_total, DIM, dtype=torch.bfloat16),
        shard_heads,
        ttnn.TILE_LAYOUT,
        ttnn.bfloat16,
    )
    workers = int(os.environ.get("VSA_RING_WORKERS", "2"))
    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sems = [[ttnn.create_global_semaphore(mesh_device, crs, 0) for _ in range(2)] for _ in range(2)]
    common = dict(list_len=k_sel, exempt_ids=exempt_ids)
    # consecutive CCLs must strictly alternate the semaphore sets (the model's CCL manager does the same)
    calls = [0]

    def next_sems():
        calls[0] += 1
        return sems[calls[0] % 2]

    def ring(i):
        return ttnn.transformer.vsa_ring_sdpa(
            tt_q,
            tt_kv,
            tt_idx,
            tt_counts,
            tt_gkv,
            multi_device_global_semaphore=next_sems(),
            num_links=num_links,
            cluster_axis=sp_axis,
            mesh_device=mesh_device,
            topology=ttnn.Topology.Ring,
            num_workers_per_link=workers,
            **common,
        )

    def concat_only(i):  # the model's extra op for the fused path: K/V concatenated on the head dim
        return ttnn.concat([tt_k, tt_v], dim=1)

    def two_op(i):
        gk = ttnn.experimental.all_gather_async(
            tt_k,
            persistent_output_buffer=tt_gk,
            dim=2,
            multi_device_global_semaphore=next_sems(),
            num_links=num_links,
            topology=ttnn.Topology.Ring,
            cluster_axis=sp_axis,
        )
        gv = ttnn.experimental.all_gather_async(
            tt_v,
            persistent_output_buffer=tt_gv,
            dim=2,
            multi_device_global_semaphore=next_sems(),
            num_links=num_links,
            topology=ttnn.Topology.Ring,
            cluster_axis=sp_axis,
        )
        return ttnn.transformer.vsa_sdpa(tt_q, gk, gv, tt_idx, tt_counts, streaming=True, **common)

    def bench(fn, name):
        out = fn(0)  # compile / cache
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for i in range(1, iters + 1):
            out = fn(i)
        ttnn.synchronize_device(mesh_device)
        ms = (time.perf_counter() - t0) * 1e3 / iters
        logger.info(f"{name}: {ms:.2f} ms per call (slowest device, {iters} iters)")
        return ms, out

    def ag_only(i):
        ttnn.experimental.all_gather_async(
            tt_k,
            persistent_output_buffer=tt_gk,
            dim=2,
            multi_device_global_semaphore=next_sems(),
            num_links=num_links,
            topology=ttnn.Topology.Ring,
            cluster_axis=sp_axis,
        )
        return ttnn.experimental.all_gather_async(
            tt_v,
            persistent_output_buffer=tt_gv,
            dim=2,
            multi_device_global_semaphore=next_sems(),
            num_links=num_links,
            topology=ttnn.Topology.Ring,
            cluster_axis=sp_axis,
        )

    def vsa_only(i):  # gathered buffers already hold the full K/V after ag_only
        return ttnn.transformer.vsa_sdpa(tt_q, tt_gk, tt_gv, tt_idx, tt_counts, streaming=True, **common)

    ms_cat, _ = bench(concat_only, "concat(k, v) alone")
    ms_ag, _ = bench(ag_only, "all_gather x2 alone")
    ms_vsa, _ = bench(vsa_only, "vsa_sdpa alone (pre-gathered K/V)")
    ms_two, out_two = bench(two_op, "all_gather x2 + vsa_sdpa")
    ms_ring, out_ring = bench(ring, "vsa_ring_sdpa")
    compose = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=shard_qkv)
    a = ttnn.to_torch(out_two, mesh_composer=compose).float()
    b = ttnn.to_torch(out_ring, mesh_composer=compose).float()
    pcc = torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
    print(
        f"\nVSA_RING_PERF concat={ms_cat:.2f} ag={ms_ag:.2f} vsa={ms_vsa:.2f} two_op={ms_two:.2f} ring={ms_ring:.2f} ms "
        f"saving={ms_two - ms_ring - ms_cat:.2f} ms (incl. concat) pcc={pcc:.6f} workers={workers} "
        f"wait_all={os.environ.get('TT_VSA_RING_WAIT_ALL', '0')} rmax={os.environ.get('TT_VSA_RMAX', '-')} depth={os.environ.get('TT_VSA_DEPTH', '-')}"
    )
    assert pcc > 0.999
