# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vsa_ring_sdpa on the 4x8 galaxy: the VSA fine stage fused with the SP-ring all-gather of K/V.

Checks, per device, that the fused op equals (a) the torch block-sparse reference and (b) vsa_sdpa run on
the all-gathered K/V (same math, shard-major visit order -> bf16 rounding differences only), that it is
bit-exact run to run, and that it replays under trace with the alternate semaphore set (program-cache hit).
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.tt_dit.utils.test import ring_params_8k_req_exact_devices, skip_if_unsupported_num_links

from .test_vsa_sdpa import fine_attention_ref

BLOCK = 64
DIM = 128
SENTINEL = 0xFFFFFFFF

GALAXY_4X8 = pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params"),
    [
        pytest.param(
            (4, 8),
            1,
            0,
            nl,
            {**ring_params_8k_req_exact_devices, "trace_region_size": 50_000_000, "l1_small_size": 65536, **extra},
            id=f"4x8sp1tp0nl{nl}{tag}",
        )
        for nl in (1, 2)
        for tag, extra in (("", {}),)
    ],
    indirect=["mesh_device", "device_params"],
)


def _build_indices(heads, n_rows, n_blocks, w, k_sel, exempt_ids, seed):
    """Raw top-k rows over the GLOBAL block space (real ids): k_sel random blocks per row, unsorted."""
    gen = torch.Generator().manual_seed(seed)
    idx = torch.full((1, heads, n_rows, w), SENTINEL, dtype=torch.int64)
    candidates = torch.tensor([b for b in range(n_blocks) if b not in exempt_ids])
    for h in range(heads):
        for r in range(n_rows):
            pick = candidates[torch.randperm(candidates.numel(), generator=gen)[:k_sel]]
            idx[0, h, r, :k_sel] = pick
    return idx


def _assembled_rows(idx, exempt_ids, k_sel, n_blocks, dense_rows):
    """The block set each row attends (list + exempt ids, or every block for dense rows), as a wide
    sentinel-padded index tensor the torch reference and plain vsa_sdpa consume."""
    _, heads, n_rows, _ = idx.shape
    out = torch.full((1, heads, n_rows, n_blocks), SENTINEL, dtype=torch.int64)
    for h in range(heads):
        for r in range(n_rows):
            if r in dense_rows:
                blocks = torch.arange(n_blocks)
            else:
                blocks = torch.unique(torch.cat([idx[0, h, r, :k_sel], torch.tensor(exempt_ids, dtype=torch.int64)]))
            out[0, h, r, : blocks.numel()] = blocks
    return out


def _run(
    mesh_device, sp_axis, tp_axis, num_links, *, heads_total, blocks_per_shard, k_sel, dense_local_rows, seed, trace
):
    sp = tuple(mesh_device.shape)[sp_axis]
    tp = tuple(mesh_device.shape)[tp_axis]
    assert heads_total % tp == 0
    heads_local = heads_total // tp
    n_blocks = blocks_per_shard * sp
    t_local = blocks_per_shard * BLOCK
    t_total = n_blocks * BLOCK
    n_rows_local = blocks_per_shard  # queries are the shard's own tiles
    w = (n_blocks + 15) // 16 * 16
    exempt_ids = [0, n_blocks // 2]
    torch.manual_seed(seed)
    q = torch.randn(1, heads_total, t_total, DIM, dtype=torch.bfloat16)
    k = torch.randn(1, heads_total, t_total, DIM, dtype=torch.bfloat16)
    v = torch.randn(1, heads_total, t_total, DIM, dtype=torch.bfloat16)
    # global index rows [1, heads_total, n_blocks(=all q tiles), w], sharded like q
    idx = _build_indices(heads_total, n_blocks, n_blocks, w, k_sel, exempt_ids, seed + 1)
    # dense rows: the same LOCAL row ids on every shard (the mask is per device; keep it simple)
    dense_words = 8
    mask = torch.zeros(dense_words, dtype=torch.int64)
    for r in dense_local_rows:
        mask[r // 32] |= 1 << (r % 32)
    dense_global = {s * blocks_per_shard + r for s in range(sp) for r in dense_local_rows}
    counts = torch.full((1, 1, 1, w), BLOCK, dtype=torch.int32)
    counts[..., n_blocks:] = 0

    shard_qkv = [None, None]
    shard_qkv[tp_axis] = 1  # heads
    shard_qkv[sp_axis] = 2  # sequence
    shard_heads = [None, None]
    shard_heads[tp_axis] = 1
    mesh_shape = tuple(mesh_device.shape)
    to_dev = lambda t, dims, layout, dtype: ttnn.from_torch(
        t,
        device=mesh_device,
        layout=layout,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims),
    )
    tt_q = to_dev(q, shard_qkv, ttnn.TILE_LAYOUT, ttnn.bfloat16)
    tt_k = to_dev(k, shard_qkv, ttnn.TILE_LAYOUT, ttnn.bfloat16)
    tt_v = to_dev(v, shard_qkv, ttnn.TILE_LAYOUT, ttnn.bfloat16)
    tt_idx = to_dev(idx.to(torch.uint32).view(torch.int32), shard_qkv, ttnn.ROW_MAJOR_LAYOUT, ttnn.uint32)
    replicate = [None, None]
    tt_counts = to_dev(counts, replicate, ttnn.ROW_MAJOR_LAYOUT, ttnn.uint32)
    tt_mask = to_dev(mask.to(torch.int32).reshape(1, 1, 1, dense_words), replicate, ttnn.ROW_MAJOR_LAYOUT, ttnn.uint32)
    # gathered buffers: [1, heads_local, t_total, d] per device (heads sharded, sequence replicated)
    tt_gk = to_dev(
        torch.zeros(1, heads_total, t_total, DIM, dtype=torch.bfloat16), shard_heads, ttnn.TILE_LAYOUT, ttnn.bfloat16
    )
    tt_gv = to_dev(
        torch.zeros(1, heads_total, t_total, DIM, dtype=torch.bfloat16), shard_heads, ttnn.TILE_LAYOUT, ttnn.bfloat16
    )
    # plain vsa_sdpa reference input: the full K/V on every device
    tt_kfull = to_dev(k, shard_heads, ttnn.TILE_LAYOUT, ttnn.bfloat16)
    tt_vfull = to_dev(v, shard_heads, ttnn.TILE_LAYOUT, ttnn.bfloat16)

    grid = mesh_device.compute_with_storage_grid_size()
    ccl_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sem_sets = [[ttnn.create_global_semaphore(mesh_device, ccl_crs, 0) for _ in range(2)] for _ in range(2)]
    common = dict(
        list_len=k_sel,
        exempt_ids=exempt_ids,
        dense_row_mask=tt_mask,
        dense_row_hint=sorted(dense_local_rows),  # local q-tile rows (a superset over shards in the model)
    )

    def ring(sems):
        return ttnn.transformer.vsa_ring_sdpa(
            tt_q,
            tt_k,
            tt_v,
            tt_idx,
            tt_counts,
            tt_gk,
            tt_gv,
            multi_device_global_semaphore=sems,
            num_links=num_links,
            cluster_axis=sp_axis,
            mesh_device=mesh_device,
            topology=ttnn.Topology.Ring,
            ccl_core_grid_offset=(0, 0),
            **common,
        )

    compose = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=shard_qkv)
    out_a = ring(sem_sets[0])
    ttnn.synchronize_device(mesh_device)
    out_a_t = ttnn.to_torch(out_a, mesh_composer=compose).float()
    out_b = ring(sem_sets[1])  # program-cache hit with the other semaphore set
    ttnn.synchronize_device(mesh_device)
    out_b_t = ttnn.to_torch(out_b, mesh_composer=compose).float()
    assert torch.equal(out_a_t, out_b_t), "vsa_ring_sdpa is not deterministic across runs / semaphore sets"

    # (b) plain vsa_sdpa on the full K/V, same indices: identical math up to visit order
    ref_dev = ttnn.transformer.vsa_sdpa(tt_q, tt_kfull, tt_vfull, tt_idx, tt_counts, streaming=True, **common)
    ref_dev_t = ttnn.to_torch(ref_dev, mesh_composer=compose).float()
    ok_dev, pcc_dev = comp_pcc(ref_dev_t, out_a_t, 0.999)
    logger.info(f"vsa_ring_sdpa vs vsa_sdpa(gathered): pcc={pcc_dev}")

    # (a) torch reference over the assembled block sets
    rows = _assembled_rows(idx, exempt_ids, k_sel, n_blocks, dense_global)
    ref_t = fine_attention_ref(q.float(), k.float(), v.float(), rows, counts.reshape(-1)[:n_blocks].to(torch.int64))
    ok_ref, pcc_ref = comp_pcc(ref_t, out_a_t, 0.99)
    logger.info(f"vsa_ring_sdpa vs torch reference: pcc={pcc_ref}")
    assert ok_dev, f"vs vsa_sdpa pcc {pcc_dev}"
    assert ok_ref, f"vs torch pcc {pcc_ref}"

    if trace:
        # capture with set 0, replay twice; the program-cache hit inside the trace must re-apply the addresses
        ring(sem_sets[0])
        ttnn.synchronize_device(mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        out_tr = ring(sem_sets[0])
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        for _ in range(2):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            out_tr_t = ttnn.to_torch(out_tr, mesh_composer=compose).float()
            assert torch.equal(out_tr_t, out_a_t), "traced replay differs from the eager result"
        ttnn.release_trace(mesh_device, tid)


@GALAXY_4X8
@pytest.mark.parametrize("trace", [False, True], ids=["eager", "trace"])
def test_vsa_ring_sdpa_small(mesh_device, sp_axis, tp_axis, num_links, trace, reset_seeds):
    skip_if_unsupported_num_links(mesh_device, num_links)
    _run(
        mesh_device,
        sp_axis,
        tp_axis,
        num_links,
        heads_total=8,
        blocks_per_shard=6,
        k_sel=9,
        dense_local_rows=[1],
        seed=0,
        trace=trace,
    )


@GALAXY_4X8
def test_vsa_ring_sdpa_medium(mesh_device, sp_axis, tp_axis, num_links, reset_seeds):
    skip_if_unsupported_num_links(mesh_device, num_links)
    _run(
        mesh_device,
        sp_axis,
        tp_axis,
        num_links,
        heads_total=56,
        blocks_per_shard=40,
        k_sel=48,
        dense_local_rows=[3, 17],
        seed=1,
        trace=False,
    )
