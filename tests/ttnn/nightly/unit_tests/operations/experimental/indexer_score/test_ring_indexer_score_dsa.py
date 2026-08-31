# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Correctness of the ring-fused indexer_score op (ttnn.experimental.ring_indexer_score_dsa) on Blackhole.
The legacy suite covers the LoudBox 2x4 -> 1x4 axis ring; full-mesh coverage uses the complete 2x4 as one
snake and adds exact-physical 2x2 plus opt-in 8x4 Galaxy gates. One op co-schedules the ring_attention
all-gather with the score; the reader gates each K band on only the SP shards it touches and dual-sources its
own slab from k_local. Checked against the same DSA references as the two-op path, including both K layouts,
indexed caches, straddle, kv_len, program-cache reuse, placement, and host validation.

Run:  scripts/run_safe_pytest.sh tests/ttnn/nightly/unit_tests/operations/experimental/indexer_score/test_ring_indexer_score_dsa.py
"""

import os

import pytest
import torch
from loguru import logger

import ttnn

from tests.ttnn.nightly.unit_tests.operations.experimental.indexer_score.test_indexer_score import (
    assert_indexer_match,
    glx_config,
    indexer_score_dsa_ref,
    _global_inputs,
    _nd_sharded_dram_config,
    _per_sp_ref,
    _straddle_ref,
    _to_slab,
    QB_HISTORY,
    QB_SQ,
    QB_CASES,
    QB_IDS,
    ST_CHUNK,
    ST_CS,
    ST_T,
)
from tests.ttnn.nightly.unit_tests.operations.experimental.indexer_score.ring_indexer_score_test_utils import (
    _open_ring4_ccl,
    _close_ring4_ccl,
    _persistent_buffer,
    _shard_k,
    RING,
    SP_AXIS,
    CHUNK_GLOBAL,
    T,
)

pytestmark = [
    pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="indexer_score is Blackhole-only"),
    pytest.mark.skipif(ttnn.get_num_devices() < 8, reason="ring-of-4 needs the 8-chip LoudBox (2x4)"),
]


def _fused_dev_inputs(submesh, q_g, w_g, k_host, *, k_dtype=ttnn.bfloat16):
    """Fused op inputs: SP-shard q/w (bf16) on dim 2, SP-shard k_local (the AG input), and a zero-seeded
    gathered buffer (AG fills remote bands; zeros prove the reader dual-sources the local band). k_dtype sets
    both k_local and k_gathered (the op requires them equal)."""
    shard = ttnn.ShardTensorToMesh(submesh, dim=2)
    q_dev = ttnn.from_torch(q_g, device=submesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)
    w_dev = ttnn.from_torch(w_g, device=submesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)
    k_local = _shard_k(submesh, k_host, dtype=k_dtype)  # [B,1,sll,D] per chip (the all-gather INPUT)
    # Indexed mode gathers one selected input slot into slot 0; batch-1 scratch also covers the ordinary B=1 path.
    k_gathered = _persistent_buffer(submesh, torch.zeros_like(k_host[:1]), dtype=k_dtype)
    return q_dev, w_dev, k_local, k_gathered


def _open_full_mesh_ccl(mesh_shape):
    """Open the complete physical 2D mesh with the torus links needed by the snake's closing edge."""
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    mesh = None
    try:
        mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape))
        grid = mesh.compute_with_storage_grid_size()
        ccl_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        worker_sub_device = ttnn.SubDevice([ccl_crs])
        worker_sub_device_id = ttnn.SubDeviceId(0)
        stall_group = [worker_sub_device_id]
        manager = mesh.create_sub_device_manager([worker_sub_device], 0)
        mesh.load_sub_device_manager(manager)
        mesh.set_sub_device_stall_group(stall_group)
        semaphores = [ttnn.create_global_semaphore(mesh, ccl_crs, 0) for _ in range(2)]
        return mesh, semaphores, worker_sub_device_id, stall_group
    except Exception:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        raise


def _close_full_mesh_ccl(mesh):
    try:
        try:
            mesh.reset_sub_device_stall_group()
            mesh.clear_loaded_sub_device_manager()
        finally:
            ttnn.close_mesh_device(mesh)
    finally:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _full_mesh_inputs(mesh, q_g, w_g, k_host, *, k_dtype=ttnn.bfloat16):
    """Canonical flat row-major sequence shards plus a complete-mesh replicated gather scratch."""
    shard = ttnn.ShardTensorToMesh(mesh, dim=2)
    q_dev = ttnn.from_torch(q_g, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)
    w_dev = ttnn.from_torch(w_g, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)
    k_local = ttnn.from_torch(k_host, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=k_dtype, mesh_mapper=shard)
    k_gathered = ttnn.from_torch(
        torch.zeros_like(k_host[:1]),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=k_dtype,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    return q_dev, w_dev, k_local, k_gathered


def _linear_full_mesh_ref(q_g, k_g, w_g, ring_size, local_sq, chunk_start):
    refs = []
    for tensor_rank in range(ring_size):
        sl = slice(tensor_rank * local_sq, (tensor_rank + 1) * local_sq)
        refs.append(
            indexer_score_dsa_ref(q_g[:, :, sl, :], k_g, w_g[:, :, sl, :], chunk_start + tensor_rank * local_sq)
        )
    return torch.cat(refs, dim=2)


def _assert_remote_gather_slots(k_local, k_gathered, ring_size, valid_local_rows=None, cache_batch_idx=None):
    """Every remote transport shard must land in its canonical row-major tensor slot."""
    local_shards = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(k_local)]
    gathered_shards = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(k_gathered)]
    assert len(local_shards) == len(gathered_shards) == ring_size
    local_rows = local_shards[0].shape[2]
    valid_rows = local_rows if valid_local_rows is None else valid_local_rows
    for destination_rank, gathered in enumerate(gathered_shards):
        for tensor_rank, local in enumerate(local_shards):
            if tensor_rank == destination_rank:
                continue  # the fused reader may direct-source its optimized local slot
            start = tensor_rank * local_rows
            expected = local if cache_batch_idx is None else local[cache_batch_idx : cache_batch_idx + 1]
            assert torch.equal(
                gathered[:, :, start : start + valid_rows, :], expected[:, :, :valid_rows, :]
            ), f"destination {destination_rank} stores tensor rank {tensor_rank} in the wrong K slot"


def _run_fused(
    heads,
    *,
    block_cyclic,
    num_links=1,
    k_dtype=ttnn.bfloat16,
):
    """Run the one fused op and check vs the per-SP reference. num_links only changes fabric routing, never
    the gathered result -> same reference."""
    submesh, parent, ccl_semaphores, subdevice_id, stall_group = _open_ring4_ccl()
    try:
        q_g, k_nat, w_g = _global_inputs(heads, CHUNK_GLOBAL, T, seed=42)
        k_host = _to_slab(k_nat, RING, CHUNK_GLOBAL) if block_cyclic else k_nat
        q_dev, w_dev, k_local, k_gathered = _fused_dev_inputs(submesh, q_g, w_g, k_host, k_dtype=k_dtype)

        bc_kwargs = dict(block_cyclic_sp_axis=SP_AXIS, block_cyclic_chunk_local=QB_SQ) if block_cyclic else {}
        out = ttnn.experimental.ring_indexer_score_dsa(
            q_dev,
            k_gathered,
            w_dev,
            k_local,
            ccl_semaphores,
            cluster_axis=SP_AXIS,
            topology=ttnn.Topology.Linear,
            num_links=num_links,
            ag_sub_device_id=subdevice_id,
            program_config=glx_config(heads),
            **bc_kwargs,
        )
        ttnn.synchronize_device(submesh, sub_device_ids=stall_group)
        out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=2))

        ref = _per_sp_ref(q_g, k_nat, w_g, RING, QB_HISTORY)
        assert_indexer_match(out_t, ref, CHUNK_GLOBAL, T, check_neg=True)
        layout = "block_cyclic" if block_cyclic else "contiguous"
        logger.info(f"ring4 fused {layout} (heads={heads}): fused all-gather + dual-source score matched reference")
    finally:
        _close_ring4_ccl(parent, submesh, stall_group)


@pytest.mark.parametrize("block_cyclic", [False, True], ids=["contiguous", "block_cyclic"])
@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_ring4_fused(case_id, heads, block_cyclic):
    """Base fused path, num_links=2 (the production Blackhole link count)."""
    _run_fused(heads, block_cyclic=block_cyclic, num_links=2)


@pytest.mark.parametrize("block_cyclic", [False, True], ids=["contiguous", "block_cyclic"])
@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_ring4_fused_bfp8_k(case_id, heads, block_cyclic):
    """Production dtype: bfloat8_b K (local shard + gathered buffer), q/w stay bf16. Same PCC floor."""
    _run_fused(heads, block_cyclic=block_cyclic, num_links=1, k_dtype=ttnn.bfloat8_b)


@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_ring4_fused_production_shape(case_id, heads):
    """All production knobs at once (each covered alone elsewhere): block-cyclic + non-zero chunk_start +
    kv_len < T_alloc + num_links=2 + bfloat8_b K. Guards their interaction (bfp8 gathered buffer + kv_len tail
    mask + block-cyclic invP on the nl2 schedule), which the model always drives together."""
    chunk_start = CHUNK_GLOBAL  # a later prefill chunk (rank r attends to chunk_start + (r+1)*QB_SQ)
    kv_len = chunk_start + CHUNK_GLOBAL  # fullest rank's causal window == kv_len exactly (validate's tightest bound)
    t_alloc = 4 * CHUNK_GLOBAL  # over-allocate so kv_len < T_alloc (ring-divisible, tile-aligned)
    submesh, parent, ccl_semaphores, subdevice_id, stall_group = _open_ring4_ccl()
    try:
        q_g, k_nat, w_g = _global_inputs(heads, CHUNK_GLOBAL, t_alloc, seed=42)
        k_bc = _to_slab(k_nat, RING, CHUNK_GLOBAL)  # block-cyclic physical layout the reader inverts
        # bfloat8_b K (the model's cache dtype) for both the local shard and the gathered buffer.
        q_dev, w_dev, k_local, k_gathered = _fused_dev_inputs(submesh, q_g, w_g, k_bc, k_dtype=ttnn.bfloat8_b)

        out = ttnn.experimental.ring_indexer_score_dsa(
            q_dev,
            k_gathered,
            w_dev,
            k_local,
            ccl_semaphores,
            cluster_axis=SP_AXIS,
            topology=ttnn.Topology.Linear,
            num_links=2,
            ag_sub_device_id=subdevice_id,
            chunk_start_idx=chunk_start,
            kv_len=kv_len,
            block_cyclic_sp_axis=SP_AXIS,
            block_cyclic_chunk_local=QB_SQ,
            program_config=glx_config(heads),
        )
        ttnn.synchronize_device(submesh, sub_device_ids=stall_group)
        out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=2))

        # Only [0, kv_len) is valid; each SP rank scores the valid key prefix at the non-zero chunk_start.
        ref = _per_sp_ref(q_g, k_nat[:, :, :kv_len, :], w_g, RING, chunk_start)
        assert_indexer_match(out_t[:, :, :, :kv_len], ref, CHUNK_GLOBAL, kv_len, check_neg=True)
        logger.info(
            f"ring4 fused production-shape (heads={heads}): block_cyclic+cs={chunk_start}+kv_len={kv_len}+nl2+bfp8 "
            f"matched reference"
        )
    finally:
        _close_ring4_ccl(parent, submesh, stall_group)


def _run_fused_multiuser(heads, *, num_users, cache_batch_idx, num_links=1):
    """Multi-user indexed cache: k_local [num_users,1,sll,D] and batch-1 gathered scratch. cache_batch_idx
    selects the single gathered slot and the reader applies the corresponding local-cache offset. Distinct
    user K values make either a gather-slot or local-slot addressing error fail PCC."""
    submesh, parent, ccl_semaphores, subdevice_id, stall_group = _open_ring4_ccl()
    try:
        # Shared q/w scoring distinct per-user caches (distinct seed per slot -> a wrong-slot read changes the score).
        q_g, _, w_g = _global_inputs(heads, CHUNK_GLOBAL, T, seed=42)
        k_multi = torch.cat([_global_inputs(heads, CHUNK_GLOBAL, T, seed=100 + u)[1] for u in range(num_users)], dim=0)
        q_dev, w_dev, k_local, k_gathered = _fused_dev_inputs(submesh, q_g, w_g, k_multi)

        out = ttnn.experimental.ring_indexer_score_dsa(
            q_dev,
            k_gathered,
            w_dev,
            k_local,
            ccl_semaphores,
            cluster_axis=SP_AXIS,
            topology=ttnn.Topology.Linear,
            num_links=num_links,
            ag_sub_device_id=subdevice_id,
            cache_batch_idx=cache_batch_idx,
            program_config=glx_config(heads),
        )
        ttnn.synchronize_device(submesh, sub_device_ids=stall_group)
        out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=2))

        ref = _per_sp_ref(q_g, k_multi[cache_batch_idx : cache_batch_idx + 1], w_g, RING, QB_HISTORY)
        assert_indexer_match(out_t, ref, CHUNK_GLOBAL, T, check_neg=True)
        logger.info(
            f"ring4 fused multi-user (heads={heads}, users={num_users}, slot={cache_batch_idx}): matched reference"
        )
    finally:
        _close_ring4_ccl(parent, submesh, stall_group)


def test_indexer_score_ring4_fused_indexed_cache():
    """cache_batch_idx=1 (2nd user slot). One representative case -- the slot offset is head-independent."""
    _run_fused_multiuser(16, num_users=2, cache_batch_idx=1)


@pytest.mark.parametrize("rows_per_shard", [32, 96], ids=["prod_rows32", "padded_rows96"])
def test_indexer_score_ring4_fused_nd_indexed_bounded_gather_cache_hit(rows_per_shard):
    """Production cache contract in one regression:

    * multi-slot k_local is ND-sharded across DRAM banks;
    * the gather selects one slot into a batch-1 scratch;
    * kv_len bounds transport to complete touched block-cyclic slabs; and
    * a second dispatch changes both slot and kv_len on the same cached program; and
    * the production two-link all-gather partition preserves those cache-hit results.
    """
    heads, num_users = 16, 3
    t_alloc = 4 * CHUNK_GLOBAL
    local_t = t_alloc // RING
    submesh, parent, ccl_semaphores, subdevice_id, stall_group = _open_ring4_ccl()
    try:
        q_g, _, w_g = _global_inputs(heads, CHUNK_GLOBAL, t_alloc, seed=42)
        k_nat = torch.cat(
            [_global_inputs(heads, CHUNK_GLOBAL, t_alloc, seed=100 + u)[1] for u in range(num_users)], dim=0
        )
        k_bc = torch.cat([_to_slab(k_nat[u : u + 1], RING, CHUNK_GLOBAL) for u in range(num_users)], dim=0)
        q_dev, w_dev, k_local_i, k_gathered = _fused_dev_inputs(submesh, q_g, w_g, k_bc, k_dtype=ttnn.bfloat8_b)
        k_local = ttnn.to_memory_config(k_local_i, _nd_sharded_dram_config(submesh, rows_per_shard=rows_per_shard))
        ttnn.deallocate(k_local_i)

        def _score(slot, chunk_start, kv_len):
            out = ttnn.experimental.ring_indexer_score_dsa(
                q_dev,
                k_gathered,
                w_dev,
                k_local,
                ccl_semaphores,
                cluster_axis=SP_AXIS,
                topology=ttnn.Topology.Linear,
                num_links=2,
                ag_sub_device_id=subdevice_id,
                chunk_start_idx=chunk_start,
                cache_batch_idx=slot,
                kv_len=kv_len,
                block_cyclic_sp_axis=SP_AXIS,
                block_cyclic_chunk_local=QB_SQ,
                program_config=glx_config(heads),
            )
            ttnn.synchronize_device(submesh, sub_device_ids=stall_group)
            return ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=2))

        # A tile past the first global-slab boundary requires TWO complete local slabs per rank. This
        # specifically exercises ceil(kv_len/chunk_global), not just exact-boundary truncation.
        large_kv_len = CHUNK_GLOBAL + 32
        out0 = _score(slot=1, chunk_start=32, kv_len=large_kv_len)
        entries_after_first = submesh.num_program_cache_entries()
        scratch_after_large = [ttnn.to_torch(t).clone() for t in ttnn.get_device_tensors(k_gathered.cpu())]
        ref0 = _straddle_ref(q_g, k_nat[1:2, :, :large_kv_len, :], w_g, RING, CHUNK_GLOBAL, 32, large_kv_len)
        assert_indexer_match(out0[:, :, :, :large_kv_len], ref0, CHUNK_GLOBAL, large_kv_len, check_neg=True)

        valid_local_large = 2 * QB_SQ
        for scratch_t in scratch_after_large:
            for rank in range(RING):
                tail = scratch_t[:, :, rank * local_t + valid_local_large : (rank + 1) * local_t, :]
                assert torch.count_nonzero(tail) == 0, "gather wrote beyond the slab-rounded kv_len extent"

        # Shrink the extent and switch users. Both are runtime values: this must reuse the same compiled
        # program, overwrite only slab 0 from slot 2, and leave slab 1 exactly as slot 1 wrote it.
        out1 = _score(slot=2, chunk_start=0, kv_len=CHUNK_GLOBAL)
        assert (
            submesh.num_program_cache_entries() == entries_after_first
        ), "slot/kv_len change recompiled instead of exercising the cache-hit runtime-arg patch"
        scratch_after_small = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(k_gathered.cpu())]
        ref1 = _per_sp_ref(q_g, k_nat[2:3, :, :CHUNK_GLOBAL, :], w_g, RING, 0)
        assert_indexer_match(out1[:, :, :, :CHUNK_GLOBAL], ref1, CHUNK_GLOBAL, CHUNK_GLOBAL, check_neg=True)

        any_first_slab_changed = False
        for before, after in zip(scratch_after_large, scratch_after_small):
            for rank in range(RING):
                base = rank * local_t
                any_first_slab_changed |= not torch.equal(
                    before[:, :, base : base + QB_SQ, :], after[:, :, base : base + QB_SQ, :]
                )
                assert torch.equal(
                    before[:, :, base + QB_SQ : base + 2 * QB_SQ, :],
                    after[:, :, base + QB_SQ : base + 2 * QB_SQ, :],
                ), "shrinking kv_len rewrote the second slab on a cache hit"
        assert any_first_slab_changed, "switching cache slots did not update any gathered first-slab data"
    finally:
        _close_ring4_ccl(parent, submesh, stall_group)


@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_ring4_fused_straddle(case_id, heads):
    """Mid-slab straddle + block-cyclic rotation (rotated-prefill/multiturn): a non-slab-aligned chunk_start
    (704) makes the boundary chip's queries cross a slab boundary, so the causal diagonal jumps by
    (chunk_global - cl). Proves the band reorder + per-band gate + dual-source read compose with the straddled
    mask. Checked vs the per-SP rotated reference."""
    cl = ST_CHUNK // RING  # per-shard chunk / per-device query rows (block-cyclic SP-only)
    submesh, parent, ccl_semaphores, subdevice_id, stall_group = _open_ring4_ccl()
    try:
        q_g, k_nat, w_g = _global_inputs(heads, ST_CHUNK, ST_T, seed=42)
        k_bc = _to_slab(k_nat, RING, ST_CHUNK)  # block-cyclic physical layout the reader inverts

        q_dev, w_dev, k_local, k_gathered = _fused_dev_inputs(submesh, q_g, w_g, k_bc)  # [1,1,ST_T/RING,D]

        out = ttnn.experimental.ring_indexer_score_dsa(
            q_dev,
            k_gathered,
            w_dev,
            k_local,
            ccl_semaphores,
            cluster_axis=SP_AXIS,
            topology=ttnn.Topology.Linear,
            num_links=1,
            ag_sub_device_id=subdevice_id,
            chunk_start_idx=ST_CS,  # mid-slab (704 % cl != 0) -> rotation + straddle
            block_cyclic_sp_axis=SP_AXIS,
            block_cyclic_chunk_local=cl,
            program_config=glx_config(heads),
        )
        ttnn.synchronize_device(submesh, sub_device_ids=stall_group)
        out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=2))

        ref = _straddle_ref(q_g, k_nat, w_g, RING, ST_CHUNK, ST_CS, ST_T)
        assert_indexer_match(out_t, ref, ST_CHUNK, ST_T, check_neg=True)
        logger.info(f"ring4 fused straddle (heads={heads}): rotated-prefill causal diagonal matched reference")
    finally:
        _close_ring4_ccl(parent, submesh, stall_group)


def test_indexer_score_ring4_fused_runtime_kv_len():
    """Padded cache: k allocated at T_alloc but only a kv_len prefix is valid; only cols [0, kv_len) are
    written. Confirms the AG gathers full T_alloc and the compute masks beyond kv_len (band_count spans full T,
    so no shard is left un-delivered). heads=16 representative -- kv_len masking is head-independent."""
    heads = 16
    kv_len = QB_HISTORY + CHUNK_GLOBAL  # valid written extent (28160 keys, 880 tiles)
    t_alloc = kv_len + CHUNK_GLOBAL  # over-allocate one more global chunk (30720, ring-divisible, tile-aligned)
    submesh, parent, ccl_semaphores, subdevice_id, stall_group = _open_ring4_ccl()
    try:
        q_g, k_nat, w_g = _global_inputs(heads, CHUNK_GLOBAL, t_alloc, seed=42)
        q_dev, w_dev, k_local, k_gathered = _fused_dev_inputs(submesh, q_g, w_g, k_nat)  # [1,1,t_alloc/RING,D]

        out = ttnn.experimental.ring_indexer_score_dsa(
            q_dev,
            k_gathered,
            w_dev,
            k_local,
            ccl_semaphores,
            cluster_axis=SP_AXIS,
            topology=ttnn.Topology.Linear,
            num_links=1,
            ag_sub_device_id=subdevice_id,
            chunk_start_idx=QB_HISTORY,  # rank r attends up to QB_HISTORY + (r+1)*QB_SQ; fullest = kv_len exactly
            kv_len=kv_len,
            program_config=glx_config(heads),
        )
        ttnn.synchronize_device(submesh, sub_device_ids=stall_group)
        out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=2))

        # Only [0, kv_len) is valid; reference scores each rank against the valid key prefix (rest is stale tail).
        ref = _per_sp_ref(q_g, k_nat[:, :, :kv_len, :], w_g, RING, QB_HISTORY)
        assert_indexer_match(out_t[:, :, :, :kv_len], ref, CHUNK_GLOBAL, kv_len, check_neg=True)
        logger.info(
            f"ring4 fused runtime kv_len (kv_len={kv_len} of T_alloc={t_alloc}): valid prefix matched reference"
        )
    finally:
        _close_ring4_ccl(parent, submesh, stall_group)


@pytest.mark.parametrize("k_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bfp8"])
def test_indexer_score_ring4_fused_program_cache_reuse(k_dtype):
    """Two dispatches, identical shapes but different chunk_start/kv_len on the SAME device (2nd is a cache
    hit). chunk_start/kv_len and fused-AG semaphore identity are hash-excluded, so
    override_runtime_arguments must re-apply them; if not, the 2nd dispatch reuses the 1st's frozen offset or
    semaphore addresses. Regression guard for the program-cache stale-runtime-argument bugs (every other test
    dispatches cold). Both bf16 and production bfp8_b K."""
    heads = 16  # the scalar re-patch is head-independent, so one head count suffices (both dtypes kept)
    t_alloc = 4 * CHUNK_GLOBAL  # room for both chunks' causal windows (global block == CHUNK_GLOBAL)
    submesh, parent, ccl_semaphores, subdevice_id, stall_group = _open_ring4_ccl()
    try:
        q_g, k_nat, w_g = _global_inputs(heads, CHUNK_GLOBAL, t_alloc, seed=42)
        k_bc = _to_slab(k_nat, RING, CHUNK_GLOBAL)  # block-cyclic physical layout
        q_dev, w_dev, k_local, k_gathered = _fused_dev_inputs(submesh, q_g, w_g, k_bc, k_dtype=k_dtype)

        # A physically distinct pair exercises the semaphore-address cache-hit override. This is the same
        # A/B rotation used by model TT_CCL; changing only the addresses must not create another program.
        grid = submesh.compute_with_storage_grid_size()
        ccl_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        alternate_semaphores = [ttnn.create_global_semaphore(submesh, ccl_crs, 0) for _ in range(2)]

        def _score(chunk_start, kv_len, semaphores):  # identical shapes each call -> 2nd is a program-cache hit
            out = ttnn.experimental.ring_indexer_score_dsa(
                q_dev,
                k_gathered,
                w_dev,
                k_local,
                semaphores,
                cluster_axis=SP_AXIS,
                topology=ttnn.Topology.Linear,
                num_links=1,
                ag_sub_device_id=subdevice_id,
                chunk_start_idx=chunk_start,
                block_cyclic_sp_axis=SP_AXIS,
                block_cyclic_chunk_local=QB_SQ,
                kv_len=kv_len,
                program_config=glx_config(heads),
            )
            ttnn.synchronize_device(submesh, sub_device_ids=stall_group)
            return ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=2))

        # chunk@0 (cache miss/build) then chunk@CHUNK_GLOBAL (cache HIT -- must re-apply chunk_start + kv_len).
        out0 = _score(chunk_start=0, kv_len=CHUNK_GLOBAL, semaphores=ccl_semaphores)
        entries_after_first = submesh.num_program_cache_entries()
        out1 = _score(
            chunk_start=CHUNK_GLOBAL,
            kv_len=2 * CHUNK_GLOBAL,
            semaphores=alternate_semaphores,
        )
        assert (
            submesh.num_program_cache_entries() == entries_after_first
        ), "alternating fused-AG semaphore addresses must reuse the cached ring-indexer program"
        ref0 = _per_sp_ref(q_g, k_nat[:, :, :CHUNK_GLOBAL, :], w_g, RING, 0)
        ref1 = _per_sp_ref(q_g, k_nat[:, :, : 2 * CHUNK_GLOBAL, :], w_g, RING, CHUNK_GLOBAL)
        assert_indexer_match(out0[:, :, :, :CHUNK_GLOBAL], ref0, CHUNK_GLOBAL, CHUNK_GLOBAL, check_neg=True)
        assert_indexer_match(out1[:, :, :, : 2 * CHUNK_GLOBAL], ref1, CHUNK_GLOBAL, 2 * CHUNK_GLOBAL, check_neg=True)
        logger.info(
            f"ring4 fused program-cache reuse (heads={heads}): 2nd chunk_start and semaphore pair re-applied on cache hit"
        )
    finally:
        _close_ring4_ccl(parent, submesh, stall_group)


def _run_full_mesh_accuracy_case(mesh_shape, *, block_cyclic):
    """Exercise a non-identity snake permutation while retaining row-major causal and K-slot semantics."""
    ring_size = mesh_shape[0] * mesh_shape[1]
    local_sq = 64
    chunk_global = ring_size * local_sq
    if block_cyclic:
        # Enter slab 1, rotate ownership by one tensor rank, and make exactly that boundary rank straddle.
        chunk_start = chunk_global + local_sq + 32
        t_len = 3 * chunk_global
    else:
        chunk_start = chunk_global
        t_len = 2 * chunk_global

    mesh, semaphores, subdevice_id, stall_group = _open_full_mesh_ccl(mesh_shape)
    try:
        heads = 8
        q_g, k_nat, w_g = _global_inputs(heads, chunk_global, t_len, seed=2026)
        k_host = _to_slab(k_nat, ring_size, chunk_global) if block_cyclic else k_nat
        q_dev, w_dev, k_local, k_gathered = _full_mesh_inputs(mesh, q_g, w_g, k_host)
        kwargs = {"block_cyclic_chunk_local": local_sq} if block_cyclic else {}

        mesh.enable_program_cache()
        mesh.clear_program_cache()
        outputs = []
        entries_after_first = None
        for _ in range(2):
            out = ttnn.experimental.ring_indexer_score_dsa(
                q_dev,
                k_gathered,
                w_dev,
                k_local,
                semaphores,
                cluster_axis=None,
                topology=ttnn.Topology.Ring,
                num_links=2,
                ag_sub_device_id=subdevice_id,
                chunk_start_idx=chunk_start,
                program_config=glx_config(heads),
                **kwargs,
            )
            ttnn.synchronize_device(mesh, sub_device_ids=stall_group)
            outputs.append(ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2)))
            entries = mesh.num_program_cache_entries()
            if entries_after_first is None:
                assert entries > 0
                entries_after_first = entries
                _assert_remote_gather_slots(k_local, k_gathered, ring_size)
            else:
                assert entries == entries_after_first, "full-mesh replay added program-cache entries"

        assert torch.equal(outputs[0], outputs[1]), "full-mesh indexer replay is not bit-exact"
        if block_cyclic:
            ref = _straddle_ref(q_g, k_nat, w_g, ring_size, chunk_global, chunk_start, t_len)
        else:
            ref = _linear_full_mesh_ref(q_g, k_nat, w_g, ring_size, local_sq, chunk_start)
        assert_indexer_match(outputs[0], ref, chunk_global, t_len, check_neg=True)
        logger.info(
            f"full-mesh indexer {mesh_shape} {'block-cyclic rotated' if block_cyclic else 'contiguous'}: "
            "PCC, deterministic replay, cache reuse, and canonical remote K placement passed"
        )
    finally:
        if mesh is not None:
            mesh.disable_and_clear_program_cache()
        _close_full_mesh_ccl(mesh)


@pytest.mark.parametrize("block_cyclic", [False, True], ids=["contiguous", "block_cyclic_rotated"])
def test_indexer_score_full_mesh_loudbox_accuracy_placement_and_cache_reuse(block_cyclic):
    """Use every device on the physical 2x4 LoudBox as one eight-rank snake ring."""
    if ttnn.get_num_devices() != 8:
        pytest.skip("2x4 full-mesh indexer coverage requires the exact physical eight-device LoudBox")
    _run_full_mesh_accuracy_case((2, 4), block_cyclic=block_cyclic)


@pytest.mark.skipif(
    not os.getenv("TT_METAL_SIMULATOR")
    and (os.getenv("MESH_DEVICE") != "TG" or os.getenv("TT_METAL_RING_INDEXER_RUN_32_RANK_ACCURACY") != "1"),
    reason="requires Galaxy/simulator opt-in for the 32-rank complete-mesh indexer test",
)
def test_indexer_score_full_mesh_galaxy_8x4_accuracy():
    """Exercise the fixed 32-entry readiness tables at their supported Galaxy limit."""
    if ttnn.get_num_devices() != 32:
        pytest.skip("8x4 full-mesh indexer coverage requires exactly 32 available devices")
    _run_full_mesh_accuracy_case((8, 4), block_cyclic=False)


def test_indexer_score_full_mesh_indexed_bounded_gather_cache_hit_and_determinism():
    """Combine indexed ND-sharded K, bounded transport, rotated causal patching, and cache reuse."""
    if ttnn.get_num_devices() != 8:
        pytest.skip("complete 2x4 cache-hit coverage requires the exact physical eight-device LoudBox")

    mesh_shape = (2, 4)
    ring_size = 8
    heads, num_users, local_sq = 8, 3, 64
    chunk_global = ring_size * local_sq
    t_alloc = 3 * chunk_global
    local_t = t_alloc // ring_size
    mesh, semaphores, subdevice_id, stall_group = _open_full_mesh_ccl(mesh_shape)
    try:
        q_g, _, w_g = _global_inputs(heads, chunk_global, t_alloc, seed=2027)
        k_nat = torch.cat(
            [_global_inputs(heads, chunk_global, t_alloc, seed=2100 + user)[1] for user in range(num_users)], dim=0
        )
        k_bc = torch.cat(
            [_to_slab(k_nat[user : user + 1], ring_size, chunk_global) for user in range(num_users)], dim=0
        )
        q_dev, w_dev, k_local_i, k_gathered = _full_mesh_inputs(mesh, q_g, w_g, k_bc)
        k_local = ttnn.to_memory_config(k_local_i, _nd_sharded_dram_config(mesh, rows_per_shard=32))
        ttnn.deallocate(k_local_i)

        mesh.enable_program_cache()
        mesh.clear_program_cache()

        def _score(slot, chunk_start, kv_len):
            out = ttnn.experimental.ring_indexer_score_dsa(
                q_dev,
                k_gathered,
                w_dev,
                k_local,
                semaphores,
                cluster_axis=None,
                topology=ttnn.Topology.Ring,
                num_links=2,
                ag_sub_device_id=subdevice_id,
                chunk_start_idx=chunk_start,
                cache_batch_idx=slot,
                kv_len=kv_len,
                block_cyclic_chunk_local=local_sq,
                program_config=glx_config(heads),
            )
            ttnn.synchronize_device(mesh, sub_device_ids=stall_group)
            return ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2))

        large_kv_len = chunk_global + 32
        out0 = _score(slot=1, chunk_start=32, kv_len=large_kv_len)
        entries_after_first = mesh.num_program_cache_entries()
        scratch_after_large = [ttnn.to_torch(tensor).clone() for tensor in ttnn.get_device_tensors(k_gathered)]
        ref0 = _straddle_ref(q_g, k_nat[1:2, :, :large_kv_len, :], w_g, ring_size, chunk_global, 32, large_kv_len)
        assert_indexer_match(out0[:, :, :, :large_kv_len], ref0, chunk_global, large_kv_len, check_neg=True)
        _assert_remote_gather_slots(
            k_local,
            k_gathered,
            ring_size,
            valid_local_rows=2 * local_sq,
            cache_batch_idx=1,
        )
        for scratch in scratch_after_large:
            for tensor_rank in range(ring_size):
                tail = scratch[:, :, tensor_rank * local_t + 2 * local_sq : (tensor_rank + 1) * local_t, :]
                assert torch.count_nonzero(tail) == 0, "bounded gather wrote beyond its slab-rounded extent"

        out1 = _score(slot=2, chunk_start=0, kv_len=chunk_global)
        out2 = _score(slot=2, chunk_start=0, kv_len=chunk_global)
        assert mesh.num_program_cache_entries() == entries_after_first, "runtime scalar changes recompiled"
        assert torch.equal(out1, out2), "cache-hit replay is not bit-exact"
        ref1 = _straddle_ref(q_g, k_nat[2:3, :, :chunk_global, :], w_g, ring_size, chunk_global, 0, chunk_global)
        assert_indexer_match(out1[:, :, :, :chunk_global], ref1, chunk_global, chunk_global, check_neg=True)
        _assert_remote_gather_slots(k_local, k_gathered, ring_size, valid_local_rows=local_sq, cache_batch_idx=2)
        scratch_after_small = [ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(k_gathered)]
        any_first_slab_changed = False
        for before, after in zip(scratch_after_large, scratch_after_small):
            for tensor_rank in range(ring_size):
                base = tensor_rank * local_t
                any_first_slab_changed |= not torch.equal(
                    before[:, :, base : base + local_sq, :], after[:, :, base : base + local_sq, :]
                )
                assert torch.equal(
                    before[:, :, base + local_sq : base + 2 * local_sq, :],
                    after[:, :, base + local_sq : base + 2 * local_sq, :],
                ), "shrinking kv_len rewrote the second slab on a cache hit"
        assert any_first_slab_changed, "switching cache slots did not update any gathered first-slab data"
    finally:
        if mesh is not None:
            mesh.disable_and_clear_program_cache()
        _close_full_mesh_ccl(mesh)


def test_indexer_score_full_mesh_rejects_invalid_contracts(expect_error):
    """Reject invalid full-mesh topology, axis roles, placements, replication, and link requests on host."""
    if ttnn.get_num_devices() != 8:
        pytest.skip("complete 2x4 negative coverage requires the exact physical eight-device LoudBox")

    mesh, semaphores, subdevice_id, _ = _open_full_mesh_ccl((2, 4))
    try:
        heads, local_sq, ring_size = 8, 64, 8
        chunk_global, t_len = ring_size * local_sq, 2 * ring_size * local_sq
        q_g, k_nat, w_g = _global_inputs(heads, chunk_global, t_len, seed=2028)
        q_dev, w_dev, k_local, k_gathered = _full_mesh_inputs(mesh, q_g, w_g, k_nat)

        def _call(**overrides):
            q_arg = overrides.pop("q", q_dev)
            k_arg = overrides.pop("k", k_gathered)
            kwargs = dict(
                cluster_axis=None,
                topology=ttnn.Topology.Ring,
                num_links=2,
                ag_sub_device_id=subdevice_id,
                chunk_start_idx=chunk_global,
                program_config=glx_config(heads),
            )
            kwargs.update(overrides)
            return ttnn.experimental.ring_indexer_score_dsa(
                q_arg,
                k_arg,
                w_dev,
                k_local,
                semaphores,
                **kwargs,
            )

        with expect_error(RuntimeError, "requires Ring topology"):
            _call(topology=ttnn.Topology.Linear)
        with expect_error(RuntimeError, "does not allow seq_subshard_axis"):
            _call(seq_subshard_axis=0)
        with expect_error(RuntimeError, "does not allow block_cyclic_sp_axis"):
            _call(block_cyclic_sp_axis=0, block_cyclic_chunk_local=local_sq)
        with expect_error(RuntimeError, "requires num_links > 0"):
            _call(num_links=0)
        with expect_error(RuntimeError, "could not resolve a direct-neighbor full-mesh snake ring"):
            _call(num_links=99)

        axis_mapper = ttnn.ShardTensor2dMesh(mesh, mesh_shape=(2, 4), dims=(None, 2))
        axis_q = ttnn.from_torch(
            q_g, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=axis_mapper
        )
        with expect_error(RuntimeError, "sequence dim 2 to be sharded across all"):
            _call(q=axis_q)

        nonreplicated_k = ttnn.from_torch(
            torch.zeros_like(k_nat),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2),
        )
        with expect_error(RuntimeError, "persistent gathered K buffer replicated"):
            _call(k=nonreplicated_k)
    finally:
        _close_full_mesh_ccl(mesh)


def test_indexer_score_ring4_fused_rejects_head_streaming(expect_error):
    """The fused path requires all heads resident; a streaming config (0 < head_group_size < Hi) must be
    rejected at validate, not silently mis-scheduled. head-independent -> one representative case."""
    heads = 16  # head_group_size=8 is a streaming config (0 < 8 < 16)
    submesh, parent, ccl_semaphores, subdevice_id, stall_group = _open_ring4_ccl()
    try:
        q_g, k_nat, w_g = _global_inputs(heads, CHUNK_GLOBAL, T, seed=42)
        q_dev, w_dev, k_local, k_gathered = _fused_dev_inputs(submesh, q_g, w_g, k_nat)
        base = glx_config(heads)
        streaming_cfg = ttnn.IndexerScoreProgramConfig(
            q_chunk_size=base.q_chunk_size, k_chunk_size=base.k_chunk_size, head_group_size=heads // 2
        )
        with expect_error(RuntimeError, "head_group_size must be 0 or Hi"):
            ttnn.experimental.ring_indexer_score_dsa(
                q_dev,
                k_gathered,
                w_dev,
                k_local,
                ccl_semaphores,
                cluster_axis=SP_AXIS,
                topology=ttnn.Topology.Linear,
                num_links=1,
                ag_sub_device_id=subdevice_id,
                program_config=streaming_cfg,
            )
    finally:
        _close_ring4_ccl(parent, submesh, stall_group)
