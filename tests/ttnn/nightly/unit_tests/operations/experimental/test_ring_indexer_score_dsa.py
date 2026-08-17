# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Correctness of the ring-fused indexer_score op (ttnn.experimental.ring_indexer_score_dsa) on the 8-chip
LoudBox (2x4 -> 1x4 submesh). One op co-schedules the ring_attention all-gather with the score; the reader
gates each K band on only the SP shards it touches and dual-sources its own slab from k_local. Checked vs the
same per-SP DSA reference the two-op path uses, over both K layouts, both head counts, and both link counts,
plus the runtime knobs (bfp8_b K, multi-user cache, straddle, kv_len, program-cache reuse, validate reject).

Run:  scripts/run_safe_pytest.sh tests/ttnn/nightly/unit_tests/operations/experimental/test_ring_indexer_score_dsa.py
"""

import pytest
import torch
from loguru import logger

import ttnn

from tests.ttnn.nightly.unit_tests.operations.experimental.test_indexer_score import (
    assert_indexer_match,
    glx_config,
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
from tests.ttnn.nightly.unit_tests.operations.experimental.ring4_ccl_helpers import (
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
    """cache_batch_idx=1 (2nd user slot). One representative case (dsv32) -- the slot offset is head-independent."""
    _run_fused_multiuser(16, num_users=2, cache_batch_idx=1)


@pytest.mark.parametrize("rows_per_shard", [32, 96], ids=["prod_rows32", "padded_rows96"])
def test_indexer_score_ring4_fused_nd_indexed_bounded_gather_cache_hit(rows_per_shard):
    """Production cache contract in one regression:

    * multi-slot k_local is ND-sharded across DRAM banks;
    * the gather selects one slot into a batch-1 scratch;
    * kv_len bounds transport to complete touched block-cyclic slabs; and
    * a second dispatch changes both slot and kv_len on the same cached program.
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
                num_links=1,
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
    hit). chunk_start/kv_len are hash-excluded, so override_runtime_arguments must re-apply them; if not, the
    2nd dispatch reuses the 1st's frozen offset -> wrong logits. Regression guard for the program-cache
    stale-scalar bug (every other test dispatches cold). Both bf16 and production bfp8_b K."""
    heads = 16  # dsv32; the scalar re-patch is head-independent, so one head count suffices (both dtypes kept)
    t_alloc = 4 * CHUNK_GLOBAL  # room for both chunks' causal windows (global block == CHUNK_GLOBAL)
    submesh, parent, ccl_semaphores, subdevice_id, stall_group = _open_ring4_ccl()
    try:
        q_g, k_nat, w_g = _global_inputs(heads, CHUNK_GLOBAL, t_alloc, seed=42)
        k_bc = _to_slab(k_nat, RING, CHUNK_GLOBAL)  # block-cyclic physical layout
        q_dev, w_dev, k_local, k_gathered = _fused_dev_inputs(submesh, q_g, w_g, k_bc, k_dtype=k_dtype)

        def _score(chunk_start, kv_len):  # identical shapes each call -> 2nd is a program-cache hit
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
                chunk_start_idx=chunk_start,
                block_cyclic_sp_axis=SP_AXIS,
                block_cyclic_chunk_local=QB_SQ,
                kv_len=kv_len,
                program_config=glx_config(heads),
            )
            ttnn.synchronize_device(submesh, sub_device_ids=stall_group)
            return ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=2))

        # chunk@0 (cache miss/build) then chunk@CHUNK_GLOBAL (cache HIT -- must re-apply chunk_start + kv_len).
        out0 = _score(chunk_start=0, kv_len=CHUNK_GLOBAL)
        out1 = _score(chunk_start=CHUNK_GLOBAL, kv_len=2 * CHUNK_GLOBAL)
        ref0 = _per_sp_ref(q_g, k_nat[:, :, :CHUNK_GLOBAL, :], w_g, RING, 0)
        ref1 = _per_sp_ref(q_g, k_nat[:, :, : 2 * CHUNK_GLOBAL, :], w_g, RING, CHUNK_GLOBAL)
        assert_indexer_match(out0[:, :, :, :CHUNK_GLOBAL], ref0, CHUNK_GLOBAL, CHUNK_GLOBAL, check_neg=True)
        assert_indexer_match(out1[:, :, :, : 2 * CHUNK_GLOBAL], ref1, CHUNK_GLOBAL, 2 * CHUNK_GLOBAL, check_neg=True)
        logger.info(f"ring4 fused program-cache reuse (heads={heads}): 2nd chunk_start re-applied on cache hit")
    finally:
        _close_ring4_ccl(parent, submesh, stall_group)


def test_indexer_score_ring4_fused_rejects_head_streaming(expect_error):
    """The fused path requires all heads resident; a streaming config (0 < head_group_size < Hi) must be
    rejected at validate, not silently mis-scheduled. head-independent -> one representative case (dsv32)."""
    heads = 16  # dsv32; head_group_size=8 is a streaming config (0 < 8 < 16)
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
