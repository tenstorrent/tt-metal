# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Correctness of the ring-fused indexer_score op (see ring_indexer_score_fusion_design.md).

ONE op (ttnn.experimental.ring_indexer_score_dsa) co-schedules the ring_attention all-gather with the indexer
compute in a single program: the reader gates each K band on ONLY the SP shards it touches (per-band, not a
coarse whole-gather barrier), dual-sourcing its own slab from k_local on device while the AG fills the remote
slabs. The gathered buffer is seeded with ZEROS, so a correct score PROVES device-side local sourcing (a stale
local band would fail the -inf map + PCC). Checked vs the same per-SP DSA reference the two-op path uses, over
both K layouts (contiguous / block-cyclic), both head counts (glm5/dsv32), and both link counts (num_links
1 and the production 2). The runtime DSA knobs are covered on the fused schedule too: bfp8_b K, the
multi-user indexed cache (cache_batch_idx), the mid-slab straddle / rotated-prefill causal diagonal
(multiturn), and a padded / over-allocated cache (runtime kv_len). A companion test asserts the
head-streaming config is rejected at validate.

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
from tests.ttnn.nightly.unit_tests.operations.experimental.test_indexer_score_lb_ring4_ag_equiv import (
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
    """Build the fused op's device inputs: SP-shard q/w (always bf16, post-RoPE) on dim 2, SP-shard the local
    K slab (k_local, the all-gather INPUT), and allocate the persistent gathered-K buffer seeded with ZEROS
    (the AG fills the REMOTE bands; the reader dual-sources the local band from k_local, so zeros prove
    device-side local sourcing). k_dtype sets BOTH k_local and k_gathered (the op requires them equal). Shared
    by the correctness tests, which then run the op and check their own per-SP reference."""
    shard = ttnn.ShardTensorToMesh(submesh, dim=2)
    q_dev = ttnn.from_torch(q_g, device=submesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)
    w_dev = ttnn.from_torch(w_g, device=submesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)
    k_local = _shard_k(submesh, k_host, dtype=k_dtype)  # [B,1,sll,D] per chip (the all-gather INPUT)
    k_gathered = _persistent_buffer(submesh, torch.zeros_like(k_host), dtype=k_dtype)  # [B,1,T,D] AG OUTPUT
    return q_dev, w_dev, k_local, k_gathered


def _run_fused(heads, *, block_cyclic, num_links=1, k_dtype=ttnn.bfloat16):
    """SP-shard q/w/k_local, seed the gathered buffer, run the ONE fused op, check vs the per-SP reference.
    num_links only changes fabric routing (AG worker count), never the gathered result -> same reference.
    k_dtype sets BOTH the local shard and the gathered buffer (the op requires them equal); q/w stay bf16."""
    submesh, parent, ccl_semaphores, subdevice_id, stall_group = _open_ring4_ccl()
    try:
        q_g, k_nat, w_g = _global_inputs(heads, CHUNK_GLOBAL, T, seed=42)
        k_host = _to_slab(k_nat, RING, CHUNK_GLOBAL) if block_cyclic else k_nat
        # k_gathered is seeded with ZEROS: the AG writes only the REMOTE bands; the reader dual-sources the
        # local band from k_local on device, so a correct score PROVES device-side local sourcing (Step D).
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
    """Single fused all-gather + indexer_score op on a ring of 4, checked vs the per-SP DSA reference.
    num_links is an AG-transport knob (not indexer math), pinned to the production Blackhole value 2
    (mla.py: ccl_num_links = 2 on Blackhole)."""
    _run_fused(heads, block_cyclic=block_cyclic, num_links=2)


@pytest.mark.parametrize("block_cyclic", [False, True], ids=["contiguous", "block_cyclic"])
@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_ring4_fused_bfp8_k(case_id, heads, block_cyclic):
    """Production dtype: the K cache (both the local shard and the gathered buffer) is bfloat8_b while q/w stay
    bf16 (post-RoPE), matching how the model allocates the KV cache. PCC>=0.999, same floor as the bf16 cases."""
    _run_fused(heads, block_cyclic=block_cyclic, num_links=1, k_dtype=ttnn.bfloat8_b)


@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_ring4_fused_production_shape(case_id, heads):
    """ALL the production knobs at once, which the other fused tests only exercise in isolation: block-cyclic
    layout + a non-zero chunk_start (a later prefill chunk) + kv_len < T_alloc (over-allocated cache) +
    num_links=2 (the Blackhole production link count) + bfloat8_b K (the model's cache dtype). Each is covered
    alone elsewhere (block_cyclic/nl2 in _run_fused, kv_len in runtime_kv_len, bfp8 in bfp8_k, chunk_start in
    program_cache_reuse), but the model always drives them TOGETHER -- this guards their interaction (e.g. the
    bfp8 gathered-buffer + the kv_len tail mask + the block-cyclic invP on the nl2 schedule)."""
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
    """Multi-user indexed KV cache on the fused path: k_local is [num_users,1,sll,D] and the gathered buffer is
    [num_users,1,T,D]; cache_batch_idx selects the slot to score. The reader must add the slot's page offset to
    BOTH the remote (gathered) reads AND the LOCAL shard reads -- and the local stride is 1/ring of the gathered
    one (k_local holds sll=T/ring keys per slot). Slots hold DIFFERENT random K, so reading the local band from
    slot 0 (the pre-fix bug: local offset omitted) silently mixes users and fails the per-slot PCC for any
    cache_batch_idx>0. Contiguous layout -- the slot offset is added on top of, and is independent of, the
    block-cyclic within-shard remap. Op-level coverage of the in-kernel cache_batch_idx>0 select: this is a
    supported op capability, NOT the current model path -- indexer.py slices the (user,layer) slot to batch-1
    BEFORE the ring AG and calls the op with cache_batch_idx=None, so the model never sets cache_batch_idx>0.
    This test guards the op feature (and the local-shard offset fix) for any caller that does use it."""
    submesh, parent, ccl_semaphores, subdevice_id, stall_group = _open_ring4_ccl()
    try:
        # Shared q/w (one query set) scoring distinct per-user caches -- the multi-user decode shape. Distinct K
        # per slot (distinct seed) so a wrong-slot local read changes the score.
        q_g, _, w_g = _global_inputs(heads, CHUNK_GLOBAL, T, seed=42)
        k_multi = torch.cat([_global_inputs(heads, CHUNK_GLOBAL, T, seed=100 + u)[1] for u in range(num_users)], dim=0)

        # Multi-user cache: k_local/k_gathered are [num_users,1,*,D]; cache_batch_idx selects the slot in-kernel.
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

        # Reference: score the SELECTED slot only (each SP rank vs the full [1,1,T,D] slot).
        ref = _per_sp_ref(q_g, k_multi[cache_batch_idx : cache_batch_idx + 1], w_g, RING, QB_HISTORY)
        assert_indexer_match(out_t, ref, CHUNK_GLOBAL, T, check_neg=True)
        logger.info(
            f"ring4 fused multi-user (heads={heads}, users={num_users}, slot={cache_batch_idx}): matched reference"
        )
    finally:
        _close_ring4_ccl(parent, submesh, stall_group)


def test_indexer_score_ring4_fused_indexed_cache():
    """Multi-user indexed KV cache: cache_batch_idx=1 selects the 2nd user slot of a [B,1,T,D] gathered cache
    backed by a [B,1,sll,D] local shard. slot1 (offset>0) is the regression guard for the local-shard batch
    offset -- the reader must offset its OWN shard's read by the slot too, not just the gathered remote reads
    (pre-fix it read the local band from slot 0, silently mixing users). This exercises the op's in-kernel
    cache_batch_idx>0 select -- a supported op CAPABILITY the model itself does NOT use (indexer.py slices the
    slot to batch-1 before the AG and passes cache_batch_idx=None), so one representative case (dsv32) guards
    the feature + the offset fix for other/future callers."""
    _run_fused_multiuser(16, num_users=2, cache_batch_idx=1)  # dsv32; slot offset is head-count-independent


@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_ring4_fused_straddle(case_id, heads):
    """Mid-slab-boundary straddle + block-cyclic chip rotation on the fused ring-of-4 (the rotated-prefill /
    multiturn shape): a NON-slab-aligned chunk_start (704) makes the boundary chip's queries cross a slab
    boundary, so the causal diagonal must JUMP by (chunk_global - cl) -- driven by the SAME device_causal_geometry
    (straddle_q_tile / straddle_jump_tiles) the classic path uses (test_indexer_score_qb_straddle). This proves
    the fusion's band reorder + per-band gate + dual-source local read compose correctly with the straddled
    causal mask, not just the slab-aligned deduced chunk_start the other fused tests use. sp=4 here: cl=320,
    boundary_chip=2, offset=64. Checked vs the per-SP-rank rotated reference over both K layouts."""
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
    """Padded / oversized KV cache on the fused ring: k is ALLOCATED at T_alloc but only a kv_len prefix is
    valid this dispatch, and only output columns [0, kv_len) are written (the stale tail is sliced off). Uses an
    explicit chunk_start that keeps the fullest rank's causal window inside kv_len (the deduced base would attend
    to T_alloc and validate would reject kv_len < that). Contiguous layout; heads=16 (dsv32) is representative --
    kv_len masking is head-count-independent. Confirms the AG gathers the full T_alloc and the compute masks
    beyond kv_len on the fused schedule (band_count is built over full T, so no shard is left un-delivered)."""
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
    """Two dispatches with IDENTICAL tensor shapes but a DIFFERENT chunk_start / kv_len, on the SAME device
    (so the 2nd hits the cached program). chunk_start_idx / kv_len are HASH-EXCLUDED (one program is reused
    across chunked-prefill chunks / decode steps), so the descriptor factory MUST re-apply them in
    override_runtime_arguments; if it doesn't, the 2nd dispatch scores with the 1st's frozen causal offset /
    valid length -> wrong logits (regression guard for the fused-op program-cache stale-scalar bug: every
    other fused test dispatches cold, so only a same-device 2nd call with a new chunk_start exercises the hit).
    Block-cyclic, two consecutive prefill chunks: chunk@0 then chunk@CHUNK_GLOBAL over one shared cache. Run in
    BOTH bf16 and the production bfp8_b K, so the scalar re-patch is guarded on the deployed cache dtype too."""
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

        # chunk@0 (cache miss / build) then chunk@CHUNK_GLOBAL (cache HIT -- must re-apply chunk_start + kv_len).
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
    """The fused path requires all heads resident (head_group_size 0 or Hi); a streaming config
    (0 < head_group_size < Hi) must be rejected at validate, not silently mis-scheduled. The rejection is
    head-count-independent, so a single representative case (dsv32, Hi=16) is checked -- one ring open/close is
    also deliberate: closing the mesh after a validate-time failure leaves fabric state that a second
    consecutive ring open would trip on this box."""
    heads = 16  # dsv32 (the production DSA head count); head_group_size=8 is a streaming config (0 < 8 < 16)
    submesh, parent, ccl_semaphores, subdevice_id, stall_group = _open_ring4_ccl()
    try:
        q_g, k_nat, w_g = _global_inputs(heads, CHUNK_GLOBAL, T, seed=42)
        q_dev, w_dev, k_local, k_gathered = _fused_dev_inputs(submesh, q_g, w_g, k_nat)
        base = glx_config(heads)
        # head_group_size strictly between 0 and Hi (heads//2 divides Hi for both glm5=8 and dsv32=16) => streaming.
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
