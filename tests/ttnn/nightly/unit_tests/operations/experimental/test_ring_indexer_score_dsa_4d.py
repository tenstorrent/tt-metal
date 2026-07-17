# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Ring-fused indexer_score correctness on a 4-device Blackhole box (QuietBox-class), covering the two mesh
layouts that fit in 4 chips:

  * SP-only ring-of-4 on a (1, 4) mesh (cluster_axis = the length-4 axis) -- the same fused op the 8-chip
    LoudBox suite exercises, run here directly on 4 devices (no (2,4) parent to carve).
  * 2D SP×TP on a (2, 2) mesh: sp=2 ring (cluster_axis) × tp=2 sequence sub-shard (seq_subshard_axis). The K
    cache stays SP-sharded + TP-replicated so the ring all-gather is unchanged; TP only sub-shards the QUERY
    rows, and seq_subshard_axis carries each device's tp_rank*Sq block-cyclic sub-offset into the causal score
    (device_causal_geometry). This is the fused analogue of the classic-path
    test_indexer_score.py::test_indexer_score_sp2_tp2_seq_subshard_rotated.

Unlike the LoudBox suite (test_ring_indexer_score_dsa.py, which opens the full (2,4) and carves a (1,4)
submesh, needing 8 chips), these open the target mesh DIRECTLY so they run on any 4-device Blackhole system.
The gathered buffer is seeded with ZEROS, so a correct score PROVES device-side local sourcing (a stale local
band would fail the -inf map + PCC), exactly as the LoudBox tests.

Run:  scripts/run_safe_pytest.sh tests/ttnn/nightly/unit_tests/operations/experimental/test_ring_indexer_score_dsa_4d.py
"""

import pytest
import torch
from loguru import logger

import ttnn

from tests.ttnn.nightly.unit_tests.operations.experimental.test_indexer_score import (
    assert_indexer_match,
    glx_config,
    indexer_score_dsa_ref,
    _global_inputs,
    _per_sp_ref,
    _to_slab,
    QB_SQ,
    QB_HISTORY,
    QB_DIM,
    QB_CASES,
    QB_IDS,
)

DRAM = ttnn.DRAM_MEMORY_CONFIG

pytestmark = [
    pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="indexer_score is Blackhole-only"),
    pytest.mark.skipif(ttnn.get_num_devices() < 4, reason="needs a 4-device Blackhole box"),
]


def _open_ccl(mesh_shape):
    """Open a mesh of `mesh_shape` DIRECTLY (no parent carve), load a worker sub-device, make 2 ccl semaphores.

    Mirrors test_indexer_score_lb_ring4_ag_equiv._open_ring4_ccl but skips the (2,4)->(1,4) submesh step, so it
    runs on a 4-chip box. ring_attention_all_gather_async needs a sub-device stall group + a pair of global
    semaphores (the two ring directions). Returns (mesh, ccl_semaphores, worker_sub_device_id, stall_group)."""
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D,
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
        mgr = mesh.create_sub_device_manager([worker_sub_device], 0)
        mesh.load_sub_device_manager(mgr)
        mesh.set_sub_device_stall_group(stall_group)
        ccl_semaphores = [ttnn.create_global_semaphore(mesh, ccl_crs, 0) for _ in range(2)]
        return mesh, ccl_semaphores, worker_sub_device_id, stall_group
    except Exception:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        raise


def _close_ccl(mesh):
    try:
        try:
            mesh.reset_sub_device_stall_group()
            mesh.clear_loaded_sub_device_manager()
        finally:
            ttnn.close_mesh_device(mesh)
    finally:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# ---- SP-only ring-of-4 on a (1, 4) mesh ------------------------------------------------------------
RING4 = 4
SP4_AXIS = 1  # the length-4 axis of the (1, 4) mesh == cluster_axis
CHUNK4 = RING4 * QB_SQ  # 2560 global prefill chunk (chunk_local = QB_SQ per SP shard)
T4 = QB_HISTORY + CHUNK4  # 28160 all-gathered keys


@pytest.mark.parametrize("block_cyclic", [False, True], ids=["contiguous", "block_cyclic"])
@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_ring4_fused_4d(case_id, heads, block_cyclic):
    """SP-only ring-of-4 fused all-gather + indexer_score on a directly-opened (1,4) mesh, checked vs the
    per-SP DSA reference. Same op/knobs as the LoudBox suite; validates the base fused path on a 4-device box."""
    mesh, ccl_semaphores, subdevice_id, stall_group = _open_ccl((1, RING4))
    try:
        q_g, k_nat, w_g = _global_inputs(heads, CHUNK4, T4, seed=42)
        k_host = _to_slab(k_nat, RING4, CHUNK4) if block_cyclic else k_nat

        shard = ttnn.ShardTensorToMesh(mesh, dim=2)  # SP-shard seq over the 4 devices
        q_dev = ttnn.from_torch(q_g, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)
        w_dev = ttnn.from_torch(w_g, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)
        k_local = ttnn.from_torch(k_host, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)
        # Gathered buffer: full T on every device, seeded with ZEROS (AG fills the remote bands; the reader
        # dual-sources the local band from k_local -> zeros prove device-side local sourcing).
        k_gathered = ttnn.from_torch(
            torch.zeros_like(k_nat),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        bc_kwargs = dict(block_cyclic_sp_axis=SP4_AXIS, block_cyclic_chunk_local=QB_SQ) if block_cyclic else {}
        out = ttnn.experimental.ring_indexer_score_dsa(
            q_dev,
            k_gathered,
            w_dev,
            k_local,
            ccl_semaphores,
            cluster_axis=SP4_AXIS,
            topology=ttnn.Topology.Linear,
            num_links=1,
            ag_sub_device_id=subdevice_id,
            program_config=glx_config(heads),
            **bc_kwargs,
        )
        ttnn.synchronize_device(mesh, sub_device_ids=stall_group)
        out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2))

        ref = _per_sp_ref(q_g, k_nat, w_g, RING4, QB_HISTORY)
        assert_indexer_match(out_t, ref, CHUNK4, T4, check_neg=True)
        layout = "block_cyclic" if block_cyclic else "contiguous"
        logger.info(f"4d ring4 fused {layout} (heads={heads}): matched reference")
    finally:
        _close_ccl(mesh)


# ---- 2D SP×TP on a (2, 2) mesh: sp=2 ring × tp=2 sequence sub-shard ---------------------------------
SP2 = 2  # sequence-parallel ranks == ring size (cluster_axis extent)
TP2 = 2  # tensor-parallel ranks the QUERY sequence is ALSO sub-sharded over (seq_subshard_axis extent)
SP2_AXIS = 0  # mesh rows == SP ring (cluster_axis / block_cyclic_sp_axis)
TP2_AXIS = 1  # mesh cols == TP seq sub-shard (seq_subshard_axis)
CHUNK_SPTP = SP2 * QB_SQ  # 1280 global chunk; per-SP-shard chunk_local = QB_SQ (640), per-device Sq = 320
T_SPTP = QB_HISTORY + CHUNK_SPTP  # 26880 keys


def _per_sp_tp_ref(q_g, k_g, w_g, sp, tp, history, sq_sp):
    """Reference for a 2D SP×TP seq sub-shard, in global chunk-row order (== the row-major device order of a
    (sp, tp) mesh with SP the outer axis). Device (r, t) owns global query rows
    [r*sq_sp + t*sq_dev, r*sq_sp + (t+1)*sq_dev) and its causal chunk_start is history + that row base (the
    exact block-cyclic position device_causal_geometry reproduces for a slab-aligned start)."""
    sq_dev = sq_sp // tp
    refs = []
    for r in range(sp):
        for t in range(tp):
            g0 = r * sq_sp + t * sq_dev
            sl = slice(g0, g0 + sq_dev)
            refs.append(indexer_score_dsa_ref(q_g[:, :, sl, :], k_g, w_g[:, :, sl, :], history + g0))
    return torch.cat(refs, dim=2)


@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_sptp_fused_4d(case_id, heads):
    """2D SP×TP fused indexer_score on a (2,2) mesh: sp=2 ring (cluster_axis) + tp=2 query seq sub-shard
    (seq_subshard_axis). Block-cyclic K (required whenever seq_subshard_axis is set): the cache is SP-sharded +
    TP-replicated, so k_local / the ring AG (along cluster_axis) are unchanged and TP only sub-shards the query
    rows. Each device's causal diagonal must start at history + sp_rank*Sq_sp + tp_rank*Sq_dev -- proving the
    fused path threads the TP sub-offset into the score (the feature this test guards). Slab-aligned explicit
    chunk_start (no straddle); checked vs the per-(sp,tp) reference over both head counts."""
    chunk_local = CHUNK_SPTP // SP2  # per-SP-shard chunk == QB_SQ (640); per-device query rows = 320
    chunk_start = QB_HISTORY  # slab-aligned (QB_HISTORY % CHUNK_SPTP == 0) -> no straddle
    mesh, ccl_semaphores, subdevice_id, stall_group = _open_ccl((SP2, TP2))
    try:
        q_g, k_nat, w_g = _global_inputs(heads, CHUNK_SPTP, T_SPTP, seed=42)
        k_bc = _to_slab(k_nat, SP2, CHUNK_SPTP)  # block-cyclic physical layout the reader inverts

        # K: block-cyclic slab sharded on the SP axis (dim 2), REPLICATED across TP (dims indexed by axis).
        k_shard = ttnn.ShardTensor2dMesh(mesh, mesh_shape=(SP2, TP2), dims=(2, None))
        k_local = ttnn.from_torch(k_bc, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=k_shard)
        # Gathered buffer: full T on every device (replicated over BOTH axes), seeded with ZEROS.
        k_gathered = ttnn.from_torch(
            torch.zeros_like(k_nat),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        # q/w: SP-shard seq (dim 2) on the SP axis, then split those rows over TP (mesh_partition) -> each
        # device owns Sq_dev = Sq_sp/tp rows. Mirrors the classic sp2_tp2_seq_subshard path (rope-then-split).
        qw_shard = ttnn.ShardTensor2dMesh(mesh, mesh_shape=(SP2, TP2), dims=(2, None))
        q_dev = ttnn.from_torch(q_g, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=qw_shard)
        w_dev = ttnn.from_torch(w_g, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=qw_shard)
        q_dev = ttnn.mesh_partition(q_dev, dim=2, cluster_axis=TP2_AXIS)
        w_dev = ttnn.mesh_partition(w_dev, dim=2, cluster_axis=TP2_AXIS)

        out = ttnn.experimental.ring_indexer_score_dsa(
            q_dev,
            k_gathered,
            w_dev,
            k_local,
            ccl_semaphores,
            cluster_axis=SP2_AXIS,
            topology=ttnn.Topology.Linear,
            num_links=1,
            ag_sub_device_id=subdevice_id,
            chunk_start_idx=chunk_start,
            seq_subshard_axis=TP2_AXIS,  # the SP×TP feature under test
            block_cyclic_sp_axis=SP2_AXIS,
            block_cyclic_chunk_local=chunk_local,
            program_config=glx_config(heads),
        )
        ttnn.synchronize_device(mesh, sub_device_ids=stall_group)
        # Compose device shards back to global chunk order (row-major: SP outer, TP inner).
        shards = [ttnn.to_torch(s) for s in ttnn.get_device_tensors(out.cpu())]
        out_t = torch.cat(shards, dim=2)

        ref = _per_sp_tp_ref(q_g, k_nat, w_g, SP2, TP2, chunk_start, chunk_local)
        assert_indexer_match(out_t, ref, CHUNK_SPTP, T_SPTP, check_neg=True)
        logger.info(f"4d SP×TP fused (heads={heads}): sp2 ring × tp2 seq sub-shard matched reference")
    finally:
        _close_ccl(mesh)
