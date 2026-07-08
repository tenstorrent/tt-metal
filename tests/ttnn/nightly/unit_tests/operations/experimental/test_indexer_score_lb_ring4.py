# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Reference (UNFUSED) ring-of-4 indexer_score on a Blackhole LoudBox (2x4, 8 chips).

Baseline we iterate on for the fused "ring indexer_score" work: it runs the two phases
separately and sequentially, exactly as the model does today
(models/demos/deepseek_v3_d_p/tt/mla/indexer.py _gather_index_kbuf -> _sp_all_gather ->
indexer_score_dsa):

    1. K is SP-sharded across the 4 ring devices (each holds T/4 keys).
    2. A device all_gather over the SP ring reconstructs the full [1,1,T,D] key cache on every
       device (the blocking barrier the PERF TODO wants to fuse away).
    3. indexer_score_dsa scores each device's 640 local queries against the full key cache,
       deriving its own chunk_start from the mesh coordinate.

Two K layouts, parametrized:
    - contiguous:   K is sharded in natural token order (the simplest reference; good for a
                    fabric-free schedule spike).
    - block_cyclic: the PRODUCTION layout. Chunked prefill packs each SP shard's per-chunk slab
                    (chunk_local = global_chunk/sp keys) back-to-back, so the SP-gathered cache is
                    in a permuted physical order and indexer_score's reader reads it back in natural
                    token order (invP per tile). This is the layout the fused op must reproduce.
                    Built with _to_slab() and scored with block_cyclic_sp_axis / _chunk_local.

Both check per-SP-rank against the SAME single-device DSA reference (_per_sp_ref, natural order):
the golden math is layout-independent, so only the K permutation into the op changes.

Mesh: the LoudBox is a plain 2x4 grid (no torus), so the ring of 4 runs as FABRIC_1D /
Topology.Linear over a 1x4 submesh of the full mesh; the gather is a line all-gather along the
SP (length-4) axis.

Run:  scripts/run_safe_pytest.sh tests/ttnn/nightly/unit_tests/operations/experimental/test_indexer_score_lb_ring4.py
"""

import pytest
from loguru import logger

import ttnn

from tests.ttnn.nightly.unit_tests.operations.experimental.test_indexer_score import (
    assert_indexer_match,
    glx_config,
    _global_inputs,
    _per_sp_ref,
    _to_slab,
    QB_HISTORY,
    QB_SQ,
    QB_CASES,
    QB_IDS,
)

pytestmark = pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="indexer_score is Blackhole-only")

# Ring of 4 on the LoudBox: full physical mesh, then a 1x4 submesh (SP axis = 1).
LOUDBOX_MESH_SHAPE = (2, 4)
RING = 4
SP_AXIS = 1  # the length-4 axis of the (1, 4) submesh

# GLX chunked-prefill geometry, scaled to a ring of 4 (mirrors the QuietBox SP=4 tests).
CHUNK_GLOBAL = RING * QB_SQ  # 2560: the global prefill chunk = sp * per-shard slab (chunk_local = QB_SQ)
T = QB_HISTORY + CHUNK_GLOBAL  # 28160 all-gathered keys (880 tiles); 11 global chunks of 2560


def _open_ring4():
    """Open the full 2x4 mesh with FABRIC_1D, carve a 1x4 submesh, and set up CCL semaphores.

    Returns (submesh, parent_mesh, ccl_semaphores, barrier_semaphore). Line transport (not RING):
    a 2x4 grid has no wrap-around, so FABRIC_1D_RING would fail fabric init on a 1x4 row.
    """
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    parent = None
    try:
        parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*LOUDBOX_MESH_SHAPE))
        submesh = parent.create_submesh(ttnn.MeshShape(1, RING))

        grid = submesh.compute_with_storage_grid_size()
        ccl_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        # all_gather_async takes a pair of global semaphores + a barrier semaphore.
        ccl_semaphores = [ttnn.create_global_semaphore(submesh, ccl_crs, 0) for _ in range(2)]
        barrier_semaphore = ttnn.create_global_semaphore(submesh, ccl_crs, 0)
        return submesh, parent, ccl_semaphores, barrier_semaphore
    except Exception:
        if parent is not None:
            ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        raise


def _close_ring4(parent):
    try:
        ttnn.close_mesh_device(parent)
    finally:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _sp_all_gather_k(k_dev, ccl_semaphores, barrier_semaphore):
    """Line all-gather of the SP-sharded K cache along the ring -> full [1,1,T,D] on every device.

    Layout-agnostic: it just concatenates each device's slab along seq (dim 2) in device order, so a
    contiguous shard reassembles to natural order and a block-cyclic shard reassembles to the permuted
    physical order the block-cyclic reader expects.
    """
    return ttnn.experimental.all_gather_async(
        k_dev,
        dim=2,
        multi_device_global_semaphore=ccl_semaphores,
        num_links=2,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear,
        barrier_semaphore=barrier_semaphore,
        cluster_axis=SP_AXIS,
    )


def _run_ring4_reference(heads, *, block_cyclic):
    """SP-shard K -> device all_gather over the ring -> indexer_score_dsa, checked per SP rank."""
    submesh, parent, ccl_semaphores, barrier_semaphore = _open_ring4()
    try:
        q_g, k_nat, w_g = _global_inputs(heads, CHUNK_GLOBAL, T, seed=42)

        # q/w sharded along seq (dim 2) across the ring (each device its own 640 queries).
        shard = ttnn.ShardTensorToMesh(submesh, dim=2)
        q_dev = ttnn.from_torch(q_g, device=submesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)
        w_dev = ttnn.from_torch(w_g, device=submesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)

        # K host layout: natural order (contiguous) or the block-cyclic physical permutation. In both cases
        # each device is then handed its contiguous slab of that layout (= its per-chip cache in production).
        k_host = _to_slab(k_nat, RING, CHUNK_GLOBAL) if block_cyclic else k_nat
        k_dev = ttnn.from_torch(k_host, device=submesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)

        # Phase 1: blocking all-gather -> full key cache reconstructed on every ring device.
        k_full = _sp_all_gather_k(k_dev, ccl_semaphores, barrier_semaphore)

        # Phase 2: score. chunk_start_idx omitted -> the op deduces base = T - RING*Sq = QB_HISTORY from the
        # mesh, and device r (along SP_AXIS) scores its 640 queries with chunk_start = base + r*Sq. For the
        # block-cyclic cache the reader also remaps each logical K-tile to its physical slab row (invP), so
        # scores come out in natural token order regardless of the physical layout.
        bc_kwargs = dict(block_cyclic_sp_axis=SP_AXIS, block_cyclic_chunk_local=QB_SQ) if block_cyclic else {}
        out = ttnn.experimental.indexer_score_dsa(
            q_dev,
            k_full,
            w_dev,
            cluster_axis=SP_AXIS,
            program_config=glx_config(heads),
            **bc_kwargs,
        )
        out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=2))

        # Reference is always in natural token order; the block-cyclic reader must undo the permutation.
        ref = _per_sp_ref(q_g, k_nat, w_g, RING, QB_HISTORY)
        assert_indexer_match(out_t, ref, CHUNK_GLOBAL, T, check_neg=True)
        layout = "block_cyclic" if block_cyclic else "contiguous"
        logger.info(f"ring4 indexer_score {layout} (heads={heads}): all-gather + score matched reference")
    finally:
        _close_ring4(parent)


@pytest.mark.parametrize("block_cyclic", [False, True], ids=["contiguous", "block_cyclic"])
@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_ring4_allgather_then_score(case_id, heads, block_cyclic):
    """UNFUSED reference: SP-shard K -> device all_gather -> indexer_score_dsa, on a ring of 4.

    block_cyclic=True is the production K layout (per-SP-shard slabs); block_cyclic=False is the
    simpler natural-order layout. Both must match the natural-order per-SP DSA reference.
    """
    _run_ring4_reference(heads, block_cyclic=block_cyclic)
