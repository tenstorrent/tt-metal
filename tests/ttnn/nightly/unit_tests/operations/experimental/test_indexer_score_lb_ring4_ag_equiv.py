# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
STEP A of the ring-fused indexer_score roadmap (see ring_indexer_score_fusion_design.md).

The go/no-go spike: does `ring_attention_all_gather_async` -- the ONLY Linear+fuse-capable all-gather, the
producer the fused op must drive -- reconstruct the SP-sharded K cache into the SAME full [1,1,T,D] buffer
that the current `all_gather_async` path (test_indexer_score_lb_ring4.py) produces, on the LoudBox 1x4 Linear
submesh, for BOTH the contiguous and the production block-cyclic layout?

If yes, the entire fusion is unblocked (every later step keeps the scoring path byte-identical and only gates
the reader on slab arrival). If no, the port is blocked and we stop here. NO kernel code is touched in this
spike -- it is pure ttnn plumbing.

Two checks, parametrized over both K layouts:

  A1 (placement proof): SP-shard K, ring_attention all-gather into a zero-filled persistent buffer, and assert
      the reconstructed per-device buffer is BYTE-EXACT to the intended layout (k_host) with the device's own
      local band zeroed. The ring_attention AG deliberately does NOT write a device's local slice
      (test_ring_attention_all_gather.py:206), so we zero that band on both sides before comparing -- exactly
      as the upstream t3000 test does. bf16 => exact equality, not PCC.

  A2 (end-to-end): pre-fill the persistent buffer with the full k_host (so each device's local band is
      correct), let the AG overwrite the remote bands, then feed the UNMODIFIED indexer_score_dsa and assert
      against the natural-order per-SP reference. A1 carries the remote-placement proof; A2 confirms the
      ring_attention output tensor drives the existing op identically end-to-end (a wrong overwrite corrupts a
      band away from k_host and fails; a no-op AG is caught by A1's zeroed-local check).

Mesh: FABRIC_1D / Topology.Linear over a 1x4 submesh of the full 2x4 (no torus), matching the oracle.

Run: scripts/run_safe_pytest.sh tests/ttnn/nightly/unit_tests/operations/experimental/test_indexer_score_lb_ring4_ag_equiv.py
"""

import pytest
import torch
from loguru import logger

import ttnn

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal
from tests.ttnn.nightly.unit_tests.operations.experimental.test_indexer_score import (
    _global_inputs,
    _to_slab,
    QB_HISTORY,
    QB_SQ,
)

pytestmark = [
    pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="indexer_score is Blackhole-only"),
    pytest.mark.skipif(ttnn.get_num_devices() < 8, reason="ring-of-4 needs the 8-chip LoudBox (2x4)"),
]

# Ring of 4 on the LoudBox: full physical mesh, then a 1x4 submesh (SP axis = 1). Same geometry as the oracle.
LOUDBOX_MESH_SHAPE = (2, 4)
RING = 4
SP_AXIS = 1  # the length-4 axis of the (1, 4) submesh
CHUNK_GLOBAL = RING * QB_SQ  # 2560 global prefill chunk = sp * per-shard slab (chunk_local = QB_SQ)
T = QB_HISTORY + CHUNK_GLOBAL  # 28160 all-gathered keys (880 tiles); 11 global chunks of 2560
SLL = T // RING  # per-device slab length (keys): 7040

DRAM = ttnn.DRAM_MEMORY_CONFIG


def _open_ring4_ccl():
    """Open the full 2x4 with FABRIC_1D, carve a 1x4 submesh, load a worker sub-device, make 2 ccl semaphores.

    ring_attention_all_gather_async needs a sub-device stall group + a pair of global semaphores (NOT the
    barrier-semaphore overload used by all_gather_async). Mirrors tests/nightly/t3000/ccl/
    test_ring_attention_all_gather.py exactly, adapted to the manual full-mesh->submesh LoudBox recipe.

    Returns (submesh, parent, ccl_semaphores, worker_sub_device_id, stall_group).
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
        worker_sub_device = ttnn.SubDevice([ccl_crs])
        worker_sub_device_id = ttnn.SubDeviceId(0)
        stall_group = [worker_sub_device_id]
        mgr = submesh.create_sub_device_manager([worker_sub_device], 0)
        submesh.load_sub_device_manager(mgr)
        submesh.set_sub_device_stall_group(stall_group)

        ccl_semaphores = [ttnn.create_global_semaphore(submesh, ccl_crs, 0) for _ in range(2)]
        return submesh, parent, ccl_semaphores, worker_sub_device_id, stall_group
    except Exception:
        if parent is not None:
            ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        raise


def _close_ring4_ccl(parent, submesh, stall_group):
    try:
        try:
            submesh.reset_sub_device_stall_group()
            submesh.clear_loaded_sub_device_manager()
        finally:
            ttnn.close_mesh_device(parent)
    finally:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _make_ccl_semaphores(submesh):
    """Two fresh global semaphores (the two ring directions) over the full compute grid, for the fused op / AG."""
    grid = submesh.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    return [ttnn.create_global_semaphore(submesh, crs, 0) for _ in range(2)]


def _reset(sems):
    """Zero the direction semaphores between iterations (the fused op / AG signal into them each run)."""
    for s in sems:
        ttnn.reset_global_semaphore_value(s, 0)


# ShardTensor2dMesh dim tuples for the (1, RING) submesh (mesh axis 0 = length 1, axis 1 = SP length RING).
# Input K: replicate on axis 0, shard seq (tensor dim 2) along the SP axis -> each device gets its T/RING slab.
_INPUT_DIMS = (None, 2)
# Persistent AG buffer: full T on every device -> replicate along the SP axis; axis 0 shards the size-1 dim 1.
_BUF_DIMS = (1, None)
# Compose the reconstructed buffer back: concat the SP replicas along seq (dim 2), heads (dim 1) along axis 0.
_OUT_DIMS = (1, 2)


def _shard_k(submesh, k_host, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        k_host,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=DRAM,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=(1, RING), dims=_INPUT_DIMS),
    )


def _persistent_buffer(submesh, fill, dtype=ttnn.bfloat16):
    """Full [1,1,T,D] persistent AG output buffer, replicated across the SP axis. `fill` is the host tensor."""
    return ttnn.from_torch(
        fill,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=DRAM,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=(1, RING), dims=_BUF_DIMS),
    )


def _build_ring4_fused_inputs(submesh, heads, block_cyclic):
    """Shared device-side inputs for the fused-op perf/profile harnesses: q/w sharded on the SP axis, the local
    K shard, a zero-filled gathered buffer, the block-cyclic kwargs, and 2 fresh semaphores. (Correctness tests
    build their own inputs with a reference; these harnesses only time/profile, so no reference is returned.)"""
    q_g, k_nat, w_g = _global_inputs(heads, CHUNK_GLOBAL, T, seed=42)
    k_host = _to_slab(k_nat, RING, CHUNK_GLOBAL) if block_cyclic else k_nat
    shard = ttnn.ShardTensorToMesh(submesh, dim=2)
    q_dev = ttnn.from_torch(q_g, device=submesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)
    w_dev = ttnn.from_torch(w_g, device=submesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)
    k_local = _shard_k(submesh, k_host)
    k_gathered = _persistent_buffer(submesh, torch.zeros_like(k_host))
    bc = dict(block_cyclic_sp_axis=SP_AXIS, block_cyclic_chunk_local=QB_SQ) if block_cyclic else {}
    sems = _make_ccl_semaphores(submesh)
    return q_dev, w_dev, k_local, k_gathered, bc, sems


def _ring_attention_ag(k_dev, persistent_buf, ccl_semaphores, subdevice_id):
    """ring_attention all-gather of the SP-sharded K along the ring -> full [1,1,T,D] on every device."""
    out = ttnn.experimental.ring_attention_all_gather_async(
        [k_dev],
        persistent_output_buffer=[persistent_buf],
        dim=2,
        multi_device_global_semaphore=ccl_semaphores,
        cluster_axis=SP_AXIS,
        mesh_device=k_dev.device(),
        num_links=1,
        memory_config=DRAM,
        topology=ttnn.Topology.Linear,
        subdevice_id=subdevice_id,
    )
    return out[0]


def _per_device_buffers(gathered, submesh):
    """Compose the mesh tensor and split into the RING per-device full-T reconstructed buffers."""
    full = ttnn.to_torch(
        gathered,
        mesh_composer=ttnn.ConcatMesh2dToTensor(submesh, mesh_shape=(1, RING), dims=_OUT_DIMS),
    )
    return torch.chunk(full, RING, dim=2)  # each [1,1,T,D]


@pytest.mark.parametrize("block_cyclic", [False, True], ids=["contiguous", "block_cyclic"])
def test_stepA1_ring_attention_ag_layout_equivalence(block_cyclic):
    """A1: ring_attention AG must reconstruct k_host byte-exactly (local band zeroed) on every ring device."""
    submesh, parent, ccl_semaphores, subdevice_id, stall_group = _open_ring4_ccl()
    try:
        _, k_nat, _ = _global_inputs(8, CHUNK_GLOBAL, T, seed=42)
        k_host = _to_slab(k_nat, RING, CHUNK_GLOBAL) if block_cyclic else k_nat

        k_dev = _shard_k(submesh, k_host)
        buf = _persistent_buffer(submesh, torch.zeros_like(k_host))
        gathered = _ring_attention_ag(k_dev, buf, ccl_semaphores, subdevice_id)
        ttnn.synchronize_device(submesh, sub_device_ids=stall_group)

        per_dev = _per_device_buffers(gathered, submesh)
        for ring_idx, dev_buf in enumerate(per_dev):
            # AG omits the device's own local slice; zero it on both sides before the byte-exact compare.
            got = dev_buf.clone()
            got[:, :, ring_idx * SLL : (ring_idx + 1) * SLL, :].zero_()
            want = k_host.clone()
            want[:, :, ring_idx * SLL : (ring_idx + 1) * SLL, :].zero_()
            eq, msg = comp_equal(got, want)
            assert eq, f"ring_idx={ring_idx} block_cyclic={block_cyclic}: AG placement mismatch: {msg}"
        layout = "block_cyclic" if block_cyclic else "contiguous"
        logger.info(f"stepA1 {layout}: ring_attention AG reconstructs k_host byte-exactly on all {RING} devices")
    finally:
        _close_ring4_ccl(parent, submesh, stall_group)
