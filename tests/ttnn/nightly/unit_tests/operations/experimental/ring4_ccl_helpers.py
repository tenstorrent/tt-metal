# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared LoudBox ring-of-4 CCL plumbing for the ring-fused indexer_score correctness tests.

Not a test module (no `test_` prefix, nothing collected): it holds the fabric/sub-device/semaphore setup and
the SP-shard / persistent-buffer tensor mappers that `test_ring_indexer_score_dsa.py` reuses, so the ring-4
recipe lives in exactly one place. The 4-chip variant (`test_ring_indexer_score_dsa_4d.py`) opens its mesh
directly and keeps its own copy of `_open_ccl` (no (2,4)->(1,4) submesh carve).
"""

import ttnn

from tests.ttnn.nightly.unit_tests.operations.experimental.test_indexer_score import (
    QB_HISTORY,
    QB_SQ,
)

# Ring of 4 on the LoudBox: full physical mesh, then a 1x4 submesh (SP axis = 1).
LOUDBOX_MESH_SHAPE = (2, 4)
RING = 4
SP_AXIS = 1  # the length-4 axis of the (1, 4) submesh
CHUNK_GLOBAL = RING * QB_SQ  # 2560 global prefill chunk = sp * per-shard slab (chunk_local = QB_SQ)
T = QB_HISTORY + CHUNK_GLOBAL  # 28160 all-gathered keys (880 tiles); 11 global chunks of 2560

DRAM = ttnn.DRAM_MEMORY_CONFIG

# ShardTensor2dMesh dim tuples for the (1, RING) submesh (mesh axis 0 = length 1, axis 1 = SP length RING).
# Input K: replicate on axis 0, shard seq (tensor dim 2) along the SP axis -> each device gets its T/RING slab.
_INPUT_DIMS = (None, 2)
# Persistent AG buffer: full T on every device -> replicate along the SP axis; axis 0 shards the size-1 dim 1.
_BUF_DIMS = (1, None)


def _open_ring4_ccl():
    """Open the full 2x4 with FABRIC_1D, carve a 1x4 submesh, load a worker sub-device, make 2 ccl semaphores
    (the two ring directions, as ring_attention_all_gather_async needs). Returns
    (submesh, parent, ccl_semaphores, worker_sub_device_id, stall_group)."""
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
