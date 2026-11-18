# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Fabric init and a ring all-gather on both axes of every [RING,RING] 4x4 prefill mesh.

Run on its own (one rank, one in-Galaxy torus) it checks a single mesh. Each rank opens the 4x4 mesh it belongs to with FABRIC_2D_TORUS_XY: an in-Galaxy torus is owned by one
rank, a seam torus by two ranks on two hosts. Chip (r, c) of a mesh holds a block filled with r * 4 + c + 1;
a ring all-gather along each axis must return the blocks of the whole row or column in mesh order, on
every chip the rank owns.

Launch with a binding from models/demos/deepseek_v3_d_p/utils/rr_pipeline_binding.py:

    tt-run --rank-binding <binding.yaml> --mpi-args "--map-by rankfile:file=<rankfile>" \\
        pytest tests/ttnn/distributed/test_rr_prefill_mesh_allgather.py
"""

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn

SIDE = 4
BLOCK = 32


def rank_of_process():
    return int(ttnn.distributed_context_get_rank()) if ttnn.using_distributed_env() else 0


def barrier():
    if ttnn.using_distributed_env():
        barrier()


@pytest.fixture(scope="module")
def rr_mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D_TORUS_XY)
    try:
        mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(SIDE, SIDE))
        try:
            yield mesh_device
        finally:
            barrier()
            ttnn.close_mesh_device(mesh_device)
    finally:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def block_value(r, c):
    return float(r * SIDE + c + 1)


def expected_gather(axis, r, c):
    """The row-major block a chip (r, c) must hold after gathering along mesh axis `axis`."""
    if axis == 0:
        blocks = [np.full((BLOCK, BLOCK), block_value(i, c), dtype=np.float32) for i in range(SIDE)]
        return np.concatenate(blocks, axis=0)
    blocks = [np.full((BLOCK, BLOCK), block_value(r, j), dtype=np.float32) for j in range(SIDE)]
    return np.concatenate(blocks, axis=1)


@pytest.mark.parametrize("axis", [0, 1], ids=["axis0", "axis1"])
@pytest.mark.parametrize("num_links", [1, 2])
def test_ring_all_gather_on_every_chip(rr_mesh, axis, num_links):
    mesh_device = rr_mesh
    view = mesh_device.get_view()
    rank = rank_of_process()

    full = torch.cat(
        [
            torch.cat([torch.full((BLOCK, BLOCK), block_value(r, c)) for c in range(SIDE)], dim=1)
            for r in range(SIDE)
        ],
        dim=0,
    ).reshape(1, 1, SIDE * BLOCK, SIDE * BLOCK)
    mapper = ttnn.create_mesh_mapper(
        ttnn.MeshShape(SIDE, SIDE),
        ttnn.MeshMapperConfig([ttnn.PlacementShard(2), ttnn.PlacementShard(3)]),
    )
    host = ttnn.from_torch(full, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
    device_tensor = ttnn.to_device(host, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    gathered = ttnn.all_gather(
        device_tensor,
        dim=2 + axis,
        cluster_axis=axis,
        num_links=num_links,
        topology=ttnn.Topology.Ring,
    )
    ttnn.synchronize_device(mesh_device)
    result = ttnn.from_device(ttnn.to_layout(gathered, ttnn.ROW_MAJOR_LAYOUT))
    buffer = result.host_buffer()

    checked = 0
    for coord in result.tensor_topology().mesh_coords():
        if not view.is_local(coord):
            continue
        shard = buffer.get_shard(coord)
        if shard is None:
            continue
        got = np.from_dlpack(shard).astype(np.float32).reshape(-1)
        want = expected_gather(axis, int(coord[0]), int(coord[1])).reshape(-1)
        assert np.array_equal(got, want), f"rank {rank} chip ({coord[0]},{coord[1]}) axis {axis} links {num_links}"
        checked += 1
    assert checked > 0, "no local chip was checked"
    logger.info(f"RR_CHECK OK rank {rank} axis {axis} links {num_links}: {checked} local chips")
    barrier()
