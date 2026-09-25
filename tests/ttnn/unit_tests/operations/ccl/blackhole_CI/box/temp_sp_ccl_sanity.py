# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# TEMP: CCL ops under the streaming profiler, for the global-timeline validation.
#   CCL_OP=ag|rs|ar|ab  CCL_FABRIC=1d|ring|2d  CCL_AXIS=0|1  CCL_SHAPE=H,W  CCL_ITERS=N  CCL_LOAD=K (K matmuls per op)
import os

import pytest
import torch
import ttnn

_FAB = {
    "1d": (ttnn.FabricConfig.FABRIC_1D, ttnn.Topology.Linear),
    "ring": (ttnn.FabricConfig.FABRIC_1D_RING, ttnn.Topology.Ring),
    "2d": (ttnn.FabricConfig.FABRIC_2D, ttnn.Topology.Linear),
}[os.environ.get("CCL_FABRIC", "1d")]


@pytest.mark.parametrize("device_params", [{"fabric_config": _FAB[0]}], indirect=True)
def test_sp_ccl(bh_2d_mesh_device):
    mesh = bh_2d_mesh_device
    op = os.environ.get("CCL_OP", "ag")
    axis = int(os.environ.get("CCL_AXIS", "0"))
    h, w = (int(v) for v in os.environ.get("CCL_SHAPE", "512,2048").split(","))
    iters = int(os.environ.get("CCL_ITERS", "200"))
    rows, cols = mesh.shape[0], mesh.shape[1]
    x = torch.randn([rows, cols, h, w]).bfloat16()
    tt_x = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=(rows, cols)),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    topo = _FAB[1]
    load = int(os.environ.get("CCL_LOAD", "0"))
    if load:
        a = ttnn.from_torch(torch.randn([1, 1, 2048, 4096]).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        b = ttnn.from_torch(torch.randn([1, 1, 4096, 4096]).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for _ in range(iters):
        for _ in range(load):
            ttnn.matmul(a, b)
        if op == "ag":
            out = ttnn.all_gather(tt_x, dim=3, cluster_axis=axis, topology=topo)
        elif op == "rs":
            out = ttnn.reduce_scatter(tt_x, dim=3, cluster_axis=axis, topology=topo)
        elif op == "ar":
            out = ttnn.all_reduce(tt_x, cluster_axis=axis, topology=topo)
        elif op == "ab":
            out = ttnn.all_broadcast(tt_x, cluster_axis=axis, topology=topo)
        else:
            raise ValueError(op)
    ttnn.synchronize_device(mesh)
    print(f"DONE op={op} axis={axis} fabric={os.environ.get('CCL_FABRIC', '1d')} iters={iters}")
