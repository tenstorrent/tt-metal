# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host cost per call (program-cache hit) of fabric_all_gather vs high_bw_all_gather at the MoE gather shapes: N
back-to-back calls enqueued with no sync (host time / N), then the device drain; and the mesh workload's program
count. MIMO_GH_CALLS (50)."""

import os
import time

import torch

import ttnn
from models.demos.mimo_v2_d_p.tests.mesh import MESH_PARAMS, mesh_id

N = int(os.environ.get("MIMO_GH_CALLS", "50"))


@MESH_PARAMS
def test_gather_host_cost(mesh_device, device_params):
    rows, cols = tuple(mesh_device.shape)
    S, H = 4096 // rows, 4096
    g = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})
    sems = [ttnn.create_global_semaphore(mesh_device, crs, 0) for _ in range(2)]
    shard = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(2, None))
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    for name, shape, lay, dt in (
        ("x_rm", (S, H), ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
        ("idx_tile", (S, 8), ttnn.TILE_LAYOUT, ttnn.uint16),
    ):
        t = torch.zeros(1, 1, rows * shape[0], shape[1])
        x = ttnn.from_torch(t, device=mesh_device, layout=lay, dtype=dt, mesh_mapper=shard)
        out = ttnn.from_torch(t, device=mesh_device, layout=lay, dtype=dt, mesh_mapper=rep)
        ops = {
            "high_bw": lambda: ttnn.experimental.high_bw_all_gather(x, dim=2, output_tensor=out, cluster_axis=0),
            "fabric": lambda: ttnn.experimental.fabric_all_gather(x, dim=2, output_tensor=out, cluster_axis=0),
            "fabric+sems": lambda: ttnn.experimental.fabric_all_gather(
                x, dim=2, output_tensor=out, cluster_axis=0, ready_semaphore=sems[0], data_valid_semaphore=sems[1]
            ),
        }
        for op_name, f in ops.items():
            f()
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            for _ in range(N):
                f()
            t1 = time.perf_counter()
            ttnn.synchronize_device(mesh_device)
            t2 = time.perf_counter()
            print(
                f"GATHER_HOST {mesh_id(mesh_device)} {name} {op_name}: host {(t1 - t0) / N * 1e6:.1f} us/call, "
                f"wall {(t2 - t0) / N * 1e6:.1f} us/call"
            )
