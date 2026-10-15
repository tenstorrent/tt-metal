# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Repro for a mesh trace replay hang (BH Galaxy 8x4): the prefetcher can return from an exec_buf with a read of the
# trace's padding pages still in flight, which overwrites the host commands fetched after the trace.
import pytest
import torch

import ttnn


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((8, 4), {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 100_000_000})],
    indirect=True,
)
def test_trace_exec_buf_end_race(mesh_device):
    mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    x = ttnn.from_torch(
        torch.randn(1, 1, 32, 128), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mapper
    )
    host = ttnn.from_torch(torch.randn(1, 1, 32, 1024), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
    target = ttnn.to_device(host, mesh_device)
    coord = ttnn.MeshCoordinate(0, 0)

    def ops():
        # Programs on device (0, 0) only: every other device's trace is 2048 dummy GO signals, much shorter.
        for _ in range(2048):
            ttnn.point_to_point(x, coord, coord, topology=ttnn.Topology.Linear).deallocate(True)

    ops()
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    ops()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    for i in range(1, 401):
        for _ in range(i % 17):
            ttnn.copy_host_to_device_tensor(host, target)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)  # hangs at replay 1 without the fix
    ttnn.release_trace(mesh_device, tid)
