# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Regression test for a mesh trace replay hang (#60817): the prefetcher could return from an exec_buf with a read of
# the trace's padding pages still in flight, which overwrote the host commands fetched after the trace.
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

    def ops(n):
        # Programs on device (0, 0) only, so every other device's trace is n dummy GO signals (128 B each) and ends
        # well before the trace buffer does.
        for _ in range(n):
            ttnn.point_to_point(x, coord, coord, topology=ttnn.Topology.Linear).deallocate(True)

    # The hang needs a short device's EXEC_BUF_END near the start of a 16 KB cmddat_q chunk (2048 ops is one
    # 256 KB lap) and the next host commands inside the in-flight read, so sweep both the op count and the
    # number of host writes between replays. The sweep is tuned to BH Galaxy; on WH Galaxy it passes without the fix.
    for n in range(2032, 2068, 4):
        ops(n)
        ttnn.synchronize_device(mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        ops(n)
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        for i in range(1, 41):
            for _ in range(i % 17):
                ttnn.copy_host_to_device_tensor(host, target)
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
        ttnn.release_trace(mesh_device, tid)
