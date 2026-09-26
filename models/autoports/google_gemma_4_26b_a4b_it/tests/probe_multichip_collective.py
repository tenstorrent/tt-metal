# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shape-faithful TP4 collective smoke; run separately from model validation."""

import torch

import ttnn
from models.demos.gpt_oss.tt.ccl import CCLManager


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=1048576)
    try:
        ccl = CCLManager(mesh, 1, ttnn.Topology.Linear)
        for dtype in (ttnn.bfloat16, ttnn.float32):
            x = ttnn.from_torch(
                torch.ones(1, 1, 32, 2816),
                device=mesh,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            y = ttnn.experimental.reduce_scatter_minimal_async(
                x,
                dim=3,
                multi_device_global_semaphore=ccl.get_rs_ping_pong_semaphore(),
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Linear,
                cluster_axis=1,
                barrier_semaphore=ccl.get_barrier_semaphore(),
            )
            result = ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1))
            assert torch.equal(result, torch.full_like(result, 4))
            print("PASS", dtype, result.shape, flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
