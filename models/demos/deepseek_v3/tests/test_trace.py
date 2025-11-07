# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
def test_tracing(mesh_device):
    """
    Test reduce_scatter_minimal_async followed by all_gather_async.
    Starting with a DRAM interleaved tensor of shape [1, 1, 32, 1536] replicated on entire mesh.
    """
    logger.info(f"Running test_reduce_scatter_all_gather_async with mesh_device: {mesh_device.shape}")
    # Input shape: [1, 1, 32, 1536]
    shape = [1, 32, 32, 576]

    # Create random input tensor
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    # Create DRAM interleaved tensor replicated on entire mesh
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # tt_input = ttnn.pad(tt_input, [(0, 0), (0, ttnn.TILE_SIZE - 1), (0, 0), (0, 0)], 0)

    # tt_input = ttnn.permute(tt_input, (0, 2, 1, 3))

    grid = mesh_device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    core_range_set = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)

    # Create global semaphores for reduce_scatter
    rs_multi_device_semaphores = [ttnn.create_global_semaphore(mesh_device, core_range_set, 0) for _ in range(3)]

    rs_barrier_semaphore = ttnn.create_global_semaphore(mesh_device, core_range_set, 0)

    ttnn.synchronize_device(mesh_device)

    # Configure reduce_scatter_minimal_async
    rs_config = {
        "dim": 1,
        "multi_device_global_semaphore": rs_multi_device_semaphores,
        "num_links": 1,
        "barrier_semaphore": rs_barrier_semaphore,
        "memory_config": ttnn.DRAM_MEMORY_CONFIG,
        "topology": ttnn.Topology.Linear,
        "cluster_axis": 1,
    }

    # Apply reduce_scatter
    tt_output_rs = ttnn.experimental.reduce_scatter_minimal_async(tt_input, **rs_config)
    logger.info(f"tt_output_rs: {tt_output_rs.shape}")
    ttnn.synchronize_device(mesh_device)

    logger.info("Test passed!")


if __name__ == "__main__":
    pytest.main([__file__])
