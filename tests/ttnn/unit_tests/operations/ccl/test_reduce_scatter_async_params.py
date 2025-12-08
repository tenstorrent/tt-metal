# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from ttnn import ConcatMeshToTensor
import os
from models.tt_transformers.tt.generator import create_submeshes
from tracy import signpost


@pytest.mark.parametrize("num_iters", [5])
@pytest.mark.parametrize("num_workers_per_link", [1, 2, 3, 4])  # 1
@pytest.mark.parametrize("num_buffers_per_channel", [1, 2, 4, 8, 16])  # 2
@pytest.mark.parametrize("chunks_per_sync", [5, 10, 128, 256])  # 10
@pytest.mark.parametrize(
    "input_shard_grid, input_shard_shape, output_shard_grid, output_shard_shape",
    [
        # 8x2 input grid (16 cores), 8x4 output grid (32 cores)
        (
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))}),
            (32, 256),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
            (32, 64),
        ),
    ],
    ids=["8x2_to_8x4"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_reduce_scatter_async_params(
    mesh_device,
    num_workers_per_link,
    num_buffers_per_channel,
    chunks_per_sync,
    input_shard_grid,
    input_shard_shape,
    output_shard_grid,
    output_shard_shape,
    num_iters,
):
    """Test reduce_scatter_minimal_async with different parameter combinations"""

    if os.environ.get("MESH_DEVICE") == "TG":
        mesh_device = create_submeshes(mesh_device, 4)[0]

    num_devices = mesh_device.get_num_devices()
    if num_devices < 2:
        pytest.skip(f"Need at least 2 devices, got {num_devices}")

    run_reduce_scatter_async_test(
        mesh_device,
        num_workers_per_link,
        num_buffers_per_channel,
        chunks_per_sync,
        input_shard_grid,
        input_shard_shape,
        output_shard_grid,
        output_shard_shape,
        num_iters,
    )


def run_reduce_scatter_async_test(
    mesh_device,
    num_workers_per_link,
    num_buffers_per_channel,
    chunks_per_sync,
    input_shard_grid,
    input_shard_shape,
    output_shard_grid,
    output_shard_shape,
    num_iters,
):
    """Test reduce_scatter_minimal_async with parameterized worker/buffer/sync configs"""

    # Input setup - shape (1, 1, 32, 4096) per device, bfloat16, tiled
    input_shape = [1, 1, 32, 4096]
    dim = 3
    num_devices = mesh_device.get_num_devices()

    # Create WIDTH_SHARDED memory config for input
    # Input: 8x2 grid (16 cores), shard shape (32, 256) → width = 16 × 256 = 4096
    input_shard_spec = ttnn.ShardSpec(
        input_shard_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    input_mem_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=input_shard_spec,
    )

    # Output memory config - after reduce_scatter, width is divided by num_devices
    # Input per device: (1, 1, 32, 4096) with shard (32, 256) on 16 cores
    # Output per device: (1, 1, 32, 2048) with shard (32, 64) on 32 cores (8x4 grid)
    output_shard_spec = ttnn.ShardSpec(
        output_shard_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    output_mem_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=output_shard_spec,
    )

    # Setup sub-devices
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]

    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    # Create global semaphores - reduce_scatter needs list of 3 semaphores
    rs_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(3)]
    barrier_semaphore_handle = ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0)

    try:
        # Generate golden input - each device gets same input (replicated)
        torch.manual_seed(0)
        input_tensor_golden = torch.rand(input_shape).bfloat16()

        # Create input tensor - replicate to all devices with WIDTH_SHARDED config
        input_tensor_mesh = ttnn.from_torch(
            input_tensor_golden,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=input_mem_cfg,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        logger.info(
            f"Running reduce_scatter_minimal_async with num_workers_per_link={num_workers_per_link}, "
            f"num_buffers_per_channel={num_buffers_per_channel}, chunks_per_sync={chunks_per_sync}"
        )

        signpost(f"========")
        signpost(f"rs_{num_workers_per_link}_{num_buffers_per_channel}_{chunks_per_sync}")

        # Run reduce_scatter_minimal_async - requires list of 3 semaphores
        for _ in range(num_iters):
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
            tt_out = ttnn.experimental.reduce_scatter_minimal_async(
                input_tensor_mesh,
                persistent_output_buffers=None,
                dim=dim,
                multi_device_global_semaphore=rs_semaphore_handles,
                barrier_semaphore=barrier_semaphore_handle,
                num_links=4,
                memory_config=output_mem_cfg,
                intermediate_memory_config=ttnn.L1_MEMORY_CONFIG,
                topology=ttnn.Topology.Ring,
                chunks_per_sync=chunks_per_sync,
                num_workers_per_link=num_workers_per_link,
                num_buffers_per_channel=num_buffers_per_channel,
            )
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)

        # Verify output
        tt_out_torch = ttnn.from_device(tt_out)
        tt_out_torch = ttnn.to_torch(tt_out_torch, mesh_composer=ConcatMeshToTensor(mesh_device, dim=dim))

        # Expected output: reduce_scatter does sum reduction + scatter
        # Input: each device has (1, 1, 32, 4096)
        # After reduce: sum of all inputs = input * num_devices
        # After scatter: split reduced result along dim=3
        # Each device gets (1, 1, 32, 2048) slice (for num_devices=2)
        # Concatenating all outputs gives the full reduced tensor
        expected_reduced = input_tensor_golden * num_devices  # sum reduction

        # Compare - after reduce_scatter, concatenating all device outputs should give full reduced tensor
        eq, output = comp_pcc(tt_out_torch, expected_reduced, 0.9999)
        logger.info(f"PCC check: {output}")
        assert eq, f"FAILED: {output}"

        logger.info("Test PASSED")

    finally:
        mesh_device.reset_sub_device_stall_group()
        mesh_device.clear_loaded_sub_device_manager()
