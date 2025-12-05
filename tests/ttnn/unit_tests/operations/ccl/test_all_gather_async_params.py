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


# @pytest.mark.parametrize("chunks_per_sync", [5, 10, 20, 40, 80, 1000])
@pytest.mark.parametrize("num_workers_per_link", [1, 2, 3, 4])
@pytest.mark.parametrize("num_buffers_per_channel", [1, 2, 3, 4])
@pytest.mark.parametrize("chunks_per_sync", [5, 10, 20])
@pytest.mark.parametrize(
    "shard_grid, input_shard_shape",
    [
        # First norm config
        (
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))}),
            (32, 32),
        ),
        # Second norm config
        (
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
            (32, 128),
        ),
    ],
    ids=["first_norm", "second_norm"],
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
def test_all_gather_async_params(
    mesh_device,
    num_workers_per_link,
    num_buffers_per_channel,
    chunks_per_sync,
    shard_grid,
    input_shard_shape,
):
    """Test all_gather_async with different parameter combinations"""

    if os.environ.get("MESH_DEVICE") == "TG":
        mesh_device = create_submeshes(mesh_device, 4)[0]

    num_devices = mesh_device.get_num_devices()
    if num_devices < 4:
        pytest.skip(f"Need at least 4 devices, got {num_devices}")

    run_all_gather_async_test(
        mesh_device,
        num_workers_per_link,
        num_buffers_per_channel,
        chunks_per_sync,
        shard_grid,
        input_shard_shape,
    )


def run_all_gather_async_test(
    mesh_device,
    num_workers_per_link,
    num_buffers_per_channel,
    chunks_per_sync,
    shard_grid,
    input_shard_shape,
):
    """Test all_gather_async with parameterized worker/buffer/sync configs"""

    # Input setup - shape (1, 1, 32, 512) per device, bfloat16, tiled
    ag_output_shape = [1, 1, 32, 512]
    dim = 3
    num_devices = mesh_device.get_num_devices()
    num_links = 4 if os.environ.get("MESH_DEVICE") == "TG" else 1

    # Create WIDTH_SHARDED memory config
    input_shard_spec = ttnn.ShardSpec(
        shard_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    input_mem_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=input_shard_spec,
    )

    # Output memory config - width multiplied by num_devices for all_gather on dim=3
    output_shard_shape = (input_shard_shape[0], input_shard_shape[1] * num_devices)
    output_shard_spec = ttnn.ShardSpec(
        shard_grid,
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

    # Create global semaphores
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(2)]
    barrier_semaphore_handle = ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0)

    try:
        # Generate golden output
        torch.manual_seed(0)
        ag_output_tensor_golden = torch.rand(ag_output_shape).bfloat16()

        # Create input tensor - shard along dim=3
        input_tensor_mesh = ttnn.from_torch(
            ag_output_tensor_golden,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=input_mem_cfg,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=dim),
        )

        logger.info(
            f"Running all_gather_async with num_workers_per_link={num_workers_per_link}, "
            f"num_buffers_per_channel={num_buffers_per_channel}, chunks_per_sync={chunks_per_sync}"
        )

        # Determine grid type for signpost
        grid_type = "first" if input_shard_shape == (32, 32) else "second"
        signpost(f"========")
        signpost(f"{grid_type}_{num_workers_per_link}_{num_buffers_per_channel}_{chunks_per_sync}")

        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)

        # Run all_gather_async - requires list of 2 semaphores
        tt_out = ttnn.experimental.all_gather_async(
            input_tensor_mesh,
            persistent_output_buffer=None,
            dim=dim,
            multi_device_global_semaphore=ccl_semaphore_handles,
            num_links=num_links,
            topology=ttnn.Topology.Ring,
            memory_config=output_mem_cfg,
            barrier_semaphore=barrier_semaphore_handle,
            chunks_per_sync=chunks_per_sync,
            num_workers_per_link=num_workers_per_link,
            num_buffers_per_channel=num_buffers_per_channel,
            subdevice_id=worker_sub_device_id,
        )

        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)

        # Verify output
        tt_out_torch = ttnn.from_device(tt_out)
        tt_out_torch = ttnn.to_torch(tt_out_torch, mesh_composer=ConcatMeshToTensor(mesh_device, dim=dim))

        # Compare - after all_gather, all devices should have the full tensor
        # Each device originally had ag_output_shape[dim] / num_devices elements
        # After gather, each should have the full ag_output_shape
        expected_full_shape = list(ag_output_shape)
        expected_full_shape[dim] = ag_output_shape[dim] * num_devices

        # Create expected output by replicating the gathered result
        expected_tensor = ag_output_tensor_golden.repeat(1, 1, 1, num_devices)

        eq, output = comp_pcc(tt_out_torch[:, :, :, : expected_tensor.shape[3]], expected_tensor, 0.9999)
        logger.info(f"PCC check: {output}")
        assert eq, f"FAILED: {output}"

        logger.info("Test PASSED")

    finally:
        mesh_device.reset_sub_device_stall_group()
        mesh_device.clear_loaded_sub_device_manager()
