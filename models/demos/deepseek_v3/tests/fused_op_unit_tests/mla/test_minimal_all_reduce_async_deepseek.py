# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn


@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 2129920,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        }
    ],
    indirect=True,
)
def test_minimal_all_reduce_async_width_sharded(
    mesh_device,
    function_level_defaults,
):
    """
    Test the minimal all_reduce_async operation with WIDTH_SHARDED memory configuration.

    This test uses the optimized all_reduce_async path that requires:
    - A persistent buffer_tensor for output
    - A single multi_device_global_semaphore (not a list)
    - use_noc1_only and use_optimal_ccl_for_llama flags

    Configuration:
    - Input shape: [1, 1, 32, 2112] (after matmul wq_kv_a)
    - Input memory: WIDTH_SHARDED 7x4 grid, shard [32, 128]
    - Buffer memory: WIDTH_SHARDED 7x4 grid, shard [32, 1280] (larger for ring intermediates)
    - Cluster axis: 0 (reduce across 8 devices in column)
    - Topology: Ring
    - Num links: 4
    """
    torch.manual_seed(0)

    # Set up sub-devices and semaphores for async operation
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

    # Create single global semaphore for minimal all_reduce_async
    multi_device_global_semaphore = ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0)

    # Input shape matching tt_q_kv after matmul in test_wq_kv_a_sequence
    input_shape = [1, 1, 32, 2112]

    logger.info(f"Running minimal all_reduce_async test")
    logger.info(f"Running on {mesh_device.get_num_devices()} devices")
    logger.info(f"Input shape: {input_shape}")

    # Create golden reference tensor
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

    # Expected output: all_reduce (sum) across 8 devices means output = input * 8
    # since all devices have replicated copies
    torch_expected = torch_input * 8.0

    # Create WIDTH_SHARDED memory config for input (7x4 grid, shard [32, 128])
    # 7x4 = 28 cores, covering width 2112 with 128 per shard: 2112/128 = 16.5 shards
    matmul_core_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(6, 3),  # 7x4 grid (28 cores)
            )
        }
    )

    sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(32, 128),
        core_grid=matmul_core_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Convert input to ttnn with proper mesh mapping for 8x4 mesh, replicated
    input_tensor_mesh = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh_device,
            ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementReplicate()], ttnn.MeshShape(8, 4)),
        ),
    )

    # Apply WIDTH sharding to input
    input_tensor_mesh = ttnn.to_memory_config(input_tensor_mesh, sharded_mem_config)

    # Create persistent buffer tensor with larger shard for intermediate results
    # Buffer needs to hold output_shard * ring_size (8 devices)
    # Required: 40,960 elements = 32 height * 1,280 width
    buffer_shard_config = ttnn.create_sharded_memory_config(
        shape=(32, 1280),
        core_grid=matmul_core_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    buffer_tensor = ttnn.empty(
        shape=input_shape,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=buffer_shard_config,
    )

    logger.info("Calling minimal all_reduce_async (invoke 4)")

    # Call the minimal all_reduce_async
    # This uses the invoke at line 426 in all_reduce_async.cpp (invoke 4)
    output_tensor = ttnn.experimental.all_reduce_async(
        input_tensor_mesh,
        buffer_tensor,
        cluster_axis=0,
        mesh_device=mesh_device,
        multi_device_global_semaphore=multi_device_global_semaphore,
        dtype=ttnn.bfloat16,
        memory_config=sharded_mem_config,
        topology=ttnn.Topology.Linear,
        num_links=4,
        subdevice_id=worker_sub_device_id,
        use_noc1_only=False,
        use_optimal_ccl_for_llama=False,
    )

    logger.info("All-reduce completed, verifying output")

    # Verify correctness
    passed = True
    for i, t in enumerate(ttnn.get_device_tensors(output_tensor)):
        tt_output_tensor = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        logger.info(f"Checking output for device {t.device().id()}")

        # Check if output matches expected (input * 8)
        if not torch.allclose(tt_output_tensor, torch_expected, rtol=1e-2, atol=1e-2):
            logger.error(f"Output mismatch for device {i}")
            logger.error(f"Expected mean: {torch_expected.mean():.6f}, Got: {tt_output_tensor.mean():.6f}")
            logger.error(f"Expected std: {torch_expected.std():.6f}, Got: {tt_output_tensor.std():.6f}")
            passed = False
        else:
            logger.info(f"✓ Device {i} output matches expected (mean: {tt_output_tensor.mean():.6f})")

    # Clean up
    mesh_device.reset_sub_device_stall_group()

    assert passed, "Output verification failed for one or more devices"
    logger.info("✓ Minimal all_reduce_async test passed with correct outputs")
