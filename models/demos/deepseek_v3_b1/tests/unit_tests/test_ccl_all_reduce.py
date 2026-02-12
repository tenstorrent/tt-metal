# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN CCL All-Reduce Test

Tests the deepseek_minimal_all_reduce operation implemented using the generic op infrastructure.
This test validates all-reduce on a 1D mesh (2 devices) where:
1. Each device sends its data to its neighbor
2. Each device receives data from its neighbor
3. Each device sums local data with received data
4. Optionally, a residual tensor is added to the final result to fuse the next residual add block
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.ccl_all_reduce.op import DeepseekMinimalAllReduce
from models.perf.benchmarking_utils import BenchmarkProfiler


def create_fabric_router_config(max_payload_size):
    """Helper to create FabricRouterConfig with custom max payload size."""
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


@pytest.mark.parametrize(
    "num_devices, output_shape, input_shard_shape, tensor_mem_layout",
    [
        (
            2,
            [1, 7168],
            (1, 7168),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("cluster_axis", [0])
@pytest.mark.parametrize("use_persistent", [True])
@pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)  # Open full mesh, create submesh
@pytest.mark.parametrize("num_iter, num_warmup_iter", [(30, 15)])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 573440,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("fuse_residual_add", [True])
def test_ccl_all_reduce(
    mesh_device,
    num_devices,
    output_shape,
    input_shard_shape,
    tensor_mem_layout,
    layout,
    input_dtype,
    cluster_axis,
    use_persistent,
    fuse_residual_add,
    num_warmup_iter,
    num_iter,
):
    # Validate mesh size
    if mesh_device.shape[0] * mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    # Create submesh - fabric requires opening full system mesh first
    submesh = mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))

    # Set up sub-device
    compute_grid_size = submesh.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
    sub_device_manager = submesh.create_sub_device_manager([worker_sub_device], 0)
    submesh.load_sub_device_manager(sub_device_manager)
    submesh.set_sub_device_stall_group(sub_device_stall_group)

    # Set up sharded memory config
    input_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    input_shard_spec = ttnn.ShardSpec(
        input_shard_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=input_shard_spec)

    # Create input tensors for each device
    device_tensors = []
    for device_idx in range(num_devices):
        tensor = torch.rand(output_shape, dtype=torch.bfloat16)
        device_tensors.append(tensor)

    # Create mesh tensor (concatenate along dim 0)
    mesh_tensor_torch = torch.cat(device_tensors, dim=0)
    mesh_mapper_config = ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementReplicate()], submesh.shape)
    input_tensor_mesh = ttnn.from_torch(
        mesh_tensor_torch,
        device=submesh,
        layout=layout,
        tile=ttnn.Tile((1, 32)),
        dtype=input_dtype,
        memory_config=input_mem_config,
        mesh_mapper=ttnn.create_mesh_mapper(submesh, mesh_mapper_config),
    )

    # Optionally create residual tensor
    if fuse_residual_add:
        residual_tensor = torch.rand(output_shape, dtype=torch.bfloat16)
        residual_tensors = [residual_tensor for _ in range(num_devices)]
        residual_mesh_tensor_torch = torch.cat(residual_tensors, dim=0)
        residual_tensor_mesh = ttnn.from_torch(
            residual_mesh_tensor_torch,
            device=submesh,
            layout=layout,
            tile=ttnn.Tile((1, 32)),
            dtype=input_dtype,
            memory_config=input_mem_config,
            mesh_mapper=ttnn.create_mesh_mapper(submesh, mesh_mapper_config),
        )
    else:
        residual_tensor_mesh = None

    # Create output tensor with tiny tiles (1x32), same as input
    output_tensor_per_device = torch.zeros(output_shape, dtype=torch.bfloat16)
    output_tensor_torch = torch.cat([output_tensor_per_device] * num_devices, dim=0)
    output_tensor = ttnn.from_torch(
        output_tensor_torch,
        device=submesh,
        layout=layout,
        tile=ttnn.Tile((1, 32)),
        dtype=input_dtype,
        memory_config=input_mem_config,
        mesh_mapper=ttnn.create_mesh_mapper(submesh, mesh_mapper_config),
    )

    # Create intermediate tensor with standard 32x32 tiles
    # Intermediate shape: [32, 224] to hold 7 standard tiles (7168 elements = 7 * 32 * 32)
    intermediate_shape = [32, 224]
    intermediate_shard_shape = tuple(intermediate_shape)
    intermediate_tensor_torch = torch.zeros(intermediate_shape, dtype=torch.bfloat16)
    intermediate_shard_spec = ttnn.ShardSpec(
        input_shard_grid,
        intermediate_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    intermediate_mem_config = ttnn.MemoryConfig(
        tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=intermediate_shard_spec
    )

    # Concatenate for mesh
    intermediate_mesh_torch = torch.cat([intermediate_tensor_torch] * num_devices, dim=0)
    intermediate_tensor = ttnn.from_torch(
        intermediate_mesh_torch,
        device=submesh,
        layout=layout,
        tile=ttnn.Tile((32, 32)),
        dtype=input_dtype,
        memory_config=intermediate_mem_config,
        mesh_mapper=ttnn.create_mesh_mapper(submesh, mesh_mapper_config),
    )

    # Compute expected output using golden function
    if fuse_residual_add:
        torch_expected = DeepseekMinimalAllReduce.golden(device_tensors, residual_tensor)
    else:
        torch_expected = DeepseekMinimalAllReduce.golden(device_tensors)

    if use_persistent:
        persistent_output_tensor = output_tensor
    else:
        persistent_output_tensor = None

    # semaphores
    num_cores = compute_grid_size.x * compute_grid_size.y
    available_cores = ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, row_wise=True)
    semaphore1 = ttnn.create_global_semaphore(submesh, available_cores, 0)
    semaphore2 = ttnn.create_global_semaphore(submesh, available_cores, 0)
    semaphores = [semaphore1, semaphore2]

    # Run all-reduce operation
    logger.info(f"Running CCL all-reduce: num_devices={num_devices}")

    profiler = BenchmarkProfiler()

    # Compile Run
    logger.info("Compiling model")
    ttnn_result = DeepseekMinimalAllReduce.op(
        input_tensor_mesh,
        intermediate_tensor,
        cluster_axis=cluster_axis,
        persistent_output_tensor=persistent_output_tensor,
        residual_tensor_mesh=residual_tensor_mesh,
        semaphores=semaphores,
    )
    ttnn.synchronize_device(submesh)

    # Capture warmup trace
    logger.info("Capturing warmup trace")
    trace_id_warmup = ttnn.begin_trace_capture(submesh, cq_id=0)
    for i in range(num_warmup_iter):
        ttnn_result = DeepseekMinimalAllReduce.op(
            input_tensor_mesh,
            intermediate_tensor,
            cluster_axis=cluster_axis,
            persistent_output_tensor=persistent_output_tensor,
            residual_tensor_mesh=residual_tensor_mesh,
            semaphores=semaphores,
        )
    ttnn.end_trace_capture(submesh, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(submesh)

    # Capture main trace
    logger.info("Capturing trace")
    trace_id = ttnn.begin_trace_capture(submesh, cq_id=0)
    for i in range(num_iter):
        ttnn_result = DeepseekMinimalAllReduce.op(
            input_tensor_mesh,
            intermediate_tensor,
            cluster_axis=cluster_axis,
            persistent_output_tensor=persistent_output_tensor,
            residual_tensor_mesh=residual_tensor_mesh,
            semaphores=semaphores,
        )
    ttnn.end_trace_capture(submesh, trace_id, cq_id=0)
    ttnn.synchronize_device(submesh)

    # Execute warmup trace
    logger.info("Executing warmup trace...")
    profiler.start("deepseek-all-reduce-warmup")
    ttnn.execute_trace(submesh, trace_id_warmup, blocking=False)
    ttnn.release_trace(submesh, trace_id_warmup)
    ttnn.synchronize_device(submesh)
    profiler.end("deepseek-all-reduce-warmup")

    # Execute main trace with signposts for profiling
    logger.info("Starting Trace perf test...")
    signpost("start")
    profiler.start("deepseek-all-reduce-trace")

    ttnn.execute_trace(submesh, trace_id, blocking=False)
    ttnn.release_trace(submesh, trace_id)
    ttnn.synchronize_device(submesh)

    profiler.end("deepseek-all-reduce-trace")
    signpost("stop")

    # Verify output
    logger.info("Verifying all-reduce results...")

    # Convert output tensor to torch
    output_tensor_torch = ttnn.to_torch(
        ttnn_result,
        mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0),
    )

    all_passed = True
    for device_idx in range(num_devices):
        received = output_tensor_torch[device_idx : device_idx + 1, :]

        assert received.shape == torch_expected.shape, f"Shape mismatch at device {device_idx}"

        if not torch.allclose(received, torch_expected, rtol=1e-2, atol=1e-2):
            logger.error(f"Output mismatch for device {device_idx}")
            logger.error(f"Expected: {torch_expected[:5, :5]}")
            logger.error(f"Received: {received[:5, :5]}")
            all_passed = False
        else:
            logger.info(f"Device {device_idx}: PASSED")

    # Cleanup
    submesh.reset_sub_device_stall_group()
    submesh.clear_loaded_sub_device_manager()

    assert all_passed, "Not all devices have the correct all-reduced data"
    logger.info("CCL all-reduce test passed!")
