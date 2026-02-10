# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from tracy import signpost
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.common.utility_functions import skip_for_wormhole_b0

# CoreRangeSet for CCL operations (subset of compute grid)
CCL_CRS = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 7))])


def setup_all_reduce_sync(submesh_device, num_buffers=8):
    """
    Set up resources for all_reduce used as device synchronization barrier.
    Returns semaphores, intermediate tensors, and memory config needed for all_reduce_async.
    """
    # Create global semaphores for all_reduce
    ccl_semaphore_handles = [ttnn.create_global_semaphore(submesh_device, CCL_CRS, 0) for _ in range(num_buffers)]

    # Simple tensor shape for sync (small footprint, tile-aligned 32x32)
    sync_shape = [4, 2, 32, 32]  # mesh_shape + minimal 32x32 tile
    cluster_shape = (4, 2)

    # Memory config for sync tensor (32x32 shard for tile alignment)
    sync_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))]),
            [32, 32],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    # Intermediate tensor shape for all_reduce (gather dimension expanded)
    intermediate_shape = [4, 2, 32, 32 * cluster_shape[0]]  # expanded along cluster_axis=0
    intermediate_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))]),
            [32, 32 * cluster_shape[0]],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    # Create sync input tensor
    sync_data = torch.ones(sync_shape, dtype=torch.bfloat16)
    sync_tensor = ttnn.from_torch(
        sync_data,
        device=submesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=sync_mem_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh_device, dims=(0, 1), mesh_shape=cluster_shape),
    )

    # Create intermediate tensors for all_reduce
    intermediate_data = torch.zeros(intermediate_shape, dtype=torch.bfloat16)
    intermediate_tensors = []
    for _ in range(num_buffers):
        intermediate_tensor = ttnn.from_torch(
            intermediate_data,
            device=submesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=intermediate_mem_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(submesh_device, dims=(0, 1), mesh_shape=cluster_shape),
        )
        intermediate_tensors.append(intermediate_tensor)

    return {
        "semaphores": ccl_semaphore_handles,
        "sync_tensor": sync_tensor,
        "intermediate_tensors": intermediate_tensors,
        "mem_config": sync_mem_config,
    }


def compute_reference_reduce_to_one(data_per_device):
    """
    Compute the reference output for reduce_to_one operation.
    Simple sum of all device tensors.
    """
    result = data_per_device[0].clone()
    for i in range(1, len(data_per_device)):
        result = result + data_per_device[i]
    return result


def setup_reduce_to_one_test(mesh_device):
    """Common setup for reduce_to_one tests. Returns test configuration."""
    # Log mesh device info
    logger.info(f"mesh_device shape: {mesh_device.shape}")
    logger.info(f"mesh_device num_devices: {mesh_device.get_num_devices()}")

    # Validate mesh has enough devices for 4x2 submesh
    mesh_rows, mesh_cols = mesh_device.shape
    if mesh_rows * mesh_cols < 8:
        pytest.skip(f"Need at least 8 devices, got {mesh_rows * mesh_cols}")
    logger.info(f"Mesh is {mesh_rows}x{mesh_cols} = {mesh_rows * mesh_cols} devices")

    # Setup - create 4x2 submesh
    num_devices = 8
    exit_coord = (0, 1)
    root_coord = (1, 1)

    submesh_device = mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    logger.info(f"Created submesh with shape: {submesh_device.shape}")

    assert submesh_device.shape == ttnn.MeshShape((4, 2)), f"Expected 4x2 mesh, got {submesh_device.shape}"

    # Tensor shape: (1, 7168) sharded across 8 cores
    tensor_shape = [1, 7168]
    dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT
    tile = ttnn.Tile((1, 32))

    # Get optimal cores for DRAM access
    compute_cores = submesh_device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    num_cores = len(compute_cores)
    logger.info(f"Using {num_cores} optimal DRAM cores: {compute_cores[:8]}")

    # Build shard grid from optimal cores (use first 8 cores)
    num_shard_cores = 8
    shard_cores = compute_cores[:num_shard_cores]
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(core, core) for core in shard_cores})

    shard_shape = [1, 896]
    shard_spec = ttnn.ShardSpec(
        shard_grid,
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    mem_config = ttnn.MemoryConfig(ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec)

    # Mesh mapper
    mesh_mapper_config = ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)], submesh_device.shape)
    mesh_mapper = ttnn.create_mesh_mapper(submesh_device, mesh_mapper_config)

    # Create 3 intermediate tensors for 3 reduction rounds
    intermediate_tensors = []
    for _ in range(3):
        intermediate_data = torch.zeros([4, 2] + tensor_shape, dtype=torch.bfloat16)
        intermediate_tensor = ttnn.from_torch(
            intermediate_data,
            device=submesh_device,
            layout=layout,
            tile=tile,
            dtype=dtype,
            memory_config=mem_config,
            mesh_mapper=mesh_mapper,
        )
        intermediate_tensors.append(intermediate_tensor)

    # Create output tensor sharded on a single core (bottom-right of compute grid)
    # Get the full compute grid and use the bottom-right core
    compute_grid = submesh_device.compute_with_storage_grid_size()
    output_core = ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1)
    logger.info(f"Compute grid: {compute_grid}, output core: {output_core}")
    output_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(output_core, output_core)})
    output_shard_shape = tensor_shape  # Full tensor on single core
    output_shard_spec = ttnn.ShardSpec(
        output_shard_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, output_shard_spec
    )

    # Create output tensor (zeros, will be filled by reduce_to_one)
    output_data = torch.zeros([4, 2] + tensor_shape, dtype=torch.bfloat16)
    output_tensor = ttnn.from_torch(
        output_data,
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=output_mem_config,
        mesh_mapper=mesh_mapper,
    )
    logger.info(f"Created output tensor sharded on single core: {output_core}")

    # Generate test data
    data_per_device = []
    torch.manual_seed(42)
    for _ in range(num_devices):
        data = torch.randn(tensor_shape, dtype=torch.bfloat16)
        data_per_device.append(data)

    data_all = torch.stack(data_per_device, dim=0)
    data_all = data_all.reshape(4, 2, *tensor_shape)

    # Create input tensor
    input_tensor = ttnn.from_torch(
        data_all,
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config,
        mesh_mapper=mesh_mapper,
    )

    # Compute reference output
    ref_output = compute_reference_reduce_to_one(data_per_device)

    return {
        "submesh_device": submesh_device,
        "input_tensor": input_tensor,
        "intermediate_tensors": intermediate_tensors,
        "output_tensor": output_tensor,
        "ref_output": ref_output,
        "root_coord": root_coord,
        "exit_coord": exit_coord,
        "output_core": output_core,
    }


def verify_output(output_tensor, submesh_device, root_coord, ref_output):
    """Verify output matches reference."""
    output_torch = ttnn.to_torch(output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=0))

    root_device_idx = root_coord[0] * submesh_device.shape[1] + root_coord[1]
    output_root = output_torch[root_device_idx]

    rtol = 0.01
    atol = 0.05

    match = torch.allclose(output_root, ref_output, rtol=rtol, atol=atol)

    if not match:
        print(f"Output mismatch!")
        print(f"Reference:\n{ref_output[:1, :8]}")
        print(f"Output:\n{output_root[:1, :8]}")
        diff = torch.abs(output_root - ref_output)
        print(f"Max diff: {diff.max()}, Mean diff: {diff.mean()}")

    return match


def run_reduce_to_one(mesh_device, topology):
    """Run reduce_to_one test."""
    print(f"\n=== Testing reduce_to_one with {topology} ===")

    config = setup_reduce_to_one_test(mesh_device)

    # Run reduce_to_one
    print("Running reduce_to_one...")
    output_tensor = ttnn.experimental.deepseek_b1_reduce_to_one(
        config["input_tensor"],
        root_coord=ttnn.MeshCoordinate(config["root_coord"]),
        exit_coord=ttnn.MeshCoordinate(config["exit_coord"]),
        topology=topology,
        output_tensor=config["output_tensor"],
        intermediate_tensors=config["intermediate_tensors"],
    )
    ttnn.synchronize_device(config["submesh_device"])

    # Verify output
    print("\nVerifying output...")
    match = verify_output(
        output_tensor,
        config["submesh_device"],
        config["root_coord"],
        config["ref_output"],
    )

    assert match, "Output tensor does not match reference"
    print("Test passed!")


def run_reduce_to_one_with_trace(mesh_device, topology):
    """Run reduce_to_one test with trace capture and replay."""
    print(f"\n=== Testing reduce_to_one with trace ({topology}) ===")

    config = setup_reduce_to_one_test(mesh_device)
    submesh_device = config["submesh_device"]
    input_tensor = config["input_tensor"]
    intermediate_tensors = config["intermediate_tensors"]
    output_tensor_preallocated = config["output_tensor"]
    root_coord = config["root_coord"]
    exit_coord = config["exit_coord"]
    ref_output = config["ref_output"]

    # Set up all_reduce for device synchronization
    sync_config = setup_all_reduce_sync(submesh_device, num_buffers=8)
    sync_semaphores = sync_config["semaphores"]
    sync_tensor = sync_config["sync_tensor"]
    sync_intermediate_tensors = sync_config["intermediate_tensors"]
    sync_mem_config = sync_config["mem_config"]

    profiler = BenchmarkProfiler()

    # Run once to compile
    print("Running reduce_to_one (compiling)...")
    output_tensor = ttnn.experimental.deepseek_b1_reduce_to_one(
        input_tensor,
        root_coord=ttnn.MeshCoordinate(root_coord),
        exit_coord=ttnn.MeshCoordinate(exit_coord),
        topology=topology,
        output_tensor=output_tensor_preallocated,
        intermediate_tensors=intermediate_tensors,
    )
    # Also compile all_reduce
    _ = ttnn.experimental.all_reduce_async(
        sync_tensor,
        sync_intermediate_tensors[0],
        cluster_axis=0,
        mesh_device=submesh_device,
        multi_device_global_semaphore=sync_semaphores[0],
        memory_config=sync_mem_config,
        dtype=ttnn.bfloat16,
        topology=ttnn.Topology.Linear,
        num_links=1,
    )
    ttnn.synchronize_device(submesh_device)

    # Helper to run reduce_to_one with all_reduce sync every 4 iterations
    def run_with_sync(num_iters, sem_offset=0):
        for i in range(num_iters):
            output_tensor = ttnn.experimental.deepseek_b1_reduce_to_one(
                input_tensor,
                root_coord=ttnn.MeshCoordinate(root_coord),
                exit_coord=ttnn.MeshCoordinate(exit_coord),
                topology=topology,
                output_tensor=output_tensor_preallocated,
                intermediate_tensors=intermediate_tensors,
            )
            # Insert all_reduce every 4 iterations for device sync
            if (i + 1) % 4 == 0:
                sem_idx = ((i // 4) + sem_offset) % len(sync_semaphores)
                _ = ttnn.experimental.all_reduce_async(
                    sync_tensor,
                    sync_intermediate_tensors[sem_idx],
                    cluster_axis=0,
                    mesh_device=submesh_device,
                    multi_device_global_semaphore=sync_semaphores[sem_idx],
                    memory_config=sync_mem_config,
                    dtype=ttnn.bfloat16,
                    topology=ttnn.Topology.Linear,
                    num_links=1,
                )

    # Warmup trace
    logger.info("Capturing warmup trace")
    trace_id_warmup = ttnn.begin_trace_capture(submesh_device, cq_id=0)
    run_with_sync(15, sem_offset=0)
    ttnn.end_trace_capture(submesh_device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(submesh_device)

    # Capture main trace
    logger.info("Capturing main trace")
    trace_id = ttnn.begin_trace_capture(submesh_device, cq_id=0)
    run_with_sync(20, sem_offset=3)
    ttnn.end_trace_capture(submesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(submesh_device)

    # Capture tail trace
    logger.info("Capturing tail trace")
    trace_id_tail = ttnn.begin_trace_capture(submesh_device, cq_id=0)
    run_with_sync(20, sem_offset=7)
    ttnn.end_trace_capture(submesh_device, trace_id_tail, cq_id=0)
    ttnn.synchronize_device(submesh_device)

    # Execute warmup trace
    logger.info("Execute trace warmup")
    profiler.start("reduce-to-one-warmup")
    ttnn.execute_trace(submesh_device, trace_id_warmup, blocking=False)
    ttnn.release_trace(submesh_device, trace_id_warmup)
    ttnn.synchronize_device(submesh_device)
    profiler.end("reduce-to-one-warmup")

    # Execute main trace
    logger.info("Execute main trace")
    signpost("start")
    profiler.start("reduce-to-one-trace")
    ttnn.execute_trace(submesh_device, trace_id, blocking=False)
    ttnn.release_trace(submesh_device, trace_id)
    ttnn.synchronize_device(submesh_device)
    profiler.end("reduce-to-one-trace")
    signpost("stop")

    # Execute tail trace
    logger.info("Execute tail trace")
    profiler.start("reduce-to-one-tail")
    ttnn.execute_trace(submesh_device, trace_id_tail, blocking=False)
    ttnn.release_trace(submesh_device, trace_id_tail)
    ttnn.synchronize_device(submesh_device)
    profiler.end("reduce-to-one-tail")

    # Verify output
    print("\nVerifying trace output...")
    match = verify_output(output_tensor, submesh_device, root_coord, ref_output)

    assert match, "Output tensor does not match reference after trace execution"
    print("Trace test passed!")


# === Basic Tests ===
@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize(
    "device_params",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_1D})],
    indirect=["device_params"],
    ids=["fabric_1d"],
)
def test_reduce_to_one_1d(bh_2d_mesh_device):
    """Test reduce_to_one with 1D fabric."""
    run_reduce_to_one(bh_2d_mesh_device, ttnn.Topology.Linear)


# === Trace Tests ===
@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize(
    "device_params",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 425984})],
    indirect=["device_params"],
    ids=["fabric_1d_trace"],
)
def test_reduce_to_one_with_trace_1d(bh_2d_mesh_device):
    """Test reduce_to_one with trace capture/replay on 1D fabric."""
    run_reduce_to_one_with_trace(bh_2d_mesh_device, ttnn.Topology.Linear)
