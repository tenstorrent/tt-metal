# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for ReduceToRootB1 operation.

This test validates the 3-level reduction tree for a 4x2 mesh,
matching the configuration from test_deepseek_b1_reduce_to_one.py.
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import skip_for_wormhole_b0
from models.demos.deepseek_v3_b1.micro_ops.reduce_to_root_b1.op import ReduceToRootB1
from models.perf.benchmarking_utils import BenchmarkProfiler

# CoreRangeSet for CCL operations (subset of compute grid)
CCL_CRS = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 7))])


def compute_reference_reduce_to_one(data_per_device):
    """
    Compute the reference output for reduce_to_one operation.
    Simple sum of all device tensors.
    """
    result = data_per_device[0].clone()
    for i in range(1, len(data_per_device)):
        result = result + data_per_device[i]
    return result


def setup_reduce_to_root_test(mesh_device):
    """Common setup for reduce_to_root tests. Returns test configuration."""
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

    # Create output tensor (zeros, will be filled by reduce_to_root)
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
    ref_output = ReduceToRootB1.golden(data_per_device)

    # Create 4 semaphores for reduce_to_root (round1, round2, round3, exit)
    semaphores = [ttnn.create_global_semaphore(submesh_device, CCL_CRS, 0) for _ in range(4)]

    return {
        "submesh_device": submesh_device,
        "input_tensor": input_tensor,
        "intermediate_tensors": intermediate_tensors,
        "output_tensor": output_tensor,
        "ref_output": ref_output,
        "root_coord": root_coord,
        "exit_coord": exit_coord,
        "output_core": output_core,
        "semaphores": semaphores,
    }


def verify_output(output_tensor, submesh_device, root_coord, ref_output):
    """Verify output matches reference."""
    output_torch = ttnn.to_torch(output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=0))

    print(f"DEBUG: output_torch.shape = {output_torch.shape}")
    print(f"DEBUG: ref_output.shape = {ref_output.shape}")
    print(f"DEBUG: root_coord = {root_coord}")

    root_device_idx = root_coord[0] * submesh_device.shape[1] + root_coord[1]
    print(f"DEBUG: root_device_idx = {root_device_idx}")
    output_root = output_torch[root_device_idx]
    print(f"DEBUG: output_root.shape = {output_root.shape}")

    # Squeeze extra dimensions if needed
    output_root_squeezed = output_root  # .squeeze()
    ref_output_squeezed = ref_output  # .squeeze()
    print(f"DEBUG: output_root_squeezed.shape = {output_root_squeezed.shape}")
    print(f"DEBUG: ref_output_squeezed.shape = {ref_output_squeezed.shape}")

    # Print more values to see the pattern
    print(f"DEBUG: ref_output_squeezed[:32] = {ref_output_squeezed[:32]}")
    print(f"DEBUG: output_root_squeezed[:32] = {output_root_squeezed[:32]}")

    # Check non-zero count
    nonzero_count = (output_root_squeezed != 0).sum().item()
    print(f"DEBUG: nonzero_count in output = {nonzero_count} out of {output_root_squeezed.numel()}")

    rtol = 0.01
    atol = 0.05

    match = torch.allclose(output_root_squeezed, ref_output_squeezed, rtol=rtol, atol=atol)

    if not match:
        print(f"Output mismatch!")
        print(f"Reference:\n{ref_output_squeezed[:8]}")
        print(f"Output:\n{output_root_squeezed[:8]}")
        diff = torch.abs(output_root_squeezed - ref_output_squeezed)
        print(f"Max diff: {diff.max()}, Mean diff: {diff.mean()}")

        # Find where differences occur
        diff_mask = diff > atol
        diff_indices = torch.where(diff_mask)[0]
        if len(diff_indices) > 0:
            print(f"DEBUG: First 10 indices with large diff: {diff_indices[:10].tolist()}")

    return match


def run_reduce_to_root(mesh_device):
    """Run reduce_to_root test."""
    print(f"\n=== Testing reduce_to_root ===")

    config = setup_reduce_to_root_test(mesh_device)

    # Run reduce_to_root
    print("Running reduce_to_root...")
    output_tensor = ReduceToRootB1.op(
        config["input_tensor"],
        config["intermediate_tensors"],
        config["output_tensor"],
        config["semaphores"],
        ttnn.MeshCoordinate(config["root_coord"]),
        ttnn.MeshCoordinate(config["exit_coord"]),
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


def run_reduce_to_root_with_trace(mesh_device):
    """Run reduce_to_root test with trace capture and replay."""
    print(f"\n=== Testing reduce_to_root with trace ===")

    config = setup_reduce_to_root_test(mesh_device)
    submesh_device = config["submesh_device"]
    input_tensor = config["input_tensor"]
    intermediate_tensors = config["intermediate_tensors"]
    output_tensor_preallocated = config["output_tensor"]
    root_coord = config["root_coord"]
    exit_coord = config["exit_coord"]
    ref_output = config["ref_output"]
    semaphores = config["semaphores"]

    # Run once to compile
    print("Running reduce_to_root (compiling)...")
    output_tensor = ReduceToRootB1.op(
        input_tensor,
        intermediate_tensors,
        output_tensor_preallocated,
        semaphores,
        ttnn.MeshCoordinate(root_coord),
        ttnn.MeshCoordinate(exit_coord),
    )
    ttnn.synchronize_device(submesh_device)

    # Helper to run reduce_to_root multiple iterations
    profiler = BenchmarkProfiler()

    def run_iterations(num_iters):
        for _ in range(num_iters):
            output_tensor = ReduceToRootB1.op(
                input_tensor,
                intermediate_tensors,
                output_tensor_preallocated,
                semaphores,
                ttnn.MeshCoordinate(root_coord),
                ttnn.MeshCoordinate(exit_coord),
            )

    # Capture warmup trace
    logger.info("Capturing warmup trace")
    trace_id_warmup = ttnn.begin_trace_capture(submesh_device, cq_id=0)
    run_iterations(15)
    ttnn.end_trace_capture(submesh_device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(submesh_device)

    # Capture main trace
    logger.info("Capturing main trace")
    trace_id = ttnn.begin_trace_capture(submesh_device, cq_id=0)
    run_iterations(30)
    ttnn.end_trace_capture(submesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(submesh_device)

    # Execute warmup trace
    logger.info("Execute trace warmup")
    profiler.start("warmup-trace")
    ttnn.execute_trace(submesh_device, trace_id_warmup, blocking=False)
    ttnn.release_trace(submesh_device, trace_id_warmup)
    ttnn.synchronize_device(submesh_device)
    profiler.end("warmup-trace")

    # Execute main trace
    logger.info("Execute main trace")
    signpost("start")
    profiler.start("main-trace")
    ttnn.execute_trace(submesh_device, trace_id, blocking=False)
    ttnn.release_trace(submesh_device, trace_id)
    ttnn.synchronize_device(submesh_device)

    profiler.end("main-trace")
    signpost("stop")

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
def test_reduce_to_root_1d(bh_2d_mesh_device):
    """Test reduce_to_root with 1D fabric."""
    run_reduce_to_root(bh_2d_mesh_device)


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize(
    "device_params",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_2D})],
    indirect=["device_params"],
    ids=["fabric_2d"],
)
def test_reduce_to_root_2d(bh_2d_mesh_device):
    """Test reduce_to_root with 2D fabric."""
    run_reduce_to_root(bh_2d_mesh_device)


# === Trace Tests ===
@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize(
    "device_params",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 425984})],
    indirect=["device_params"],
    ids=["fabric_1d_trace"],
)
def test_reduce_to_root_with_trace_1d(bh_2d_mesh_device):
    """Test reduce_to_root with trace capture/replay on 1D fabric."""
    run_reduce_to_root_with_trace(bh_2d_mesh_device)


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize(
    "device_params",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_2D, "trace_region_size": 425984})],
    indirect=["device_params"],
    ids=["fabric_2d_trace"],
)
def test_reduce_to_root_with_trace_2d(bh_2d_mesh_device):
    """Test reduce_to_root with trace capture/replay on 2D fabric."""
    run_reduce_to_root_with_trace(bh_2d_mesh_device)
