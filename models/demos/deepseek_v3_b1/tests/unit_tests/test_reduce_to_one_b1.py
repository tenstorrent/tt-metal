# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for ReduceToOneB1 operation.

This test validates the 3-level reduction tree for a 4x2 mesh
"""

from typing import Any

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import is_slow_dispatch, skip_for_wormhole_b0
from models.demos.deepseek_v3_b1.micro_ops.reduce_to_one_b1.op import ReduceToOneB1
from models.demos.deepseek_v3_b1.model_dimensions import LogicalModelDimensions, RoutedExpert
from models.demos.deepseek_v3_b1.tests.unit_tests.ccl_test_utils import (
    capture_trace,
    compile_trace_op,
    create_fabric_router_config,
    get_env_int,
)
from models.demos.deepseek_v3_b1.utils import get_pinned_optimal_dram_bank_to_logical_worker_assignment
from models.perf.benchmarking_utils import BenchmarkProfiler

ENV_MAX_PAYLOAD_SIZE = "CCL_REDUCE_TO_ONE_MAX_PAYLOAD_SIZE_BYTES"
TRACE_MAX_PAYLOAD_SIZE = get_env_int(ENV_MAX_PAYLOAD_SIZE, 15232)
ENV_TRACE_NUM_WARMUP_SAMPLES = "CCL_REDUCE_TO_ONE_TRACE_NUM_WARMUP_SAMPLES"
ENV_TRACE_NUM_PERF_SAMPLES = "CCL_REDUCE_TO_ONE_TRACE_NUM_PERF_SAMPLES"
TRACE_REPLAY_BLOCKING = True
ROTATED_ROOT_COORDS = ((1, 0), (1, 1), (2, 0), (2, 1))

NUM_WORKERS = RoutedExpert.NUM_CORES
VALID_TOTAL_WIDTH = LogicalModelDimensions.HIDDEN_SIZE
PADDED_WIDTH_PER_CORE = RoutedExpert.FINAL_OUTPUT_WIDTH_PER_CORE
VALID_WIDTH_PER_CORE = VALID_TOTAL_WIDTH // NUM_WORKERS
PADDED_TOTAL_WIDTH = PADDED_WIDTH_PER_CORE * NUM_WORKERS


def _build_reduce_to_one_op(config: dict, *, root_coord: tuple[int, int], exit_coord: tuple[int, int], is_torus: bool):
    input_tensor = config["input_tensor"]
    intermediate_tensor = config["intermediate_tensor"]
    output_tensor_preallocated = config["output_tensor"]
    semaphores = config["semaphores"]
    worker_fabric_ready_semaphore = config["worker_fabric_ready_semaphore"]

    def run_reduce_to_one_op():
        return ReduceToOneB1.op(
            input_tensor,
            intermediate_tensor,
            output_tensor_preallocated,
            semaphores,
            ttnn.MeshCoordinate(root_coord),
            exit_coord=ttnn.MeshCoordinate(exit_coord),
            num_iterations=1,
            is_torus=is_torus,
            worker_fabric_ready_semaphore=worker_fabric_ready_semaphore,
        )

    return run_reduce_to_one_op


def _capture_rotated_trace_sets(
    mesh_device,
    ops_by_root: dict[tuple[int, int], Any],
    *,
    phase_name: str,
    num_samples: int,
) -> tuple[list[dict[tuple[int, int], Any]], Any]:
    roots = list(ROTATED_ROOT_COORDS)
    trace_sets: list[dict[tuple[int, int], Any]] = []
    output_tensor = None
    total_traces = len(roots) * num_samples
    progress_step = 1 if num_samples <= 3 else max(1, num_samples // 3)
    logger.info(
        "Capturing {} trace sets: roots={}, samples_per_root={}, total_traces={}",
        phase_name,
        roots,
        num_samples,
        total_traces,
    )
    for sample_idx in range(num_samples):
        trace_ids_by_root = {}
        for root_coord in roots:
            trace_id, output_tensor = capture_trace(mesh_device, ops_by_root[root_coord], trace_body_iters=1)
            trace_ids_by_root[root_coord] = trace_id
        trace_sets.append(trace_ids_by_root)
        completed_samples = sample_idx + 1
        if completed_samples % progress_step == 0 or completed_samples == num_samples:
            logger.info("Captured {} trace sets: {}/{}", phase_name, completed_samples, num_samples)
    return trace_sets, output_tensor


def _execute_rotated_interleaved_fresh_trace_schedule(
    mesh_device,
    trace_sets_by_sample: list[dict[tuple[int, int], Any]],
) -> None:
    roots = list(ROTATED_ROOT_COORDS)
    num_roots = len(roots)
    for sample_idx, trace_ids_by_root in enumerate(trace_sets_by_sample):
        offset = sample_idx % num_roots
        ordered_roots = roots[offset:] + roots[:offset]
        for root_coord in ordered_roots:
            trace_id = trace_ids_by_root[root_coord]
            ttnn.execute_trace(mesh_device, trace_id, blocking=TRACE_REPLAY_BLOCKING)
            ttnn.release_trace(mesh_device, trace_id)


def _pad_reduce_shards(valid_tensor: torch.Tensor) -> torch.Tensor:
    """Pack 8 valid 896-wide shards into 8 padded 1024-wide reduce shards."""
    padded = torch.zeros((valid_tensor.shape[0], PADDED_TOTAL_WIDTH), dtype=valid_tensor.dtype)
    for shard_idx in range(NUM_WORKERS):
        valid_start = shard_idx * VALID_WIDTH_PER_CORE
        valid_end = valid_start + VALID_WIDTH_PER_CORE
        padded_start = shard_idx * PADDED_WIDTH_PER_CORE
        padded[:, padded_start : padded_start + VALID_WIDTH_PER_CORE] = valid_tensor[:, valid_start:valid_end]
    return padded


def setup_reduce_to_one_test(mesh_device, root_coord, exit_coord):
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

    submesh_device = mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    logger.info(f"Created submesh with shape: {submesh_device.shape}")

    assert submesh_device.shape == ttnn.MeshShape((4, 2)), f"Expected 4x2 mesh, got {submesh_device.shape}"

    # Use the padded 8-shard reduce layout that matches the decoder/MoE reduce buffers.
    tensor_shape = [1, PADDED_TOTAL_WIDTH]
    valid_tensor_shape = [1, VALID_TOTAL_WIDTH]
    dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT
    tile = ttnn.Tile((1, 32))

    # Use the pinned decoder/MoE worker layout.
    compute_cores = get_pinned_optimal_dram_bank_to_logical_worker_assignment(submesh_device, ttnn.NOC.NOC_0)
    num_cores = len(compute_cores)
    logger.info(f"Using {num_cores} optimal DRAM cores: {compute_cores[:8]}")

    # Build shard grid from optimal cores (use first 8 cores)
    num_shard_cores = NUM_WORKERS
    shard_cores = compute_cores[:num_shard_cores]
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(core, core) for core in shard_cores})

    shard_width = PADDED_WIDTH_PER_CORE
    shard_shape = [1, shard_width]
    shard_spec = ttnn.ShardSpec(
        shard_grid,
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    mem_config = ttnn.MemoryConfig(ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec)

    # Mesh mapper
    mesh_mapper_config = ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)], submesh_device.shape)
    mesh_mapper = ttnn.create_mesh_mapper(submesh_device, mesh_mapper_config)

    # Single intermediate tensor with 3× shard width (one 3-page CB for all reduction rounds)
    intermediate_shard_shape = [1, shard_shape[1] * 3]
    intermediate_tensor_shape = [1, tensor_shape[1] * 3]
    intermediate_shard_spec = ttnn.ShardSpec(
        shard_grid,
        intermediate_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    intermediate_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, intermediate_shard_spec
    )
    intermediate_data = torch.zeros([4, 2] + intermediate_tensor_shape, dtype=torch.bfloat16)
    intermediate_tensor = ttnn.from_torch(
        intermediate_data,
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=intermediate_mem_config,
        mesh_mapper=mesh_mapper,
    )

    # Output tensor sharded on the first worker core (aggregator core)
    shard_cores_list = ttnn.corerange_to_cores(shard_grid, row_wise=True)
    aggregator_core = shard_cores_list[0]
    logger.info(f"Aggregator core (first worker core): {aggregator_core}")

    output_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(aggregator_core, aggregator_core)})
    output_shard_shape = tensor_shape
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
    logger.info(f"Created output tensor sharded on aggregator core: {aggregator_core}")

    logger.info(
        "Using padded reduce geometry: valid_width_per_core={}, padded_width_per_core={}, total_width={}",
        VALID_WIDTH_PER_CORE,
        PADDED_WIDTH_PER_CORE,
        PADDED_TOTAL_WIDTH,
    )

    data_per_device = []
    torch.manual_seed(42)
    for _ in range(num_devices):
        valid_data = torch.randn(valid_tensor_shape, dtype=torch.bfloat16)
        data = _pad_reduce_shards(valid_data)
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
    ref_output = ReduceToOneB1.golden(data_per_device)

    # Create 4 reduction semaphores (round1, round2, round3, exit) plus one shared
    # worker->fabric ready semaphore for the standalone forwarder ready-mask path.
    compute_grid = submesh_device.compute_with_storage_grid_size()
    num_cores = compute_grid.x * compute_grid.y
    available_cores = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, row_wise=True)
    ttnn.synchronize_device(submesh_device)
    semaphores = [ttnn.create_global_semaphore(submesh_device, available_cores, 0) for _ in range(4)]
    worker_fabric_ready_semaphore = ttnn.create_global_semaphore(submesh_device, available_cores, 0)
    ttnn.synchronize_device(submesh_device)
    logger.info("Created shared worker->fabric ready semaphore")

    return {
        "submesh_device": submesh_device,
        "input_tensor": input_tensor,
        "intermediate_tensor": intermediate_tensor,
        "output_tensor": output_tensor,
        "ref_output": ref_output,
        "root_coord": root_coord,
        "exit_coord": exit_coord,
        "output_core": aggregator_core,
        "semaphores": semaphores,
        "worker_fabric_ready_semaphore": worker_fabric_ready_semaphore,
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


def run_reduce_to_one(mesh_device, root_coord=(1, 1), exit_coord=(0, 1), is_torus=False):
    """Run single-epoch reduce_to_one correctness test."""
    print("\n=== Testing reduce_to_one ===")

    config = setup_reduce_to_one_test(mesh_device, root_coord, exit_coord)

    # Keep standalone correctness coverage single-epoch only. The current
    # in-kernel loop reuses round semaphores, the shared worker->fabric ready
    # semaphore, `received_cb`, and FC packet staging without an explicit epoch
    # protocol, so num_loop_iters>1 is not a safe stale-state/epoch-safety test.
    # A single epoch still exercises the intended cleanup path: semaphores are
    # consumed/reset and the active CB pages are popped before return.
    print("Running single-epoch reduce_to_one...")
    output_tensor = ReduceToOneB1.op(
        config["input_tensor"],
        config["intermediate_tensor"],
        config["output_tensor"],
        config["semaphores"],
        ttnn.MeshCoordinate(config["root_coord"]),
        exit_coord=ttnn.MeshCoordinate(config["exit_coord"]),
        num_iterations=1,
        is_torus=is_torus,
        worker_fabric_ready_semaphore=config["worker_fabric_ready_semaphore"],
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


def run_reduce_to_one_with_trace(
    mesh_device,
    *,
    root_coord=(1, 1),
    exit_coord=(0, 1),
    is_torus=False,
    num_warmup_iter=15,
    num_iter=30,
):
    """Run reduce_to_one with the fixed rotated-fresh blocking benchmark strategy."""
    print(f"\n=== Testing reduce_to_one with trace ===")

    num_warmup_samples = get_env_int(ENV_TRACE_NUM_WARMUP_SAMPLES, num_warmup_iter)
    num_perf_samples = get_env_int(ENV_TRACE_NUM_PERF_SAMPLES, num_iter)

    config = setup_reduce_to_one_test(mesh_device, root_coord, exit_coord)
    submesh_device = config["submesh_device"]
    root_coords_to_verify = list(ROTATED_ROOT_COORDS)

    logger.info(
        "Running reduce_to_one trace benchmark: max_payload_size_bytes={}, roots={}, "
        "warmup_samples_per_root={}, perf_samples_per_root={}, replay_blocking={}",
        TRACE_MAX_PAYLOAD_SIZE,
        root_coords_to_verify,
        num_warmup_samples,
        num_perf_samples,
        TRACE_REPLAY_BLOCKING,
    )

    ops_by_root = {
        root: _build_reduce_to_one_op(config, root_coord=root, exit_coord=root, is_torus=is_torus)
        for root in root_coords_to_verify
    }

    output_tensor = None
    logger.info("Compiling reduce_to_one trace variants for roots={}", root_coords_to_verify)
    for root in root_coords_to_verify:
        compile_trace_op(submesh_device, ops_by_root[root])

    warmup_trace_sets, output_tensor = _capture_rotated_trace_sets(
        submesh_device,
        ops_by_root,
        phase_name="warmup",
        num_samples=num_warmup_samples,
    )
    perf_trace_sets, output_tensor = _capture_rotated_trace_sets(
        submesh_device,
        ops_by_root,
        phase_name="perf",
        num_samples=num_perf_samples,
    )

    profiler = BenchmarkProfiler()
    logger.info(
        "Replaying warmup trace sets: samples_per_root={}, total_replays={}, replay_blocking={}",
        num_warmup_samples,
        len(root_coords_to_verify) * num_warmup_samples,
        TRACE_REPLAY_BLOCKING,
    )
    profiler.start("deepseek-reduce-to-one-warmup")
    _execute_rotated_interleaved_fresh_trace_schedule(
        submesh_device,
        warmup_trace_sets,
    )
    ttnn.synchronize_device(submesh_device)
    profiler.end("deepseek-reduce-to-one-warmup")

    logger.info(
        "Replaying perf trace sets: samples_per_root={}, total_replays={}, replay_blocking={}",
        num_perf_samples,
        len(root_coords_to_verify) * num_perf_samples,
        TRACE_REPLAY_BLOCKING,
    )
    signpost("start")
    try:
        profiler.start("deepseek-reduce-to-one-trace")
        _execute_rotated_interleaved_fresh_trace_schedule(
            submesh_device,
            perf_trace_sets,
        )
        ttnn.synchronize_device(submesh_device)
        profiler.end("deepseek-reduce-to-one-trace")
    finally:
        signpost("stop")

    config["root_coords_to_verify"] = root_coords_to_verify
    return output_tensor or config["output_tensor"], config


# === Basic Tests ===
@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize(
    "device_params",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "fabric_router_config": create_fabric_router_config(15232)})],
    indirect=["device_params"],
    ids=["fabric_1d"],
)
def test_reduce_to_one_1d(bh_2d_mesh_device):
    """Test reduce_to_one with 1D fabric."""
    run_reduce_to_one(bh_2d_mesh_device)


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize(
    "device_params",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_2D, "fabric_router_config": create_fabric_router_config(15232)})],
    indirect=["device_params"],
    ids=["fabric_2d"],
)
def test_reduce_to_one_2d(bh_2d_mesh_device):
    """Test reduce_to_one with 2D fabric."""
    run_reduce_to_one(bh_2d_mesh_device)


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize("num_iter, num_warmup_iter", [(30, 15)])
@pytest.mark.parametrize(
    "device_params",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_2D,
                "fabric_router_config": create_fabric_router_config(TRACE_MAX_PAYLOAD_SIZE),
                "trace_region_size": 1048576,
            }
        )
    ],
    indirect=["device_params"],
    ids=["fabric_2d_trace"],
)
def test_reduce_to_one_trace(
    bh_2d_mesh_device,
    num_warmup_iter,
    num_iter,
):
    """Trace benchmark for standalone reduce_to_one using the padded decoder/MoE geometry."""
    if is_slow_dispatch():
        pytest.skip("reduce_to_one trace test needs fast dispatch")

    output_tensor, config = run_reduce_to_one_with_trace(
        bh_2d_mesh_device,
        num_warmup_iter=num_warmup_iter,
        num_iter=num_iter,
    )

    print("\nVerifying trace output...")
    for root_coord in config.get("root_coords_to_verify", [config["root_coord"]]):
        match = verify_output(
            output_tensor,
            config["submesh_device"],
            root_coord,
            config["ref_output"],
        )
        assert match, f"Output tensor does not match reference after trace execution for root_coord={root_coord}"
    print("Trace test passed!")
