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

from dataclasses import dataclass
from typing import Any, Optional

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.ccl_all_reduce.op import DeepseekMinimalAllReduce

# Tensor shard placement for dual-core all-reduce tests (op derives sender/receiver from these grids).
DUAL_CORE_TEST_SENDER = ttnn.CoreCoord(10, 9)
DUAL_CORE_TEST_RECEIVER = ttnn.CoreCoord(11, 9)
from models.perf.benchmarking_utils import BenchmarkProfiler


def create_fabric_router_config(max_payload_size):
    """Helper to create FabricRouterConfig with custom max payload size."""
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


@dataclass(frozen=True)
class AllReduceTestInputs:
    input_tensors_per_device: list[torch.Tensor]
    torch_expected: torch.Tensor
    input_tensor_mesh: Any
    intermediate_tensor_mesh: Any
    output_tensor_mesh: Any
    residual_tensor_mesh: Optional[Any]
    semaphores: list[Any]


def build_all_reduce_test_inputs(
    *,
    mesh_device,
    num_devices,
    output_shape,
    input_shard_shape,
    tensor_mem_layout,
    layout,
    input_dtype,
    fuse_residual_add,
    semaphore_count=2,
):
    # Set up sharded memory config
    input_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    input_shard_spec = ttnn.ShardSpec(
        input_shard_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=input_shard_spec)

    # Create input tensors for each device
    input_tensors_per_device = [torch.rand(output_shape, dtype=torch.bfloat16) for _ in range(num_devices)]

    mesh_mapper_config = ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementReplicate()], mesh_device.shape)

    input_tensor_mesh = ttnn.from_torch(
        torch.cat(input_tensors_per_device, dim=0),
        device=mesh_device,
        layout=layout,
        tile=ttnn.Tile((1, 32)),
        dtype=input_dtype,
        memory_config=input_mem_config,
        mesh_mapper=ttnn.create_mesh_mapper(mesh_device, mesh_mapper_config),
    )

    residual_tensor_mesh = None
    residual_tensor_torch = None
    if fuse_residual_add:
        residual_tensor_torch = torch.rand(output_shape, dtype=torch.bfloat16)
        residual_tensor_mesh = ttnn.from_torch(
            torch.cat([residual_tensor_torch for _ in range(num_devices)], dim=0),
            device=mesh_device,
            layout=layout,
            tile=ttnn.Tile((1, 32)),
            dtype=input_dtype,
            memory_config=input_mem_config,
            mesh_mapper=ttnn.create_mesh_mapper(mesh_device, mesh_mapper_config),
        )

    # Output tensor with tiny tiles (1x32), same as input
    output_tensor_mesh = ttnn.from_torch(
        torch.cat([torch.zeros(output_shape, dtype=torch.bfloat16)] * num_devices, dim=0),
        device=mesh_device,
        layout=layout,
        tile=ttnn.Tile((1, 32)),
        dtype=input_dtype,
        memory_config=input_mem_config,
        mesh_mapper=ttnn.create_mesh_mapper(mesh_device, mesh_mapper_config),
    )

    # Intermediate tensor with standard 32x32 tiles.
    intermediate_shape = [32, 224]
    intermediate_shard_spec = ttnn.ShardSpec(
        input_shard_grid,
        tuple(intermediate_shape),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    intermediate_mem_config = ttnn.MemoryConfig(
        tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=intermediate_shard_spec
    )
    intermediate_tensor_mesh = ttnn.from_torch(
        torch.cat([torch.zeros(intermediate_shape, dtype=torch.bfloat16)] * num_devices, dim=0),
        device=mesh_device,
        layout=layout,
        tile=ttnn.Tile((32, 32)),
        dtype=input_dtype,
        memory_config=intermediate_mem_config,
        mesh_mapper=ttnn.create_mesh_mapper(mesh_device, mesh_mapper_config),
    )

    # Golden output
    if fuse_residual_add:
        torch_expected = DeepseekMinimalAllReduce.golden(input_tensors_per_device, residual_tensor_torch)
    else:
        torch_expected = DeepseekMinimalAllReduce.golden(input_tensors_per_device)

    # Semaphores
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    num_cores = compute_grid_size.x * compute_grid_size.y
    available_cores = ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, row_wise=True)
    semaphores = [ttnn.create_global_semaphore(mesh_device, available_cores, 0) for _ in range(semaphore_count)]

    return AllReduceTestInputs(
        input_tensors_per_device=input_tensors_per_device,
        torch_expected=torch_expected,
        input_tensor_mesh=input_tensor_mesh,
        intermediate_tensor_mesh=intermediate_tensor_mesh,
        output_tensor_mesh=output_tensor_mesh,
        residual_tensor_mesh=residual_tensor_mesh,
        semaphores=semaphores,
    )


def build_dual_core_all_reduce_test_inputs(
    *,
    mesh_device,
    num_devices,
    output_shape,
    input_shard_shape,
    tensor_mem_layout,
    layout,
    input_dtype,
    fuse_residual_add,
    semaphore_count=3,
):
    """Same tensor geometry as build_all_reduce_test_inputs, but input on sender core and recv tensors on receiver core.

    ``semaphore_count`` defaults to 3 = two fabric links + ``local_ready`` (see ``get_num_semaphores_dual_core``).
    """
    sender_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(DUAL_CORE_TEST_SENDER, DUAL_CORE_TEST_SENDER)})
    receiver_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(DUAL_CORE_TEST_RECEIVER, DUAL_CORE_TEST_RECEIVER)})
    input_shard_spec = ttnn.ShardSpec(
        sender_shard_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=input_shard_spec)

    input_tensors_per_device = [torch.rand(output_shape, dtype=torch.bfloat16) for _ in range(num_devices)]

    mesh_mapper_config = ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementReplicate()], mesh_device.shape)

    input_tensor_mesh = ttnn.from_torch(
        torch.cat(input_tensors_per_device, dim=0),
        device=mesh_device,
        layout=layout,
        tile=ttnn.Tile((1, 32)),
        dtype=input_dtype,
        memory_config=input_mem_config,
        mesh_mapper=ttnn.create_mesh_mapper(mesh_device, mesh_mapper_config),
    )

    residual_tensor_mesh = None
    residual_tensor_torch = None
    if fuse_residual_add:
        residual_tensor_torch = torch.rand(output_shape, dtype=torch.bfloat16)
        residual_shard_spec = ttnn.ShardSpec(
            receiver_shard_grid,
            input_shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        residual_mem_config = ttnn.MemoryConfig(
            tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=residual_shard_spec
        )
        residual_tensor_mesh = ttnn.from_torch(
            torch.cat([residual_tensor_torch for _ in range(num_devices)], dim=0),
            device=mesh_device,
            layout=layout,
            tile=ttnn.Tile((1, 32)),
            dtype=input_dtype,
            memory_config=residual_mem_config,
            mesh_mapper=ttnn.create_mesh_mapper(mesh_device, mesh_mapper_config),
        )

    output_shard_spec = ttnn.ShardSpec(
        receiver_shard_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(
        tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=output_shard_spec
    )
    output_tensor_mesh = ttnn.from_torch(
        torch.cat([torch.zeros(output_shape, dtype=torch.bfloat16)] * num_devices, dim=0),
        device=mesh_device,
        layout=layout,
        tile=ttnn.Tile((1, 32)),
        dtype=input_dtype,
        memory_config=output_mem_config,
        mesh_mapper=ttnn.create_mesh_mapper(mesh_device, mesh_mapper_config),
    )

    intermediate_shape = [32, 224]
    intermediate_shard_spec = ttnn.ShardSpec(
        receiver_shard_grid,
        tuple(intermediate_shape),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    intermediate_mem_config = ttnn.MemoryConfig(
        tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=intermediate_shard_spec
    )
    intermediate_tensor_mesh = ttnn.from_torch(
        torch.cat([torch.zeros(intermediate_shape, dtype=torch.bfloat16)] * num_devices, dim=0),
        device=mesh_device,
        layout=layout,
        tile=ttnn.Tile((32, 32)),
        dtype=input_dtype,
        memory_config=intermediate_mem_config,
        mesh_mapper=ttnn.create_mesh_mapper(mesh_device, mesh_mapper_config),
    )

    if fuse_residual_add:
        torch_expected = DeepseekMinimalAllReduce.golden(input_tensors_per_device, residual_tensor_torch)
    else:
        torch_expected = DeepseekMinimalAllReduce.golden(input_tensors_per_device)

    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    num_cores = compute_grid_size.x * compute_grid_size.y
    available_cores = ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, row_wise=True)
    semaphores = [ttnn.create_global_semaphore(mesh_device, available_cores, 0) for _ in range(semaphore_count)]

    return AllReduceTestInputs(
        input_tensors_per_device=input_tensors_per_device,
        torch_expected=torch_expected,
        input_tensor_mesh=input_tensor_mesh,
        intermediate_tensor_mesh=intermediate_tensor_mesh,
        output_tensor_mesh=output_tensor_mesh,
        residual_tensor_mesh=residual_tensor_mesh,
        semaphores=semaphores,
    )


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
@pytest.mark.parametrize("num_links", [1])
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
    bh_2d_mesh_device,
    num_devices,
    output_shape,
    input_shard_shape,
    tensor_mem_layout,
    layout,
    input_dtype,
    cluster_axis,
    fuse_residual_add,
    num_warmup_iter,
    num_iter,
    num_links,
):
    # Validate mesh size
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    # Create submesh - fabric requires opening full system mesh first
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))

    inputs = build_all_reduce_test_inputs(
        mesh_device=submesh,
        num_devices=num_devices,
        output_shape=output_shape,
        input_shard_shape=input_shard_shape,
        tensor_mem_layout=tensor_mem_layout,
        layout=layout,
        input_dtype=input_dtype,
        fuse_residual_add=fuse_residual_add,
        semaphore_count=2,
    )

    # Run all-reduce operation
    logger.info(f"Running CCL all-reduce: num_devices={num_devices}")

    profiler = BenchmarkProfiler()

    # Compile Run
    logger.info("Compiling model")
    ttnn_result = DeepseekMinimalAllReduce.op(
        inputs.input_tensor_mesh,
        inputs.intermediate_tensor_mesh,
        cluster_axis=cluster_axis,
        persistent_output_tensor=inputs.output_tensor_mesh,
        residual_tensor_mesh=inputs.residual_tensor_mesh,
        semaphores=inputs.semaphores[:num_links],
        num_links=num_links,
    )
    ttnn.synchronize_device(submesh)

    # Capture warmup trace
    logger.info("Capturing warmup trace")
    trace_id_warmup = ttnn.begin_trace_capture(submesh, cq_id=0)
    for i in range(num_warmup_iter):
        ttnn_result = DeepseekMinimalAllReduce.op(
            inputs.input_tensor_mesh,
            inputs.intermediate_tensor_mesh,
            cluster_axis=cluster_axis,
            persistent_output_tensor=inputs.output_tensor_mesh,
            residual_tensor_mesh=inputs.residual_tensor_mesh,
            semaphores=inputs.semaphores[:num_links],
            num_links=num_links,
        )
    ttnn.end_trace_capture(submesh, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(submesh)

    # Capture main trace
    logger.info("Capturing trace")
    trace_id = ttnn.begin_trace_capture(submesh, cq_id=0)
    for i in range(num_iter):
        ttnn_result = DeepseekMinimalAllReduce.op(
            inputs.input_tensor_mesh,
            inputs.intermediate_tensor_mesh,
            cluster_axis=cluster_axis,
            persistent_output_tensor=inputs.output_tensor_mesh,
            residual_tensor_mesh=inputs.residual_tensor_mesh,
            semaphores=inputs.semaphores[:num_links],
            num_links=num_links,
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
    ref_device_idx = 0
    ref_device_output = output_tensor_torch[ref_device_idx : ref_device_idx + 1, :]
    for device_idx in range(num_devices):
        received = output_tensor_torch[device_idx : device_idx + 1, :]

        assert received.shape == inputs.torch_expected.shape, f"Shape mismatch at device {device_idx}"

        if device_idx != ref_device_idx:
            dev_eq = torch.equal(received, ref_device_output)
            assert dev_eq, f"Device {device_idx} output mismatch"

        if not torch.allclose(received, inputs.torch_expected, rtol=1e-2, atol=1e-2):
            logger.error(f"Output mismatch for device {device_idx}")
            logger.error(f"Expected: {inputs.torch_expected[:5, :5]}")
            logger.error(f"Received: {received[:5, :5]}")
            all_passed = False
        else:
            logger.info(f"Device {device_idx}: PASSED")

    assert all_passed, "Not all devices have the correct all-reduced data"
    logger.info("CCL all-reduce test passed!")


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
@pytest.mark.parametrize("num_links", [1, 2])
@pytest.mark.parametrize("chunk_num_tiles", [1, 2, 10])
@pytest.mark.parametrize("fuse_residual_add", [True, False])
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
def test_ccl_all_reduce_chunk_and_link_matrix(
    bh_2d_mesh_device,
    num_devices,
    output_shape,
    input_shard_shape,
    tensor_mem_layout,
    layout,
    input_dtype,
    cluster_axis,
    num_links,
    chunk_num_tiles,
    fuse_residual_add,
):
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    inputs = build_all_reduce_test_inputs(
        mesh_device=submesh,
        num_devices=num_devices,
        output_shape=output_shape,
        input_shard_shape=input_shard_shape,
        tensor_mem_layout=tensor_mem_layout,
        layout=layout,
        input_dtype=input_dtype,
        fuse_residual_add=fuse_residual_add,
        semaphore_count=2,
    )

    logger.info(
        f"Running all-reduce feature matrix: num_links={num_links}, chunk_num_tiles={chunk_num_tiles}, "
        f"fuse_residual_add={fuse_residual_add}"
    )
    ttnn_result = DeepseekMinimalAllReduce.op(
        inputs.input_tensor_mesh,
        inputs.intermediate_tensor_mesh,
        semaphores=inputs.semaphores[:num_links],
        cluster_axis=cluster_axis,
        num_links=num_links,
        chunk_num_tiles=chunk_num_tiles,
        persistent_output_tensor=inputs.output_tensor_mesh,
        residual_tensor_mesh=inputs.residual_tensor_mesh,
    )
    ttnn.synchronize_device(submesh)

    output_tensor_torch = ttnn.to_torch(
        ttnn_result,
        mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0),
    )

    ref_device_idx = 0
    ref_device_output = output_tensor_torch[ref_device_idx : ref_device_idx + 1, :]
    for device_idx in range(num_devices):
        received = output_tensor_torch[device_idx : device_idx + 1, :]
        assert received.shape == inputs.torch_expected.shape, f"Shape mismatch at device {device_idx}"
        if device_idx != ref_device_idx:
            assert torch.equal(received, ref_device_output), f"Device {device_idx} output mismatch"
        assert torch.allclose(received, inputs.torch_expected, rtol=1e-2, atol=1e-2), (
            f"Output mismatch for device {device_idx}; "
            f"num_links={num_links}, chunk_num_tiles={chunk_num_tiles}, fuse_residual_add={fuse_residual_add}"
        )


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
@pytest.mark.parametrize("num_links", [2])
@pytest.mark.parametrize("chunk_num_tiles", [1, 2, 10])
@pytest.mark.parametrize("fuse_residual_add", [False, True])
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
def test_ccl_all_reduce_dual_core_sender(
    bh_2d_mesh_device,
    num_devices,
    output_shape,
    input_shard_shape,
    tensor_mem_layout,
    layout,
    input_dtype,
    cluster_axis,
    num_links,
    chunk_num_tiles,
    fuse_residual_add,
):
    """Standalone dual-core sender path (§11.7): NCRISC+BRISC fabric links vs single-core control."""
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    inputs = build_dual_core_all_reduce_test_inputs(
        mesh_device=submesh,
        num_devices=num_devices,
        output_shape=output_shape,
        input_shard_shape=input_shard_shape,
        tensor_mem_layout=tensor_mem_layout,
        layout=layout,
        input_dtype=input_dtype,
        fuse_residual_add=fuse_residual_add,
    )

    ttnn_result = DeepseekMinimalAllReduce.op(
        inputs.input_tensor_mesh,
        inputs.intermediate_tensor_mesh,
        semaphores=inputs.semaphores[: num_links + 1],
        cluster_axis=cluster_axis,
        num_links=num_links,
        chunk_num_tiles=chunk_num_tiles,
        persistent_output_tensor=inputs.output_tensor_mesh,
        residual_tensor_mesh=inputs.residual_tensor_mesh,
        dual_core_sender=True,
    )
    ttnn.synchronize_device(submesh)

    output_tensor_torch = ttnn.to_torch(
        ttnn_result,
        mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0),
    )

    ref_device_idx = 0
    ref_device_output = output_tensor_torch[ref_device_idx : ref_device_idx + 1, :]
    for device_idx in range(num_devices):
        received = output_tensor_torch[device_idx : device_idx + 1, :]
        assert received.shape == inputs.torch_expected.shape, f"Shape mismatch at device {device_idx}"
        if device_idx != ref_device_idx:
            assert torch.equal(received, ref_device_output), f"Device {device_idx} output mismatch"
        assert torch.allclose(received, inputs.torch_expected, rtol=1e-2, atol=1e-2), (
            f"Dual-core output mismatch for device {device_idx}; "
            f"num_links={num_links}, chunk_num_tiles={chunk_num_tiles}, fuse_residual_add={fuse_residual_add}"
        )


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
@pytest.mark.parametrize("num_links", [2])
@pytest.mark.parametrize("num_iter, num_warmup_iter", [(50, 20)])
@pytest.mark.parametrize("fuse_residual_add", [True])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(4096),
            "trace_region_size": 573440,
        }
    ],
    indirect=True,
)
def test_ccl_all_reduce_dual_core_sender_trace(
    bh_2d_mesh_device,
    num_devices,
    output_shape,
    input_shard_shape,
    tensor_mem_layout,
    layout,
    input_dtype,
    cluster_axis,
    num_iter,
    num_warmup_iter,
    num_links,
    fuse_residual_add,
):
    """Dual-core trace benchmark path mirroring test_ccl_all_reduce flow."""
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    inputs = build_dual_core_all_reduce_test_inputs(
        mesh_device=submesh,
        num_devices=num_devices,
        output_shape=output_shape,
        input_shard_shape=input_shard_shape,
        tensor_mem_layout=tensor_mem_layout,
        layout=layout,
        input_dtype=input_dtype,
        fuse_residual_add=fuse_residual_add,
    )

    logger.info(f"Running dual-core CCL all-reduce trace: num_devices={num_devices}")
    profiler = BenchmarkProfiler()

    logger.info("Compiling dual-core model")
    ttnn_result = DeepseekMinimalAllReduce.op(
        inputs.input_tensor_mesh,
        inputs.intermediate_tensor_mesh,
        cluster_axis=cluster_axis,
        persistent_output_tensor=inputs.output_tensor_mesh,
        residual_tensor_mesh=inputs.residual_tensor_mesh,
        semaphores=inputs.semaphores[: num_links + 1],
        num_links=num_links,
        dual_core_sender=True,
    )
    ttnn.synchronize_device(submesh)

    logger.info("Capturing dual-core warmup trace")
    trace_id_warmup = ttnn.begin_trace_capture(submesh, cq_id=0)
    for i in range(num_warmup_iter):
        ttnn_result = DeepseekMinimalAllReduce.op(
            inputs.input_tensor_mesh,
            inputs.intermediate_tensor_mesh,
            cluster_axis=cluster_axis,
            persistent_output_tensor=inputs.output_tensor_mesh,
            residual_tensor_mesh=inputs.residual_tensor_mesh,
            semaphores=inputs.semaphores[: num_links + 1],
            num_links=num_links,
            dual_core_sender=True,
        )
    ttnn.end_trace_capture(submesh, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(submesh)

    logger.info("Capturing dual-core main trace")
    trace_id = ttnn.begin_trace_capture(submesh, cq_id=0)
    for i in range(num_iter):
        ttnn_result = DeepseekMinimalAllReduce.op(
            inputs.input_tensor_mesh,
            inputs.intermediate_tensor_mesh,
            cluster_axis=cluster_axis,
            persistent_output_tensor=inputs.output_tensor_mesh,
            residual_tensor_mesh=inputs.residual_tensor_mesh,
            semaphores=inputs.semaphores[: num_links + 1],
            num_links=num_links,
            dual_core_sender=True,
        )
    ttnn.end_trace_capture(submesh, trace_id, cq_id=0)
    ttnn.synchronize_device(submesh)

    logger.info("Executing dual-core warmup trace...")
    profiler.start("deepseek-all-reduce-dual-core-warmup")
    ttnn.execute_trace(submesh, trace_id_warmup, blocking=False)
    ttnn.release_trace(submesh, trace_id_warmup)
    ttnn.synchronize_device(submesh)
    profiler.end("deepseek-all-reduce-dual-core-warmup")

    logger.info("Starting dual-core trace perf test...")
    signpost("start")
    profiler.start("deepseek-all-reduce-dual-core-trace")
    ttnn.execute_trace(submesh, trace_id, blocking=False)
    ttnn.release_trace(submesh, trace_id)
    ttnn.synchronize_device(submesh)
    profiler.end("deepseek-all-reduce-dual-core-trace")
    signpost("stop")

    output_tensor_torch = ttnn.to_torch(
        ttnn_result,
        mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0),
    )

    ref_device_idx = 0
    ref_device_output = output_tensor_torch[ref_device_idx : ref_device_idx + 1, :]
    for device_idx in range(num_devices):
        received = output_tensor_torch[device_idx : device_idx + 1, :]
        assert received.shape == inputs.torch_expected.shape, f"Shape mismatch at device {device_idx}"
        if device_idx != ref_device_idx:
            assert torch.equal(received, ref_device_output), f"Device {device_idx} output mismatch"
        assert torch.allclose(received, inputs.torch_expected, rtol=1e-2, atol=1e-2), (
            f"Dual-core trace output mismatch for device {device_idx}; "
            f"num_links={num_links}, fuse_residual_add={fuse_residual_add}"
        )
