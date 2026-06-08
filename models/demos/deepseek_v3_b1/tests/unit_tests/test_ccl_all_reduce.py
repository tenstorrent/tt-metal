# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
TTNN CCL All-Reduce Test

Tests the deepseek_minimal_all_reduce operation implemented using the generic op infrastructure.
This test validates all-reduce on a 1D mesh (2 devices) where:
1. Each device sends its data to its neighbor via a dedicated sender core
2. Each device receives data from its neighbor on a receiver core
3. The receiver core sums local data (NOC-read from sender) with received data
4. Optionally, a residual tensor is added to the final result to fuse the next residual add block
"""

import os
from dataclasses import dataclass
from typing import Any, Optional

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_slow_dispatch, skip_with_llk_assert
from models.demos.deepseek_v3_b1.micro_ops.ccl_all_gather.op import AllGatherConfig
from models.demos.deepseek_v3_b1.micro_ops.ccl_all_reduce.op import DeepseekMinimalAllReduce
from models.demos.deepseek_v3_b1.tests.unit_tests.ccl_test_utils import (
    create_fabric_router_config,
    get_env_int,
    get_num_links_env_params,
    run_trace_benchmark,
)
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)

TEST_SENDER_CORE = ttnn.CoreCoord(0, 0)
TEST_RECEIVER_CORE = ttnn.CoreCoord(0, 1)

ENV_NUM_LINKS = "CCL_ALL_REDUCE_NUM_LINKS"
ENV_MAX_PAYLOAD_SIZE = "CCL_ALL_REDUCE_MAX_PAYLOAD_SIZE_BYTES"
ALL_REDUCE_OUTPUT_WIDTH = 2048
ALL_REDUCE_OUTPUT_SHAPE = [1, ALL_REDUCE_OUTPUT_WIDTH]
ALL_REDUCE_INPUT_SHARD_SHAPE = (1, ALL_REDUCE_OUTPUT_WIDTH)


def _get_intermediate_shape(input_shard_shape: tuple[int, int]) -> list[int]:
    if input_shard_shape[0] != 1:
        raise ValueError(f"Expected input_shard_shape height 1, got {input_shard_shape[0]}")
    if input_shard_shape[1] % 32 != 0:
        raise ValueError(f"Expected input_shard_shape width divisible by 32, got {input_shard_shape[1]}")
    return [32, input_shard_shape[1] // 32]


MAX_PAYLOAD_SIZE = get_env_int(ENV_MAX_PAYLOAD_SIZE, 15232)


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
    semaphore_count=3,
):
    """Build test tensors with input on sender core and intermediate/output/residual on receiver core.

    ``semaphore_count`` defaults to 3 = two fabric links + ``local_ready``.
    """
    sender_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(TEST_SENDER_CORE, TEST_SENDER_CORE)})
    receiver_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(TEST_RECEIVER_CORE, TEST_RECEIVER_CORE)})
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

    intermediate_shape = _get_intermediate_shape(input_shard_shape)
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


@skip_with_llk_assert("Hit LLK_ASSERT for unpacker data format conversion. Issue: #41024")
@pytest.mark.parametrize(
    "num_devices, output_shape, input_shard_shape, tensor_mem_layout",
    [
        (
            2,
            ALL_REDUCE_OUTPUT_SHAPE,
            ALL_REDUCE_INPUT_SHARD_SHAPE,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("cluster_axis", [0])
@pytest.mark.parametrize("num_links", get_num_links_env_params(ENV_NUM_LINKS, [2]))
@pytest.mark.parametrize("num_iter, num_warmup_iter", [(30, 15)])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(MAX_PAYLOAD_SIZE),
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
    if is_slow_dispatch():
        pytest.skip("CCL all-reduce trace test needs fast dispatch")

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
        semaphore_count=num_links + 1,
    )

    logger.info(
        f"Running CCL all-reduce: num_devices={num_devices}, num_links={num_links}, "
        f"max_payload_size_bytes={MAX_PAYLOAD_SIZE}"
    )

    def run_all_reduce():
        return DeepseekMinimalAllReduce.op(
            inputs.input_tensor_mesh,
            inputs.intermediate_tensor_mesh,
            cluster_axis=cluster_axis,
            persistent_output_tensor=inputs.output_tensor_mesh,
            residual_tensor_mesh=inputs.residual_tensor_mesh,
            semaphores=inputs.semaphores[: num_links + 1],
            num_links=num_links,
        )

    ttnn_result = run_trace_benchmark(
        submesh,
        run_all_reduce,
        num_warmup_iter=num_warmup_iter,
        num_iter=num_iter,
        profiler_name="deepseek-all-reduce",
    )

    logger.info("Verifying all-reduce results...")
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


@skip_with_llk_assert("Hit LLK_ASSERT for unpacker data format conversion. Issue: #41024")
@pytest.mark.parametrize(
    "num_devices, output_shape, input_shard_shape, tensor_mem_layout",
    [
        (
            2,
            ALL_REDUCE_OUTPUT_SHAPE,
            ALL_REDUCE_INPUT_SHARD_SHAPE,
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
        semaphore_count=num_links + 1,
    )

    logger.info(
        f"Running all-reduce feature matrix: num_links={num_links}, chunk_num_tiles={chunk_num_tiles}, "
        f"fuse_residual_add={fuse_residual_add}"
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


def _run_all_reduce_skip_local_push_smoke(
    *,
    input_tensor_mesh,
    intermediate_tensor,
    output_tensor,
    semaphores,
    cluster_axis,
    num_links,
    chunk_num_tiles,
):
    mesh_device = input_tensor_mesh.device()
    mesh_shape = mesh_device.shape
    mesh_rows, mesh_cols = mesh_shape

    allreduce_config = DeepseekMinimalAllReduce.configure(
        mesh_device=mesh_device,
        intermediate_tensor=intermediate_tensor,
        output_tensor=output_tensor,
        semaphores=semaphores,
        cluster_axis=cluster_axis,
        num_links=num_links,
        chunk_num_tiles=chunk_num_tiles,
        input_tensor_mesh=input_tensor_mesh,
        skip_local_push=True,
    )

    mesh_program_descriptor = ttnn.MeshProgramDescriptor()
    kernel_path = "models/demos/deepseek_v3_b1/tests/unit_tests/kernels/all_reduce_skip_local_push_smoke_kernel.cpp"

    sender_core = allreduce_config.sender_core
    receiver_core = allreduce_config.receiver_core
    sender_grid = ttnn.CoreRangeSet([ttnn.CoreRange(sender_core, sender_core)])
    receiver_grid = ttnn.CoreRangeSet([ttnn.CoreRange(receiver_core, receiver_core)])
    combined_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(sender_core, sender_core), ttnn.CoreRange(receiver_core, receiver_core)]
    )

    for row in range(mesh_rows):
        for col in range(mesh_cols):
            coord = ttnn.MeshCoordinate(row, col)
            unified_kernel = UnifiedKernelDescriptor(
                kernel_source=kernel_path,
                core_ranges=combined_grid,
                ncrisc_named_compile_time_args=allreduce_config.get_ncrisc_named_ct_args(coord),
                brisc_named_compile_time_args=allreduce_config.get_brisc_named_ct_args(coord),
                trisc_named_compile_time_args=allreduce_config.get_trisc_named_ct_args(coord)
                + [
                    ("allreduce_local_data_cb_id", allreduce_config.local_data_cb_id),
                    ("allreduce_input_num_tiles", allreduce_config.total_num_tiles),
                ],
                trisc_compute_config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    fp32_dest_acc_en=True,
                    dst_full_sync_en=True,
                    math_approx_mode=False,
                ),
                unified_compile_time_core_descriptors=[
                    UnifiedCompileTimeCoreDescriptor(
                        named_compile_time_arg="is_allreduce_sender_core",
                        core_range=sender_grid,
                        value=1,
                        other_value=0,
                    ),
                    UnifiedCompileTimeCoreDescriptor(
                        named_compile_time_arg="is_allreduce_receiver_core",
                        core_range=receiver_grid,
                        value=1,
                        other_value=0,
                    ),
                ],
                per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                    ncrisc_args=[(sender_core, []), (receiver_core, [])],
                    brisc_args=[(sender_core, []), (receiver_core, [])],
                    trisc_args=[(sender_core, []), (receiver_core, [])],
                ),
                noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
            )
            kernel_result = unified_kernel.get_kernel_descriptors()
            sender_group = kernel_result.get_group_by_arg("is_allreduce_sender_core", 1)
            receiver_group = kernel_result.get_group_by_arg("is_allreduce_receiver_core", 1)
            if sender_group is None or receiver_group is None:
                raise ValueError("Expected sender and receiver kernel groups")

            program = ttnn.ProgramDescriptor(
                kernels=kernel_result.kernels,
                semaphores=[],
                cbs=allreduce_config.get_cb_descriptors(coord),
            )
            program.kernels[
                sender_group.ncrisc_kernel_index
            ].common_runtime_args = allreduce_config.get_sender_ncrisc_common_rt_args(coord)
            program.kernels[
                sender_group.brisc_kernel_index
            ].common_runtime_args = allreduce_config.get_sender_brisc_common_rt_args(coord)
            program.kernels[sender_group.trisc_kernel_index].common_runtime_args = []
            program.kernels[
                receiver_group.ncrisc_kernel_index
            ].common_runtime_args = allreduce_config.get_receiver_ncrisc_common_rt_args(coord)
            program.kernels[receiver_group.brisc_kernel_index].common_runtime_args = []
            program.kernels[receiver_group.trisc_kernel_index].common_runtime_args = []

            ncrisc_per_core_rt = program.kernels[sender_group.ncrisc_kernel_index].runtime_args[sender_core.x][
                sender_core.y
            ]
            ncrisc_per_core_rt.extend(allreduce_config.get_ncrisc_per_core_rt_args(coord, program, sender_core))
            brisc_per_core_rt = program.kernels[sender_group.brisc_kernel_index].runtime_args[sender_core.x][
                sender_core.y
            ]
            brisc_per_core_rt.extend(allreduce_config.get_brisc_per_core_rt_args(coord, program, sender_core))

            mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

    ttnn.generic_op([input_tensor_mesh, output_tensor, intermediate_tensor], mesh_program_descriptor)
    return output_tensor


@pytest.mark.parametrize(
    "num_devices, output_shape, input_shard_shape, tensor_mem_layout",
    [
        (
            2,
            ALL_REDUCE_OUTPUT_SHAPE,
            ALL_REDUCE_INPUT_SHARD_SHAPE,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("cluster_axis", [0])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("chunk_num_tiles", [1])
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
def test_ccl_all_reduce_skip_local_push_smoke(
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
        fuse_residual_add=False,
        semaphore_count=num_links + 1,
    )

    ttnn_result = _run_all_reduce_skip_local_push_smoke(
        input_tensor_mesh=inputs.input_tensor_mesh,
        intermediate_tensor=inputs.intermediate_tensor_mesh,
        output_tensor=inputs.output_tensor_mesh,
        semaphores=inputs.semaphores[: num_links + 1],
        cluster_axis=cluster_axis,
        num_links=num_links,
        chunk_num_tiles=chunk_num_tiles,
    )
    ttnn.synchronize_device(submesh)

    output_tensor_torch = ttnn.to_torch(
        ttnn_result,
        mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0),
    )

    ref_device_output = output_tensor_torch[0:1, :]
    for device_idx in range(num_devices):
        received = output_tensor_torch[device_idx : device_idx + 1, :]
        if device_idx != 0:
            assert torch.equal(received, ref_device_output), f"Device {device_idx} output mismatch"
        assert torch.allclose(received, inputs.torch_expected, rtol=1e-2, atol=1e-2), (
            f"Output mismatch for device {device_idx}; "
            f"num_links={num_links}, chunk_num_tiles={chunk_num_tiles}, skip_local_push=True"
        )


def _create_mesh_tensor_from_per_device(
    *,
    mesh_device,
    per_device_tensors,
    core_range_set,
    shard_shape,
    tensor_mem_layout,
    tile,
    dtype,
):
    mesh_mapper_config = ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementReplicate()], mesh_device.shape)
    shard_spec = ttnn.ShardSpec(
        core_range_set,
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=shard_spec)
    return ttnn.from_torch(
        torch.cat(per_device_tensors, dim=0),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config,
        mesh_mapper=ttnn.create_mesh_mapper(mesh_device, mesh_mapper_config),
    )


def _create_mesh_tensor_from_per_device_2d(
    *,
    mesh_device,
    per_device_tensors,
    core_range_set,
    shard_shape,
    tensor_mem_layout,
    tile,
    dtype,
):
    mesh_rows, mesh_cols = mesh_device.shape
    if len(per_device_tensors) != mesh_rows * mesh_cols:
        raise ValueError(f"Expected {mesh_rows * mesh_cols} tensors, got {len(per_device_tensors)}")

    rows = []
    for row in range(mesh_rows):
        row_tensors = []
        for col in range(mesh_cols):
            row_tensors.append(per_device_tensors[row * mesh_cols + col])
        rows.append(torch.cat(row_tensors, dim=1))
    host_tensor = torch.cat(rows, dim=0)

    mesh_mapper_config = ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)], mesh_device.shape)
    shard_spec = ttnn.ShardSpec(
        core_range_set,
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=shard_spec)
    return ttnn.from_torch(
        host_tensor,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config,
        mesh_mapper=ttnn.create_mesh_mapper(mesh_device, mesh_mapper_config),
    )


def _run_gather_reduce_all_reduce_smoke(
    *,
    source_tensor_mesh,
    scratch_tensor_mesh,
    gather_out_tensor_mesh,
    intermediate_tensor_mesh,
    output_tensor_mesh,
    allgather_output_tensor_mesh,
    semaphores,
    gather_semaphore_addrs,
    ccl_sync_semaphore_addr,
    ccl_sync2_semaphore_addr,
    allgather_handoff_semaphore_addr,
    allgather_recv_semaphore_addr,
    allgather_transport_risc,
    allgather_open_after_allreduce,
    allgather_use_config,
    allgather_gather_enabled,
    allreduce_cluster_axis,
    allgather_cluster_axis,
    num_links,
    chunk_num_tiles,
    gather_dst_num_tiles,
    gather_sender_grid,
    gather_receiver_core,
    allreduce_receiver_core,
    gather_reduce_data_size_bytes,
    gather_reduce_src_num_pages,
):
    source_cb = 0
    gather_scratch_cb = 1
    gather_out_cb = 2
    ccl_recv_local_data_cb = 3
    ccl_remote_data_cb = 4
    ccl_output_cb = 5

    tile_32x32 = ttnn.Tile((32, 32))
    tile_32x32_size = tile_32x32.get_tile_size(ttnn.bfloat16)

    mesh_device = source_tensor_mesh.device()
    mesh_shape = mesh_device.shape
    mesh_rows, mesh_cols = mesh_shape

    allreduce_config = DeepseekMinimalAllReduce.configure(
        mesh_device=mesh_device,
        intermediate_tensor=intermediate_tensor_mesh,
        output_tensor=output_tensor_mesh,
        semaphores=semaphores,
        cluster_axis=allreduce_cluster_axis,
        num_links=num_links,
        chunk_num_tiles=chunk_num_tiles,
        local_data_cb_id=gather_out_cb,
        recv_local_data_cb_id=ccl_recv_local_data_cb,
        remote_data_cb_id=ccl_remote_data_cb,
        output_cb_id=ccl_output_cb,
        input_tensor_mesh=gather_out_tensor_mesh,
        skip_local_push=True,
        num_tiles_override=gather_dst_num_tiles,
        page_size_override=tile_32x32_size,
    )

    gather_receiver_grid = ttnn.CoreRangeSet([ttnn.CoreRange(gather_receiver_core, gather_receiver_core)])
    allreduce_receiver_grid = ttnn.CoreRangeSet([ttnn.CoreRange(allreduce_receiver_core, allreduce_receiver_core)])
    combined_grid = gather_sender_grid.merge(gather_receiver_grid).merge(allreduce_receiver_grid)

    ref_device = ttnn.get_device_tensors(gather_out_tensor_mesh)[0].device()
    gather_receiver_noc = ref_device.worker_core_from_logical_core(gather_receiver_core)
    allreduce_receiver_noc = ref_device.worker_core_from_logical_core(allreduce_receiver_core)
    allgather_scratch_base_addr = ttnn.get_device_tensors(scratch_tensor_mesh)[0].buffer_address()
    allgather_output_base_addr = ttnn.get_device_tensors(allgather_output_tensor_mesh)[0].buffer_address()
    allgather_config = None
    if allgather_use_config:
        allgather_config = AllGatherConfig(
            mesh_device=mesh_device,
            cluster_axis=allgather_cluster_axis,
            num_links=1,
            slice_size_bytes=gather_dst_num_tiles * tile_32x32_size,
            output_addr=allgather_output_base_addr,
            scratch_base_addr=allgather_scratch_base_addr,
            handoff_sem_addr=allgather_handoff_semaphore_addr,
            recv_sem_addr=allgather_recv_semaphore_addr,
            gather_noc_core=allreduce_receiver_noc,
            transport_noc_core=gather_receiver_noc,
            output_cb_id=ccl_output_cb,
            output_num_tiles=gather_dst_num_tiles,
        )

    def build_allgather_transport_per_core_rt_args(coord, program, core, row, col):
        if allgather_cluster_axis == 0:
            neighbor_coord = ttnn.MeshCoordinate(1 - row, col)
        else:
            neighbor_coord = ttnn.MeshCoordinate(row, 1 - col)
        src_fabric_node_id = mesh_device.get_fabric_node_id(coord)
        dst_fabric_node_id = mesh_device.get_fabric_node_id(neighbor_coord)
        out = [
            int(dst_fabric_node_id.mesh_id),
            int(dst_fabric_node_id.chip_id),
        ]
        out.extend(
            ttnn.setup_fabric_connection(
                src_fabric_node_id=src_fabric_node_id,
                dst_fabric_node_id=dst_fabric_node_id,
                link_idx=0,
                program_descriptor=program,
                worker_core=core,
            )
        )
        return out

    ncrisc_named_compile_time_args = [
        ("gather_reduce_dest_noc_x", gather_receiver_noc.x),
        ("gather_reduce_dest_noc_y", gather_receiver_noc.y),
        ("gather_reduce_data_size_bytes", gather_reduce_data_size_bytes),
        ("gather_reduce_receiver_semaphore_addr", gather_semaphore_addrs[0]),
        ("gather_reduce_src_cb", source_cb),
        ("gather_reduce_src_num_pages", gather_reduce_src_num_pages),
        ("gather_reduce_grid_start_x", 0),
        ("gather_reduce_grid_start_y", 0),
        ("gather_reduce_grid_end_x", 1),
        ("gather_reduce_grid_end_y", 0),
        ("gather_reduce_half_num_cores", 1),
        ("gather_reduce_dst_cb", gather_scratch_cb),
        ("gather_reduce_half_size_bytes", gather_dst_num_tiles * tile_32x32_size),
    ]

    brisc_named_compile_time_args = [
        ("gather_reduce_noc0_num_senders", 2),
        ("gather_reduce_noc1_num_senders", 0),
        ("gather_reduce_noc0_receiver_semaphore_addr", gather_semaphore_addrs[0]),
        ("gather_reduce_noc1_receiver_semaphore_addr", gather_semaphore_addrs[1]),
        ("gather_reduce_dst_cb", gather_scratch_cb),
        ("gather_reduce_dst_num_tiles", gather_dst_num_tiles),
    ]

    trisc_named_compile_time_args = [
        ("gather_reduce_scratch_cb", gather_scratch_cb),
        ("gather_reduce_out_cb", gather_out_cb),
        ("gather_reduce_dst_num_tiles", gather_dst_num_tiles),
        ("allgather_transport_enabled", 0),
        ("allgather_gather_enabled", int(allgather_gather_enabled)),
        ("allgather_open_after_allreduce", int(allgather_open_after_allreduce)),
    ]
    ccl_sync_named_compile_time_args = [
        ("ccl_sync_semaphore_addr", ccl_sync_semaphore_addr),
        ("ccl_sync_semaphore2_addr", ccl_sync2_semaphore_addr),
        ("ccl_sync_dest_noc_x", gather_receiver_noc.x),
        ("ccl_sync_dest_noc_y", gather_receiver_noc.y),
        ("ccl_sync2_dest_noc_x", gather_receiver_noc.x),
        ("ccl_sync2_dest_noc_y", gather_receiver_noc.y),
        ("sdpa_fwd_num_cores", 1),
    ]
    if allgather_transport_risc not in {"none", "brisc", "ncrisc", "both"}:
        raise ValueError(
            "Expected allgather_transport_risc to be one of none/brisc/ncrisc/both, " f"got {allgather_transport_risc}"
        )
    ncrisc_allgather_transport_enabled = 1 if allgather_transport_risc in {"ncrisc", "both"} else 0
    brisc_allgather_transport_enabled = 1 if allgather_transport_risc in {"brisc", "both"} else 0

    mesh_program_descriptor = ttnn.MeshProgramDescriptor()
    kernel_path = "models/demos/deepseek_v3_b1/tests/unit_tests/kernels/gather_reduce_all_reduce_smoke_kernel.cpp"

    for row in range(mesh_rows):
        for col in range(mesh_cols):
            coord = ttnn.MeshCoordinate(row, col)
            if allgather_config is None:
                allgather_ncrisc_named_compile_time_args = [
                    ("allgather_slice_size_bytes", tile_32x32_size),
                    ("allgather_num_chunks", 1),
                    ("allgather_chunk_size_bytes", tile_32x32_size),
                    ("allgather_last_chunk_bytes", tile_32x32_size),
                    ("allgather_num_links", 1),
                    ("allgather_recv_sem_bits_per_slot", 4),
                    ("allgather_r2_active", 0),
                    ("allgather_scratch_base_addr", allgather_scratch_base_addr),
                    ("allgather_handoff_sem_addr", allgather_handoff_semaphore_addr),
                    ("allgather_dest_output_base_addr", allgather_output_base_addr),
                    ("allgather_r1_dest_slot_index", 0),
                    ("allgather_dest_noc_x", allreduce_receiver_noc.x),
                    ("allgather_dest_noc_y", allreduce_receiver_noc.y),
                    ("allgather_dest_recv_sem_addr", allgather_recv_semaphore_addr),
                    ("allgather_r2_dest_slot_index", 0),
                    ("allgather_gather_slice_size_bytes", tile_32x32_size),
                    ("allgather_gather_num_chunks", 1),
                    ("allgather_ring_size", 4),
                    ("allgather_self_slot_index", 0),
                    ("allgather_transport_noc_x", gather_receiver_noc.x),
                    ("allgather_transport_noc_y", gather_receiver_noc.y),
                    ("allgather_recv_sem_addr", allgather_recv_semaphore_addr),
                    ("allgather_r2_src_slot_index", 0),
                    ("output_cb_id", ccl_output_cb),
                    ("output_num_tiles", gather_dst_num_tiles),
                ]
                allgather_brisc_named_compile_time_args = list(allgather_ncrisc_named_compile_time_args)
            else:
                allgather_ncrisc_named_compile_time_args = allgather_config.get_fused_ncrisc_named_ct_args(coord)
                allgather_brisc_named_compile_time_args = allgather_config.get_fused_brisc_named_ct_args(coord)
            allgather_ncrisc_named_compile_time_args = allgather_ncrisc_named_compile_time_args + [
                ("allgather_transport_enabled", ncrisc_allgather_transport_enabled),
                ("allgather_gather_enabled", int(allgather_gather_enabled)),
                ("allgather_open_after_allreduce", int(allgather_open_after_allreduce)),
            ]
            allgather_brisc_named_compile_time_args = allgather_brisc_named_compile_time_args + [
                ("allgather_transport_enabled", brisc_allgather_transport_enabled),
                ("allgather_gather_enabled", int(allgather_gather_enabled)),
                ("allgather_open_after_allreduce", int(allgather_open_after_allreduce)),
            ]
            unified_kernel = UnifiedKernelDescriptor(
                kernel_source=kernel_path,
                core_ranges=combined_grid,
                ncrisc_named_compile_time_args=ncrisc_named_compile_time_args
                + ccl_sync_named_compile_time_args
                + allgather_ncrisc_named_compile_time_args
                + allreduce_config.get_ncrisc_named_ct_args(coord),
                brisc_named_compile_time_args=brisc_named_compile_time_args
                + ccl_sync_named_compile_time_args
                + allgather_brisc_named_compile_time_args
                + allreduce_config.get_brisc_named_ct_args(coord),
                trisc_named_compile_time_args=trisc_named_compile_time_args
                + allreduce_config.get_trisc_named_ct_args(coord),
                trisc_compute_config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    fp32_dest_acc_en=True,
                    dst_full_sync_en=True,
                    math_approx_mode=False,
                ),
                unified_compile_time_core_descriptors=[
                    UnifiedCompileTimeCoreDescriptor(
                        named_compile_time_arg="is_gather_sender_core",
                        core_range=gather_sender_grid,
                        value=1,
                        other_value=0,
                    ),
                    UnifiedCompileTimeCoreDescriptor(
                        named_compile_time_arg="is_gather_receiver_core",
                        core_range=gather_receiver_grid,
                        value=1,
                        other_value=0,
                    ),
                    UnifiedCompileTimeCoreDescriptor(
                        named_compile_time_arg="is_allreduce_sender_core",
                        core_range=gather_receiver_grid,
                        value=1,
                        other_value=0,
                    ),
                    UnifiedCompileTimeCoreDescriptor(
                        named_compile_time_arg="is_allreduce_receiver_core",
                        core_range=allreduce_receiver_grid,
                        value=1,
                        other_value=0,
                    ),
                    UnifiedCompileTimeCoreDescriptor(
                        named_compile_time_arg="is_ccl_sync_producer_core",
                        core_range=allreduce_receiver_grid,
                        value=1,
                        other_value=0,
                    ),
                    UnifiedCompileTimeCoreDescriptor(
                        named_compile_time_arg="is_ccl_sync2_producer_core",
                        core_range=allreduce_receiver_grid,
                        value=1,
                        other_value=0,
                    ),
                ],
                per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                    ncrisc_args=[(gather_receiver_core, []), (allreduce_receiver_core, [])],
                    brisc_args=[(gather_receiver_core, []), (allreduce_receiver_core, [])],
                    trisc_args=[(gather_receiver_core, []), (allreduce_receiver_core, [])],
                ),
                noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
            )
            kernel_result = unified_kernel.get_kernel_descriptors()
            sender_group = kernel_result.get_group_by_arg("is_allreduce_sender_core", 1)
            receiver_group = kernel_result.get_group_by_arg("is_allreduce_receiver_core", 1)
            if sender_group is None or receiver_group is None:
                raise ValueError("Expected all-reduce sender and receiver kernel groups")

            device_idx = row * mesh_cols + col
            source_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                source_cb, ttnn.get_device_tensors(source_tensor_mesh)[device_idx]
            )
            gather_scratch_grid = gather_sender_grid.merge(gather_receiver_grid)
            scratch_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                gather_scratch_cb,
                ttnn.get_device_tensors(scratch_tensor_mesh)[device_idx],
                total_size=2 * gather_dst_num_tiles * tile_32x32_size,
                core_ranges=gather_scratch_grid,
            )
            scratch_cb_descriptor.format_descriptors = [
                ttnn.CBFormatDescriptor(
                    buffer_index=gather_scratch_cb,
                    data_format=ttnn.bfloat16,
                    page_size=tile_32x32_size,
                    tile=ttnn.TileDescriptor(tile_32x32),
                )
            ]

            allreduce_cb_descriptors = allreduce_config.get_cb_descriptors(coord)

            program = ttnn.ProgramDescriptor(
                kernels=kernel_result.kernels,
                semaphores=[],
                cbs=[
                    source_cb_descriptor,
                    scratch_cb_descriptor,
                    *allreduce_cb_descriptors,
                ],
            )
            program.kernels[
                sender_group.ncrisc_kernel_index
            ].common_runtime_args = allreduce_config.get_sender_ncrisc_common_rt_args(coord)
            program.kernels[
                sender_group.brisc_kernel_index
            ].common_runtime_args = allreduce_config.get_sender_brisc_common_rt_args(coord)
            program.kernels[sender_group.trisc_kernel_index].common_runtime_args = []
            program.kernels[
                receiver_group.ncrisc_kernel_index
            ].common_runtime_args = allreduce_config.get_receiver_ncrisc_common_rt_args(coord)
            program.kernels[receiver_group.brisc_kernel_index].common_runtime_args = []
            program.kernels[receiver_group.trisc_kernel_index].common_runtime_args = []

            ncrisc_allreduce_per_core_rt = allreduce_config.get_ncrisc_per_core_rt_args(
                coord, program, gather_receiver_core
            )
            ncrisc_allgather_start_idx = len(ncrisc_allreduce_per_core_rt)
            program.kernels[sender_group.ncrisc_kernel_index].common_runtime_args.append(ncrisc_allgather_start_idx)
            ncrisc_per_core_rt = program.kernels[sender_group.ncrisc_kernel_index].runtime_args[gather_receiver_core.x][
                gather_receiver_core.y
            ]
            ncrisc_per_core_rt.extend(ncrisc_allreduce_per_core_rt)
            if allgather_config is None:
                ncrisc_per_core_rt.extend(
                    build_allgather_transport_per_core_rt_args(coord, program, gather_receiver_core, row, col)
                )
            else:
                ncrisc_per_core_rt.extend(
                    allgather_config.get_transport_ncrisc_per_core_rt_args(coord, program, gather_receiver_core)
                )
            brisc_per_core_rt = program.kernels[sender_group.brisc_kernel_index].runtime_args[gather_receiver_core.x][
                gather_receiver_core.y
            ]
            brisc_allreduce_per_core_rt = allreduce_config.get_brisc_per_core_rt_args(
                coord, program, gather_receiver_core
            )
            brisc_allgather_start_idx = len(brisc_allreduce_per_core_rt)
            program.kernels[sender_group.brisc_kernel_index].common_runtime_args.append(brisc_allgather_start_idx)
            brisc_per_core_rt.extend(brisc_allreduce_per_core_rt)
            if allgather_config is None:
                brisc_per_core_rt.extend(
                    build_allgather_transport_per_core_rt_args(coord, program, gather_receiver_core, row, col)
                )
            else:
                brisc_per_core_rt.extend(
                    allgather_config.get_transport_brisc_per_core_rt_args(coord, program, gather_receiver_core)
                )

            mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

    ttnn.generic_op(
        [
            source_tensor_mesh,
            scratch_tensor_mesh,
            gather_out_tensor_mesh,
            intermediate_tensor_mesh,
            output_tensor_mesh,
            allgather_output_tensor_mesh,
        ],
        mesh_program_descriptor,
    )
    if allgather_gather_enabled:
        return allgather_output_tensor_mesh
    return output_tensor_mesh


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
def test_ccl_all_reduce_with_two_sender_gather_reduce_smoke(bh_2d_mesh_device):
    num_devices = 2
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    gather_sender_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
    gather_receiver_core = ttnn.CoreCoord(2, 0)
    allreduce_receiver_core = ttnn.CoreCoord(3, 0)
    gather_receiver_grid = ttnn.CoreRangeSet([ttnn.CoreRange(gather_receiver_core, gather_receiver_core)])
    gather_scratch_grid = gather_sender_grid.merge(gather_receiver_grid)
    allreduce_receiver_grid = ttnn.CoreRangeSet([ttnn.CoreRange(allreduce_receiver_core, allreduce_receiver_core)])

    tile_1x32 = ttnn.Tile((1, 32))
    tile_32x32 = ttnn.Tile((32, 32))
    tile_1x32_size = tile_1x32.get_tile_size(ttnn.bfloat16)
    tile_32x32_size = tile_32x32.get_tile_size(ttnn.bfloat16)
    torch.manual_seed(0)
    source_per_device = [torch.rand((1, 64), dtype=torch.bfloat16) for _ in range(num_devices)]
    zeros_32x32 = [torch.zeros((32, 32), dtype=torch.bfloat16) for _ in range(num_devices)]
    zeros_scratch = [torch.zeros((192, 32), dtype=torch.bfloat16) for _ in range(num_devices)]

    source_tensor_mesh = _create_mesh_tensor_from_per_device(
        mesh_device=submesh,
        per_device_tensors=source_per_device,
        core_range_set=gather_sender_grid,
        shard_shape=(1, 32),
        tensor_mem_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        tile=tile_1x32,
        dtype=ttnn.bfloat16,
    )
    scratch_tensor_mesh = _create_mesh_tensor_from_per_device(
        mesh_device=submesh,
        per_device_tensors=zeros_scratch,
        core_range_set=gather_scratch_grid,
        shard_shape=(64, 32),
        tensor_mem_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        tile=tile_32x32,
        dtype=ttnn.bfloat16,
    )
    gather_out_tensor_mesh = _create_mesh_tensor_from_per_device(
        mesh_device=submesh,
        per_device_tensors=zeros_32x32,
        core_range_set=gather_receiver_grid,
        shard_shape=(32, 32),
        tensor_mem_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        tile=tile_32x32,
        dtype=ttnn.bfloat16,
    )
    intermediate_tensor_mesh = _create_mesh_tensor_from_per_device(
        mesh_device=submesh,
        per_device_tensors=zeros_32x32,
        core_range_set=allreduce_receiver_grid,
        shard_shape=(32, 32),
        tensor_mem_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        tile=tile_32x32,
        dtype=ttnn.bfloat16,
    )
    output_tensor_mesh = _create_mesh_tensor_from_per_device(
        mesh_device=submesh,
        per_device_tensors=zeros_32x32,
        core_range_set=allreduce_receiver_grid,
        shard_shape=(32, 32),
        tensor_mem_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        tile=tile_32x32,
        dtype=ttnn.bfloat16,
    )
    allgather_output_tensor_mesh = _create_mesh_tensor_from_per_device(
        mesh_device=submesh,
        per_device_tensors=zeros_32x32,
        core_range_set=allreduce_receiver_grid,
        shard_shape=(32, 32),
        tensor_mem_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        tile=tile_32x32,
        dtype=ttnn.bfloat16,
    )

    compute_grid_size = submesh.compute_with_storage_grid_size()
    available_cores = ttnn.num_cores_to_corerangeset(
        compute_grid_size.x * compute_grid_size.y,
        compute_grid_size,
        row_wise=True,
    )
    allreduce_semaphores = [ttnn.create_global_semaphore(submesh, available_cores, 0) for _ in range(2)]
    gather_semaphores = [ttnn.create_global_semaphore(submesh, available_cores, 0) for _ in range(2)]
    ccl_sync_semaphore = ttnn.create_global_semaphore(submesh, available_cores, 0)
    ccl_sync2_semaphore = ttnn.create_global_semaphore(submesh, available_cores, 0)
    allgather_semaphores = [ttnn.create_global_semaphore(submesh, available_cores, 0) for _ in range(2)]
    gather_semaphore_addrs = [ttnn.get_global_semaphore_address(sem) for sem in gather_semaphores]
    ccl_sync_semaphore_addr = ttnn.get_global_semaphore_address(ccl_sync_semaphore)
    ccl_sync2_semaphore_addr = ttnn.get_global_semaphore_address(ccl_sync2_semaphore)
    allgather_semaphore_addrs = [ttnn.get_global_semaphore_address(sem) for sem in allgather_semaphores]
    allgather_transport_risc = os.environ.get("GATHER_ALLGATHER_RISC", "none").lower()
    allgather_open = os.environ.get("GATHER_ALLGATHER_OPEN", "early").lower()
    if allgather_open not in {"early", "late"}:
        raise ValueError(f"Expected GATHER_ALLGATHER_OPEN to be early or late, got {allgather_open}")
    allgather_open_after_allreduce = allgather_open == "late"

    ttnn_result = _run_gather_reduce_all_reduce_smoke(
        source_tensor_mesh=source_tensor_mesh,
        scratch_tensor_mesh=scratch_tensor_mesh,
        gather_out_tensor_mesh=gather_out_tensor_mesh,
        intermediate_tensor_mesh=intermediate_tensor_mesh,
        output_tensor_mesh=output_tensor_mesh,
        allgather_output_tensor_mesh=allgather_output_tensor_mesh,
        semaphores=allreduce_semaphores,
        gather_semaphore_addrs=gather_semaphore_addrs,
        ccl_sync_semaphore_addr=ccl_sync_semaphore_addr,
        ccl_sync2_semaphore_addr=ccl_sync2_semaphore_addr,
        allgather_handoff_semaphore_addr=allgather_semaphore_addrs[0],
        allgather_recv_semaphore_addr=allgather_semaphore_addrs[1],
        allgather_transport_risc=allgather_transport_risc,
        allgather_open_after_allreduce=allgather_open_after_allreduce,
        allgather_use_config=False,
        allgather_gather_enabled=False,
        allreduce_cluster_axis=0,
        allgather_cluster_axis=0,
        num_links=1,
        chunk_num_tiles=1,
        gather_dst_num_tiles=1,
        gather_sender_grid=gather_sender_grid,
        gather_receiver_core=gather_receiver_core,
        allreduce_receiver_core=allreduce_receiver_core,
        gather_reduce_data_size_bytes=tile_1x32_size,
        gather_reduce_src_num_pages=1,
    )
    ttnn.synchronize_device(submesh)

    expected_per_device = []
    for source in source_per_device:
        reduced_row = (source[:, :32].float() + source[:, 32:64].float()).bfloat16()
        reduced = torch.zeros((32, 32), dtype=torch.bfloat16)
        reduced[0, 0:16] = reduced_row[0, 0:16]
        reduced[1, 0:16] = reduced_row[0, 16:32]
        expected_per_device.append(reduced)
    expected = sum(t.float() for t in expected_per_device).bfloat16()

    output_tensor_torch = ttnn.to_torch(
        ttnn_result,
        mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0),
    )
    for device_idx in range(num_devices):
        received = output_tensor_torch[device_idx * 32 : (device_idx + 1) * 32, :]
        assert torch.allclose(
            received, expected, rtol=1e-2, atol=1e-2
        ), f"Output mismatch for device {device_idx} in gather-reduce all-reduce smoke"


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
def test_ccl_all_reduce_gather_allgather_4x2_smoke(bh_2d_mesh_device):
    mesh_rows = 4
    mesh_cols = 2
    num_devices = mesh_rows * mesh_cols
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    gather_sender_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
    gather_receiver_core = ttnn.CoreCoord(2, 0)
    allreduce_receiver_core = ttnn.CoreCoord(3, 0)
    gather_receiver_grid = ttnn.CoreRangeSet([ttnn.CoreRange(gather_receiver_core, gather_receiver_core)])
    gather_scratch_grid = gather_sender_grid.merge(gather_receiver_grid)
    allreduce_receiver_grid = ttnn.CoreRangeSet([ttnn.CoreRange(allreduce_receiver_core, allreduce_receiver_core)])

    tile_32x32 = ttnn.Tile((32, 32))
    tile_32x32_size = tile_32x32.get_tile_size(ttnn.bfloat16)
    gather_dst_num_tiles = int(os.environ.get("GATHER_4X2_DST_TILES", "2"))
    if gather_dst_num_tiles <= 0:
        raise ValueError(f"GATHER_4X2_DST_TILES must be positive, got {gather_dst_num_tiles}")
    reduced_rows = 32 * gather_dst_num_tiles
    allgather_rows = mesh_rows * reduced_rows
    scratch_rows = 3 * 2 * reduced_rows

    source_per_device = []
    for device_idx in range(num_devices):
        left = torch.zeros((reduced_rows, 32), dtype=torch.bfloat16)
        right = torch.full((reduced_rows, 32), device_idx + 17, dtype=torch.bfloat16)
        source_per_device.append(torch.cat([left, right], dim=1))
    zeros_reduced = [torch.zeros((reduced_rows, 32), dtype=torch.bfloat16) for _ in range(num_devices)]
    zeros_scratch = [torch.zeros((scratch_rows, 32), dtype=torch.bfloat16) for _ in range(num_devices)]
    zeros_allgather = [torch.zeros((allgather_rows, 32), dtype=torch.bfloat16) for _ in range(num_devices)]

    source_tensor_mesh = _create_mesh_tensor_from_per_device_2d(
        mesh_device=submesh,
        per_device_tensors=source_per_device,
        core_range_set=gather_sender_grid,
        shard_shape=(reduced_rows, 32),
        tensor_mem_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        tile=tile_32x32,
        dtype=ttnn.bfloat16,
    )
    scratch_tensor_mesh = _create_mesh_tensor_from_per_device_2d(
        mesh_device=submesh,
        per_device_tensors=zeros_scratch,
        core_range_set=gather_scratch_grid,
        shard_shape=(2 * reduced_rows, 32),
        tensor_mem_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        tile=tile_32x32,
        dtype=ttnn.bfloat16,
    )
    gather_out_tensor_mesh = _create_mesh_tensor_from_per_device_2d(
        mesh_device=submesh,
        per_device_tensors=zeros_reduced,
        core_range_set=gather_receiver_grid,
        shard_shape=(reduced_rows, 32),
        tensor_mem_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        tile=tile_32x32,
        dtype=ttnn.bfloat16,
    )
    intermediate_tensor_mesh = _create_mesh_tensor_from_per_device_2d(
        mesh_device=submesh,
        per_device_tensors=zeros_reduced,
        core_range_set=allreduce_receiver_grid,
        shard_shape=(reduced_rows, 32),
        tensor_mem_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        tile=tile_32x32,
        dtype=ttnn.bfloat16,
    )
    output_tensor_mesh = _create_mesh_tensor_from_per_device_2d(
        mesh_device=submesh,
        per_device_tensors=zeros_reduced,
        core_range_set=allreduce_receiver_grid,
        shard_shape=(reduced_rows, 32),
        tensor_mem_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        tile=tile_32x32,
        dtype=ttnn.bfloat16,
    )
    allgather_output_tensor_mesh = _create_mesh_tensor_from_per_device_2d(
        mesh_device=submesh,
        per_device_tensors=zeros_allgather,
        core_range_set=allreduce_receiver_grid,
        shard_shape=(allgather_rows, 32),
        tensor_mem_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        tile=tile_32x32,
        dtype=ttnn.bfloat16,
    )

    compute_grid_size = submesh.compute_with_storage_grid_size()
    available_cores = ttnn.num_cores_to_corerangeset(
        compute_grid_size.x * compute_grid_size.y,
        compute_grid_size,
        row_wise=True,
    )
    allreduce_num_links = int(os.environ.get("GATHER_4X2_ALLREDUCE_LINKS", "2"))
    if allreduce_num_links <= 0:
        raise ValueError(f"GATHER_4X2_ALLREDUCE_LINKS must be positive, got {allreduce_num_links}")
    allreduce_semaphores = [
        ttnn.create_global_semaphore(submesh, available_cores, 0) for _ in range(allreduce_num_links + 1)
    ]
    gather_semaphores = [ttnn.create_global_semaphore(submesh, available_cores, 0) for _ in range(2)]
    ccl_sync_semaphore = ttnn.create_global_semaphore(submesh, available_cores, 0)
    ccl_sync2_semaphore = ttnn.create_global_semaphore(submesh, available_cores, 0)
    allgather_semaphores = [ttnn.create_global_semaphore(submesh, available_cores, 0) for _ in range(2)]
    gather_semaphore_addrs = [ttnn.get_global_semaphore_address(sem) for sem in gather_semaphores]
    ccl_sync_semaphore_addr = ttnn.get_global_semaphore_address(ccl_sync_semaphore)
    ccl_sync2_semaphore_addr = ttnn.get_global_semaphore_address(ccl_sync2_semaphore)
    allgather_semaphore_addrs = [ttnn.get_global_semaphore_address(sem) for sem in allgather_semaphores]
    allgather_transport_risc = os.environ.get("GATHER_ALLGATHER_RISC", "both").lower()
    allgather_open = os.environ.get("GATHER_ALLGATHER_OPEN", "early").lower()
    if allgather_open not in {"early", "late"}:
        raise ValueError(f"Expected GATHER_ALLGATHER_OPEN to be early or late, got {allgather_open}")
    allgather_enabled = allgather_transport_risc != "none"

    ttnn_result = _run_gather_reduce_all_reduce_smoke(
        source_tensor_mesh=source_tensor_mesh,
        scratch_tensor_mesh=scratch_tensor_mesh,
        gather_out_tensor_mesh=gather_out_tensor_mesh,
        intermediate_tensor_mesh=intermediate_tensor_mesh,
        output_tensor_mesh=output_tensor_mesh,
        allgather_output_tensor_mesh=allgather_output_tensor_mesh,
        semaphores=allreduce_semaphores,
        gather_semaphore_addrs=gather_semaphore_addrs,
        ccl_sync_semaphore_addr=ccl_sync_semaphore_addr,
        ccl_sync2_semaphore_addr=ccl_sync2_semaphore_addr,
        allgather_handoff_semaphore_addr=allgather_semaphore_addrs[0],
        allgather_recv_semaphore_addr=allgather_semaphore_addrs[1],
        allgather_transport_risc=allgather_transport_risc,
        allgather_open_after_allreduce=allgather_open == "late",
        allgather_use_config=True,
        allgather_gather_enabled=allgather_enabled,
        allreduce_cluster_axis=1,
        allgather_cluster_axis=0,
        num_links=allreduce_num_links,
        chunk_num_tiles=1,
        gather_dst_num_tiles=gather_dst_num_tiles,
        gather_sender_grid=gather_sender_grid,
        gather_receiver_core=gather_receiver_core,
        allreduce_receiver_core=allreduce_receiver_core,
        gather_reduce_data_size_bytes=gather_dst_num_tiles * tile_32x32_size,
        gather_reduce_src_num_pages=gather_dst_num_tiles,
    )
    ttnn.synchronize_device(submesh)

    row_reduced = []
    for row in range(mesh_rows):
        col_reduced = []
        for col in range(mesh_cols):
            source = source_per_device[row * mesh_cols + col]
            col_reduced.append((source[:, :32].float() + source[:, 32:64].float()).bfloat16())
        row_reduced.append(sum(t.float() for t in col_reduced).bfloat16())

    output_tensor_torch = ttnn.to_torch(
        ttnn_result,
        mesh_composer=ttnn.ConcatMesh2dToTensor(submesh, mesh_shape=submesh.shape, dims=(0, 1)),
    )
    compare_cols = slice(2, None)

    if allgather_enabled:
        expected_allgather = torch.cat(row_reduced, dim=0)
        for row in range(mesh_rows):
            for col in range(mesh_cols):
                received = output_tensor_torch[
                    row * allgather_rows : (row + 1) * allgather_rows,
                    col * 32 : (col + 1) * 32,
                ]
                device_idx = row * mesh_cols + col
                assert torch.allclose(
                    received[:, compare_cols], expected_allgather[:, compare_cols], rtol=1e-2, atol=1e-2
                ), f"All-gather output mismatch for device {device_idx} in 4x2 smoke"
    else:
        for row in range(mesh_rows):
            for col in range(mesh_cols):
                received = output_tensor_torch[
                    row * reduced_rows : (row + 1) * reduced_rows,
                    col * 32 : (col + 1) * 32,
                ]
                device_idx = row * mesh_cols + col
                assert torch.allclose(
                    received[:, compare_cols], row_reduced[row][:, compare_cols], rtol=1e-2, atol=1e-2
                ), f"All-reduce output mismatch for device {device_idx} in 4x2 smoke"
