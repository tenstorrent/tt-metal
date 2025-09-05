# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
import math
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from models.utility_functions import skip_for_grayskull


def run_all_reduce_test(
    mesh_device,
    num_devices,
    per_chip_output_shape,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    function_level_defaults,
    num_iters=1,
    topology=ttnn.Topology.Linear,
):
    if len(mesh_device.get_device_ids()) < num_devices:
        pytest.skip(
            f"Not enough devices on machine to implement test case. Wanted {num_devices} but found {len(mesh_device.get_device_ids())}"
        )

    ttnn.synchronize_device(mesh_device)

    sub_device_stall_group = []
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
    # create global semaphore handles
    from_remote_semaphore_handles = ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0)
    to_remote_semaphore_handles = ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0)
    gather_semaphore_handles = ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0)

    debug = False

    logger.info(f"Per chip output shape: {per_chip_output_shape}, devices: {num_devices}")
    # Generate input tensors

    canonical_input_tensors = []
    input_tensors = []

    numel = math.prod(per_chip_output_shape)
    if debug:
        input_tensors[-1] = torch.arange(numel).reshape(per_chip_output_shape).bfloat16()
    for i in range(num_devices):
        input_tensor = torch.rand(per_chip_output_shape).bfloat16()
        canonical_input_tensors.append(input_tensor)
        input_tensor = input_tensor.view(1, -1, input_tensor.shape[2], input_tensor.shape[3])
        input_tensors.append(input_tensor)

    unchunked_input_tensor = torch.cat(input_tensors)

    assert len(canonical_input_tensors) == num_devices
    input_tensor_mesh = ttnn.from_torch(
        torch.cat(canonical_input_tensors),
        dtype=input_dtype,
        layout=layout,
        device=mesh_device,
        memory_config=mem_config,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh_device,
            ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementShard(0)], ttnn.MeshShape(1, num_devices)),
        ),
    )
    # Run the op
    for i in range(num_iters):
        output_tensor_mesh = ttnn.experimental.all_reduce_async(
            input_tensor_mesh,
            from_remote_multi_device_global_semaphore=from_remote_semaphore_handles,
            to_remote_multi_device_global_semaphore=to_remote_semaphore_handles,
            gather_multi_device_global_semaphore=gather_semaphore_handles,
            math_op=math_op,
            num_links=num_links,
            memory_config=mem_config,
            topology=topology,
            subdevice_id=worker_sub_device_id,
        )
        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
    ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)

    mesh_device.reset_sub_device_stall_group()

    tt_out_tensors = ttnn.get_device_tensors(output_tensor_mesh)
    logger.info(f"Compare")
    golden_canonical_out_tensor = torch.sum(unchunked_input_tensor, 0, keepdim=True)
    golden_canonical_out_tensor = golden_canonical_out_tensor.view(per_chip_output_shape)
    # Compare
    mismatch = False
    for i, t in enumerate(tt_out_tensors):
        tt_output_tensor = ttnn.to_torch(t)

        eq, output = comp_pcc(tt_output_tensor, golden_canonical_out_tensor)
        mismatch = mismatch or not eq
        if not eq:
            logger.error(f"output mismatch for tensor {i}. Mesh device ID: {mesh_device.get_device_ids()[i]}")
            if debug:
                for w in range(tt_output_tensor.shape[0]):
                    for z in range(tt_output_tensor.shape[1]):
                        for y in range(tt_output_tensor.shape[2]):
                            for x in range(tt_output_tensor.shape[3]):
                                if tt_output_tensor[w, z, y, x] != golden_canonical_out_tensor[w, z, y, x]:
                                    logger.error(
                                        f"mismatch at {w}, {z}, {y}, {x}: {tt_output_tensor[w, z, y, x]} != {golden_canonical_out_tensor[w, z, y, x]}"
                                    )

        else:
            logger.info(f"output match for tensor {i}")
    assert not mismatch, f"{i} FAILED: {output}"


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (4, 1),
        # (8, 1), # skipped as 8 devices result in hang in all gather
    ],
)
@pytest.mark.parametrize(
    "per_chip_output_shape",
    [
        ([1, 1, 32, 4096]),
        ([1, 1, 32, 8192]),
        ([1, 1, 32, 1024]),
        ([1, 1, 32, 2048]),
        ([1, 1, 4096, 32]),
        # ([1, 1, 8192, 32]), # skipped as it hangs in reduce scatter part.
        ([1, 1, 1024, 32]),
        ([1, 1, 2048, 32]),
        ([4, 1, 32, 4096]),
        ([8, 1, 32, 1024]),
        ([1, 4, 1024, 32]),
        ([2, 4, 2048, 32]),
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_ring_all_reduce_post_commit(
    t3k_mesh_device,
    num_devices,
    per_chip_output_shape,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    function_level_defaults,
    num_iters=2,
):
    run_all_reduce_test(
        t3k_mesh_device,
        num_devices,
        per_chip_output_shape,
        num_links,
        math_op,
        input_dtype,
        layout,
        mem_config,
        function_level_defaults,
        num_iters=num_iters,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (2, 1),
    ],
)
@pytest.mark.parametrize(
    "per_chip_output_shape",
    [
        ([2, 2, 64, 64]),
        ([1, 1, 64, 64]),
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
    ],
)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_ring_all_reduce_post_commit_2chip(
    t3k_mesh_device,
    num_devices,
    per_chip_output_shape,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    function_level_defaults,
    num_iters=2,
):
    run_all_reduce_test(
        t3k_mesh_device,
        num_devices,
        per_chip_output_shape,
        num_links,
        math_op,
        input_dtype,
        layout,
        mem_config,
        function_level_defaults,
        num_iters=num_iters,
        topology=ttnn.Topology.Linear,
    )


def run_all_reduce_with_mesh_tensor_along_row(
    mesh_device,
    num_devices_per_line,
    per_chip_output_shape,
    num_links,
    math_op,
    input_dtype,
    layout,
    buffer_type: ttnn.BufferType,
    function_level_defaults,
    num_all_reduce_instances: int = 1,
    num_iters: int = 1,
    cluster_axis: int = 0,
):
    mem_config = ttnn.MemoryConfig(buffer_type=buffer_type)

    ttnn.synchronize_device(mesh_device)

    sub_device_stall_group = []
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
    # create global semaphore handles
    from_remote_semaphore_handles = ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0)
    to_remote_semaphore_handles = ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0)
    gather_semaphore_handles = ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0)

    try:
        debug = False

        logger.info(f"Per chip output shape: {per_chip_output_shape}, devices: {num_devices_per_line}")
        # Generate input tensors
        input_tensors = []
        tt_input_tensors = []

        numel = math.prod(per_chip_output_shape)
        if debug:
            input_tensors[-1] = torch.arange(numel).reshape(per_chip_output_shape).bfloat16()
        for i in range(num_devices_per_line):
            input_tensor = torch.rand(per_chip_output_shape).bfloat16()
            tt_input_tensors.append(input_tensor.clone())
            input_tensor = input_tensor.view(1, -1, input_tensor.shape[2], input_tensor.shape[3])
            input_tensors.append(input_tensor)

        unchunked_input_tensor = torch.cat(input_tensors)

        shard_dims = (0, 1) if cluster_axis == 0 else (1, 0)
        mesh_shape = (
            (num_devices_per_line, num_all_reduce_instances)
            if cluster_axis == 0
            else (num_all_reduce_instances, num_devices_per_line)
        )
        full_input_tensor_unfractured = torch.cat(tt_input_tensors)
        full_input_tensor_unfractured = [full_input_tensor_unfractured for _ in range(num_all_reduce_instances)]
        full_input_tensor_unfractured = torch.cat(full_input_tensor_unfractured, dim=1)

        ttnn_tensor = ttnn.from_torch(
            full_input_tensor_unfractured,
            dtype=input_dtype,
            device=mesh_device,
            layout=layout,
            memory_config=mem_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=shard_dims),
        )
        input_tensor_mesh = ttnn.to_device(ttnn_tensor, mesh_device)

        # Run the op
        for i in range(num_iters):
            output_tensor_mesh = ttnn.experimental.all_reduce_async(
                input_tensor_mesh,
                cluster_axis=cluster_axis,
                mesh_device=mesh_device,
                from_remote_multi_device_global_semaphore=from_remote_semaphore_handles,
                to_remote_multi_device_global_semaphore=to_remote_semaphore_handles,
                gather_multi_device_global_semaphore=gather_semaphore_handles,
                math_op=math_op,
                num_links=num_links,
                memory_config=mem_config,
                topology=ttnn.Topology.Linear,
                subdevice_id=worker_sub_device_id,
            )
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
    except Exception as e:
        raise e
    finally:
        mesh_device.reset_sub_device_stall_group()

    tt_out_tensors = ttnn.get_device_tensors(output_tensor_mesh)
    logger.info(f"Compare")
    golden_canonical_out_tensor = torch.sum(unchunked_input_tensor, 0, keepdim=True)
    golden_canonical_out_tensor = golden_canonical_out_tensor.view(per_chip_output_shape)

    # Compare
    mismatch = False
    for i, t in enumerate(tt_out_tensors):
        tt_output_tensor = ttnn.to_torch(t)

        eq, output = comp_pcc(tt_output_tensor, golden_canonical_out_tensor)
        mismatch = mismatch or not eq
        if not eq:
            logger.error(f"output mismatch for tensor {i}. Mesh device ID: {mesh_device.get_device_ids()[i]}")
            if debug:
                for w in range(tt_output_tensor.shape[0]):
                    for z in range(tt_output_tensor.shape[1]):
                        for y in range(tt_output_tensor.shape[2]):
                            for x in range(tt_output_tensor.shape[3]):
                                if tt_output_tensor[w, z, y, x] != golden_canonical_out_tensor[w, z, y, x]:
                                    logger.error(
                                        f"mismatch at {w}, {z}, {y}, {x}: {tt_output_tensor[w, z, y, x]} != {golden_canonical_out_tensor[w, z, y, x]}"
                                    )

        else:
            logger.info(f"output match for tensor {i}")
    assert not mismatch, f"{i} FAILED: {output}"


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, per_chip_output_shape, layout",
    [
        (4, 2, [1, 4, 32, 2304], ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "buffer_type",
    [
        ttnn.BufferType.DRAM,
        ttnn.BufferType.L1,
    ],
)
@pytest.mark.parametrize("replication_factor", [8])  # 1, 8])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_line_all_reduce_on_TG_rows_post_commit(
    mesh_device,
    num_devices,
    per_chip_output_shape,
    num_links,
    math_op,
    input_dtype,
    layout,
    buffer_type,
    function_level_defaults,
    replication_factor,
    num_iters=16,
):
    if mesh_device.get_num_devices() != 32:
        pytest.skip("Not TG!")

    run_all_reduce_with_mesh_tensor_along_row(
        mesh_device,
        num_devices,
        per_chip_output_shape,
        num_links,
        math_op,
        input_dtype,
        layout,
        buffer_type,
        function_level_defaults,
        num_iters=num_iters,
        num_all_reduce_instances=replication_factor,
        cluster_axis=1,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, per_chip_output_shape, layout",
    [
        (8, 1, [1, 8, 32, 1280], ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "buffer_type",
    [
        ttnn.BufferType.DRAM,
    ],
)
@pytest.mark.parametrize("replication_factor", [4])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_line_all_reduce_on_TG_cols_post_commit(
    mesh_device,
    num_devices,
    per_chip_output_shape,
    num_links,
    math_op,
    input_dtype,
    layout,
    buffer_type,
    function_level_defaults,
    replication_factor,
    num_iters=16,
):
    if mesh_device.get_num_devices() != 32:
        pytest.skip("Not TG!")

    run_all_reduce_with_mesh_tensor_along_row(
        mesh_device,
        num_devices,
        per_chip_output_shape,
        num_links,
        math_op,
        input_dtype,
        layout,
        buffer_type,
        function_level_defaults,
        num_iters=num_iters,
        num_all_reduce_instances=replication_factor,
        cluster_axis=0,
    )
