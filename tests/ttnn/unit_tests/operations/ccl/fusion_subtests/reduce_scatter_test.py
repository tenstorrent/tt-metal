# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.utility_functions import skip_for_grayskull
from tests.ttnn.unit_tests.operations.ccl.test_ccl_common import (
    create_and_load_sub_device_manager_with_fabric_interface,
    teardown_fabric_interface,
    create_global_semaphore_with_same_address,
)


def run_with_trace(
    mesh_device,
    all_gather_topology,
    input_tensor_mesh,
    dim,
    num_links,
    output_mem_config,
    enable_persistent_fabric,
    multi_device_global_semaphore,
    num_iter=20,
    subdevice_id=None,
):
    # Compile Run
    logger.info("Compiling model")
    tt_out_tensor = ttnn.experimental.all_gather_async(
        input_tensor_mesh,
        dim,
        multi_device_global_semaphore=multi_device_global_semaphore,
        num_links=num_links,
        memory_config=output_mem_config,
        topology=all_gather_topology,
        subdevice_id=subdevice_id,
        enable_persistent_fabric_mode=enable_persistent_fabric,
    )
    ttnn.synchronize_device(mesh_device)

    # Capture trace
    logger.info("Capturing trace")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for i in range(num_iter):
        tt_out_tensor = ttnn.experimental.all_gather_async(
            input_tensor_mesh,
            dim,
            multi_device_global_semaphore=multi_device_global_semaphore,
            num_links=num_links,
            memory_config=output_mem_config,
            topology=all_gather_topology,
            subdevice_id=subdevice_id,
            enable_persistent_fabric_mode=enable_persistent_fabric,
        )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Run the op
    logger.info("Starting Trace perf test...")
    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.synchronize_device(mesh_device)

    return tt_out_tensor


def run_reduce_scatter_impl(
    mesh_device,
    num_devices,
    output_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    use_program_cache,
    function_level_defaults,
    input_shard_shape,
    input_shard_grid,
    all_gather_topology,
    num_iters=1,
    enable_async=False,
    trace_mode=False,
    output_shard_shape=None,
    output_shard_grid=None,
    tensor_mem_layout=None,
):
    enable_persistent_fabric = True
    if num_iters < 1:
        pytest.fail("num_iters must be >= 1")
    # Use Async mode based on test input config
    mesh_device.enable_async(enable_async)

    if enable_async:
        logger.info(f"Using Async Mode for All Gather Op Dispatch")

    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice(
        [
            ccl_sub_device_crs,
        ]
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
    mesh_sub_device_manager_id = create_and_load_sub_device_manager_with_fabric_interface(
        mesh_device,
        [worker_sub_device],
        0,
        0,
        enable_persistent_fabric,
        wrap_fabric_around_mesh=True,
    )
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    # create global semaphore handles
    ccl_semaphore_handles = [
        create_global_semaphore_with_same_address(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)
    ]

    ### For sharded all gather only
    if bool(input_shard_shape) != bool(input_shard_grid) and bool(tensor_mem_layout) != bool(input_shard_grid):
        pytest.fail(
            "Both input_shard_shape, shard_grid, and tensor_mem_layout must be provided together or all must be None"
        )
    if input_shard_shape and input_shard_grid:
        input_shard_spec = ttnn.ShardSpec(
            input_shard_grid,
            input_shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        input_mem_config = ttnn.MemoryConfig(
            tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=input_shard_spec
        )
        if output_shard_shape is None:
            assert (
                output_shard_grid is None
            ), "output_shard_grid must not be provided if output_shard_shape is not provided"
            output_shard_shape = list(input_shard_shape)
            if dim == len(output_shape) - 1:
                output_shard_shape[1] *= num_devices
            else:
                output_shard_shape[0] *= num_devices
            output_shard_spec = ttnn.ShardSpec(
                input_shard_grid,
                output_shard_shape,
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            output_mem_config = ttnn.MemoryConfig(
                tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=output_shard_spec
            )
        else:
            assert output_shard_grid is not None, "output_shard_grid must be provided if output_shard_shape is provided"
            output_shard_spec = ttnn.ShardSpec(
                output_shard_grid,
                output_shard_shape,
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            output_mem_config = ttnn.MemoryConfig(
                tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=output_shard_spec
            )
    ###

    input_tensor_mesh_list = []
    output_tensor_goldens_list = []

    for i in range(num_iters):
        output_tensor = torch.rand(output_shape).bfloat16()
        output_tensor_goldens_list.append(output_tensor)
        input_tensors = torch.chunk(output_tensor, num_devices, dim)
        tt_input_tensors = []
        for i, t in enumerate(input_tensors):
            tt_input_tensors.append(
                ttnn.Tensor(t, input_dtype).to(layout).to(mesh_device.get_devices()[i], input_mem_config)
            )
            logger.info(f"using device {mesh_device.get_devices()[i].id()}")

        input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors)

        input_tensor_mesh_list.append(input_tensor_mesh)

    tt_out_tensor_list = []
    if trace_mode:
        tt_out_tensor = run_with_trace(
            mesh_device,
            all_gather_topology,
            input_tensor_mesh_list[0],
            dim,
            num_links,
            output_mem_config,
            enable_persistent_fabric,
            multi_device_global_semaphore=ccl_semaphore_handles[0],
            num_iter=num_iters,
            subdevice_id=worker_sub_device_id,
        )
        tt_out_tensor_list.append(tt_out_tensor)
    else:
        for i in range(num_iters):
            tt_out_tensor = ttnn.experimental.llama_reduce_scatter(
                input_tensor_mesh_list[i],
                dim,
                multi_device_global_semaphore=ccl_semaphore_handles[i],
                num_links=num_links,
                memory_config=output_mem_config,
                topology=all_gather_topology,
                subdevice_id=worker_sub_device_id,
                enable_persistent_fabric_mode=enable_persistent_fabric,
            )
            tt_out_tensor_list.append(tt_out_tensor)

        logger.info(f"Waiting for op")
        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
        logger.info(f"Done op")

    passed = True
    for tensor_index in range(len(tt_out_tensor_list)):
        tt_out_tensor = tt_out_tensor_list[tensor_index]
        output_tensor = output_tensor_goldens_list[tensor_index]
        for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
            tt_output_tensor = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
            logger.info(f"Checking for device {t.device().id()}")

            if input_dtype == ttnn.bfloat16:
                eq, output = comp_equal(tt_output_tensor, output_tensor)
            else:
                eq, output = comp_pcc(tt_output_tensor, output_tensor)
            if not eq:
                logger.error(f"output mismatch for tensor {i}")
                passed = False

    for i in range(num_devices):
        assert (
            mesh_device.get_devices()[i].num_program_cache_entries() == 1
            or mesh_device.get_devices()[i].num_program_cache_entries() == num_iters
        ), f"Device {i} has {mesh_device.get_devices()[i].num_program_cache_entries()} program cache entries"

    if enable_persistent_fabric:
        mesh_device.reset_sub_device_stall_group()
        teardown_fabric_interface(mesh_device)

    if not passed:
        assert eq, f"{i} FAILED: {output}"


def test_fabric_reduce_scatter(n300_mesh_device):
    torch.manual_seed(2005)
    dim = 3
    shard_height = 32
    shard_width = 160
    num_devices = n300_mesh_device.get_num_devices()
    num_cores = 12
    torch_input_tensors = []

    for _ in range(num_devices):
        for _ in range(num_cores):
            for _ in range(shard_width // 32):
                torch_input_tensors.append(torch.rand(1, 1, shard_height, 32))

    input = torch.cat(torch_input_tensors, dim=3)
    print("input.shape", input.shape)
    intermediate_outputs = torch.chunk(input, chunks=num_devices, dim=3)
    output = torch.zeros(intermediate_outputs[0].shape)
    for i in range(0, len(intermediate_outputs)):
        output += intermediate_outputs[i]

    n300_mesh_device.enable_async(True)
    compute_grid_size = n300_mesh_device.compute_with_storage_grid_size()
    sharded_mem_config = ttnn.create_sharded_memory_config(
        (32, 160),
        core_grid=ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, row_wise=True),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_input = ttnn.from_torch(
        input,
        mesh_mapper=ttnn.ShardTensorToMesh(n300_mesh_device, dim),
        device=n300_mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_mem_config,
        dtype=ttnn.bfloat8_b,
    )

    enable_persistent_fabric = True
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice(
        [
            ccl_sub_device_crs,
        ]
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
    mesh_sub_device_manager_id = create_and_load_sub_device_manager_with_fabric_interface(
        n300_mesh_device,
        [worker_sub_device],
        0,
        0,
        enable_persistent_fabric,
        wrap_fabric_around_mesh=True,
    )
    n300_mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    # create global semaphore handles
    ccl_semaphore_handles = [
        create_global_semaphore_with_same_address(n300_mesh_device, ccl_sub_device_crs, 0) for _ in range(1)
    ]
    n300_mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    # create global semaphore handles
    ccl_semaphore_handles = [create_global_semaphore_with_same_address(n300_mesh_device, ccl_sub_device_crs, 0)]
    tt_out_tensor = ttnn.experimental.llama_reduce_scatter(
        tt_input, dim, ccl_semaphore_handles[0], worker_sub_device_id, cluster_axis=1, num_links=1
    )
    ttnn.synchronize_device(n300_mesh_device, sub_device_ids=sub_device_stall_group)
    tt_torch_tensor = ttnn.to_torch(tt_out_tensor, mesh_composer=ttnn.ConcatMeshToTensor(n300_mesh_device, dim))
    torch.set_printoptions(threshold=10000)
    print("TT tensor")
    print(tt_torch_tensor[:, :, 0, 0::32])
    print("Torch tensor")
    print(output[:, :, 0, 0::32])

    eq, output_results = comp_pcc(tt_torch_tensor, output)

    print(f"PCC: {output_results}")

    n300_mesh_device.reset_sub_device_stall_group()
    teardown_fabric_interface(n300_mesh_device)
    # Assuming tt_torch_tensor and output are your tensors
    # Ensure they are on the same device and have the same shape

    # Compute the absolute differences
    differences = torch.abs(tt_torch_tensor - output)

    # Find the maximum difference
    max_difference = torch.max(differences)

    # Find the first index where the maximum difference occurs
    first_max_diff_index = None
    for index in torch.nonzero(differences == max_difference, as_tuple=False):
        first_max_diff_index = tuple(index.tolist())
        break
    print("Tenstorrent", tt_torch_tensor[first_max_diff_index])
    print("Torch", output[first_max_diff_index])
    if first_max_diff_index is not None:
        print(f"First index with maximum difference: {first_max_diff_index}")
        print(f"Index at tile: {first_max_diff_index[3]/32}")
        print(f"Maximum difference: {max_difference.item()}")
        print(f"tt_torch_tensor value: {tt_torch_tensor[first_max_diff_index]}")
        print(f"output value: {output[first_max_diff_index]}")
    else:
        print("No mismatches found.")

    print(f"Maximum difference: {max_difference.item()}")
    assert eq, f"FAILED: {output_results}"
