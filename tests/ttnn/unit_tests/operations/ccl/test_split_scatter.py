# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull
from tests.ttnn.unit_tests.operations.ccl.test_ccl_common import (
    create_and_load_sub_device_manager_with_fabric_interface,
    teardown_fabric_interface,
    create_global_semaphore_with_same_address,
)


def is_unsupported_case(input_shape, dim, mem_config, num_devices, num_links, input_dtype, layout, tile):
    if layout == ttnn.ROW_MAJOR_LAYOUT and input_dtype == ttnn.bfloat8_b:
        return True, "Invalid combination"

    if input_shape[dim] % num_devices != 0:
        return True, "Unsupported test case"
    if tile != (32, 32) and input_dtype != ttnn.bfloat16:
        return True, "Tiny tile only supports bfloat16"

    ## Check that we can readback results
    fast_dispatch_page_size_limit = 55 * 1024
    elem_size = 2 if input_dtype == ttnn.bfloat16 else 1
    if layout == ttnn.ROW_MAJOR_LAYOUT and (input_shape[dim] * elem_size) > fast_dispatch_page_size_limit:
        # Fast dispatch currently can't breakup readback of large pages into multiple smaller pages and is
        # limited to ~55K pages.
        return True, "Fast dispatch can't support reading back this page size in one shot"

    # Check that we can fit in L1 (if L1 config)
    tensor_size_bytes = elem_size
    for i in input_shape:
        tensor_size_bytes *= i
    num_l1_banks = 64
    if mem_config.buffer_type == ttnn.BufferType.L1 and tensor_size_bytes > num_l1_banks * 50 * 1024:
        return True, "L1 buffer can't support large tensor sizes"

    # Check that each chip has a non-zero amount of data available
    if input_shape[dim] < num_devices:
        return (
            True,
            f"Input shape {input_shape} incompatible with {num_devices} on dim {dim} because some chips will have no tensor",
        )

    if (
        input_shape == [8, 8, 256, 384]
        and dim == 1
        and layout == ttnn.TILE_LAYOUT
        and (input_dtype == ttnn.bfloat8_b or tile != (32, 32))
    ):
        return True, "Known failure"

    return False, ""


def is_unsupported_case_n300(input_shape, dim, mem_config, num_devices, num_links, input_dtype, layout, tile):
    return is_unsupported_case(input_shape, dim, mem_config, num_devices, num_links, input_dtype, layout, tile)


def run_split_scatter_impl(
    mesh_device,
    num_devices,
    output_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    use_program_cache,
    function_level_defaults,
    split_scatter_topology,
    num_iters=1,
    enable_async=False,
    trace_mode=False,
    rand_tensor=True,
    mem_config=None,
    tensor_mem_layout=None,
    create_persistent_fabric=True,
    teardown_persistent_fabric=True,
    wrap_fabric_around_mesh=False,
):
    enable_persistent_fabric = True
    # torch.set_printoptions(profile="full")
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
    if create_persistent_fabric:
        mesh_sub_device_manager_id = create_and_load_sub_device_manager_with_fabric_interface(
            mesh_device,
            [worker_sub_device],
            0,
            0,
            enable_persistent_fabric,
            wrap_fabric_around_mesh=wrap_fabric_around_mesh,
        )
        mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    # create global semaphore handles
    ccl_semaphore_handles = [
        create_global_semaphore_with_same_address(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)
    ]

    logger.info(f"Output shape: {output_shape}")
    logger.info(f"dim: {dim}")

    assert mem_config is not None
    input_mem_config = mem_config
    output_mem_config = mem_config
    ###

    input_tensor_mesh_list = []
    output_tensor_goldens_list = []
    print("output_shape ", output_shape)
    for i in range(num_iters):
        if rand_tensor:
            output_tensor = torch.rand(output_shape).bfloat16()
        else:
            output_tensor = torch.zeros(output_shape)
            tile_id = 1
            for w in range(output_shape[0]):
                for z in range(output_shape[1]):
                    for y in range(0, output_shape[2], 32):
                        for x in range(0, output_shape[3], 32):
                            output_tensor[w, z, y : y + 32, x : x + 32] = tile_id
                            tile_id += 1

        output_tensor_goldens_list.append(output_tensor)

        input_tensors = torch.chunk(output_tensor, num_devices, dim)
        print("INP TENSORSS ", input_tensors)
        tt_input_tensors = []

        tt_input_tensors.append(ttnn.Tensor(output_tensor, input_dtype).to(layout).to(mesh_device, input_mem_config))

        logger.info(f"using device {mesh_device.get_devices()[0].id()}")

        input_tensor_mesh = tt_input_tensors
        print("inp mesh ", input_tensor_mesh)

        input_tensor_mesh_list.append(input_tensor_mesh)

    tt_out_tensor_list = []
    tt_out_tensor = ttnn.experimental.split_scatter(
        input_tensor_mesh[0],
        dim,
        multi_device_global_semaphore=ccl_semaphore_handles[i],
        num_links=num_links,
        memory_config=output_mem_config,
        topology=split_scatter_topology,
        subdevice_id=worker_sub_device_id,
        enable_persistent_fabric_mode=enable_persistent_fabric,
    )
    tt_out_tensor_list.append(tt_out_tensor)

    logger.info(f"Waiting for op")
    ttnn.synchronize_devices(mesh_device, sub_device_ids=sub_device_stall_group)
    logger.info(f"Done op")

    passed = True
    for tensor_index in range(len(tt_out_tensor_list)):
        tt_out_tensor = tt_out_tensor_list[tensor_index]
        output_tensor = input_tensors[tensor_index]

        print("input_tensors lens ", len(input_tensors))
        for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
            tt_output_tensor = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
            logger.info(f"Checking for device {t.device().id()}")
            print("COMPAREEEE loop ----------> ", i)
            print(tt_output_tensor)
            print(input_tensors[i])
            print("tt shape ", tt_output_tensor.shape)
            print("torch shape ", input_tensors[i].shape)
            if input_dtype == ttnn.bfloat16:
                eq, output = comp_equal(tt_output_tensor, input_tensors[i])
            else:
                eq, output = comp_pcc(tt_output_tensor, input_tensors[i])
            if not eq:
                logger.error(f"output mismatch for tensor {i}")
                passed = False

    for i in range(num_devices):
        assert (
            mesh_device.get_devices()[i].num_program_cache_entries() == 1
            or mesh_device.get_devices()[i].num_program_cache_entries() == num_iters
        ), f"Device {i} has {mesh_device.get_devices()[i].num_program_cache_entries()} program cache entries"

    if enable_persistent_fabric and teardown_persistent_fabric:
        mesh_device.reset_sub_device_stall_group()
        teardown_fabric_interface(mesh_device)

    if not passed:
        assert eq, f"{i} FAILED: {output}"


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, output_shape, dim, layout",
    [
        # (4, 1, [1, 1, 64, 512], 3, ttnn.TILE_LAYOUT),
        # (4, 1, [1, 1, 32, 32768], 3, ttnn.TILE_LAYOUT),
        # (4, 1, [1, 1, 2048, 16384], 3, ttnn.TILE_LAYOUT),
        (2, 1, [1, 1, 32, 256], 3, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        # ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
    ],
)
@pytest.mark.parametrize("num_iters", [1])
@pytest.mark.parametrize("enable_async", [True])
def test_split_scatter(
    n300_mesh_device,
    num_devices,
    output_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    use_program_cache,
    function_level_defaults,
    enable_async,
):
    run_split_scatter_impl(
        n300_mesh_device,
        num_devices,
        output_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        use_program_cache,
        function_level_defaults,
        split_scatter_topology=ttnn.Topology.Linear,
        num_iters=num_iters,
        enable_async=enable_async,
        rand_tensor=True,
        create_persistent_fabric=True,
        teardown_persistent_fabric=True,
        mem_config=mem_config,
    )
