# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
import numpy as np
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull
from tests.ttnn.unit_tests.operations.ccl.test_reduce_scatter_async import (
    create_and_load_sub_device_manager_with_fabric_interface,
    teardown_fabric_interface,
)
from tests.ttnn.unit_tests.operations.speculative_execution.test_speculative_flash_decode import (
    nearest_n,
    nearest_pow_2,
    num_to_corerange,
    fa_rand,
    get_speculative_flash_decode_expected,
    prepare_test_config_and_data,
)
from tests.ttnn.unit_tests.operations.speculative_execution.test_speculative_flash_decode_ccl import (
    create_multi_device_tensors,
    read_multi_device_tensor,
)


def run_swap_tensor_impl(
    mesh_device,
    shape,
    layout,
    dtype,
    num_links,
    enable_async=False,
):
    ############################################################
    # Setup and Defines
    ############################################################
    num_devices = 2
    # Use Async mode based on test input config
    mesh_device.enable_async(enable_async)
    if enable_async:
        logger.info(f"Using Async Mode for All Gather Op Dispatch")
    enable_persistent_fabric = True
    create_persistent_fabric = True
    teardown_persistent_fabric = True
    ############################################################

    ############################################################
    ### Persistent fabric and ccl setup ###
    ############################################################
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
    if create_persistent_fabric:
        mesh_sub_device_manager_id = create_and_load_sub_device_manager_with_fabric_interface(
            mesh_device, [worker_sub_device], 0, 0, enable_persistent_fabric
        )
        mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    ############################################################

    try:
        ############################################################
        ### Warmup all gather ccl ###
        ############################################################
        logger.info(f"Performing warmup all gather ccl")
        output_shape = [1, 1, 64, 128]
        dim = 2
        logger.info(f"Output shape: {output_shape}")
        logger.info(f"dim: {dim}")
        mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)

        output_tensor = torch.rand(output_shape).bfloat16()

        input_tensors = torch.chunk(output_tensor, num_devices, dim)
        input_tensor_mesh = create_multi_device_tensors(
            input_tensors, mesh_device, mem_config, ttnn.TILE_LAYOUT, ttnn.bfloat16
        )

        # create global semaphore handles
        all_gather_semaphore_handles = ttnn.create_global_semaphore_with_same_address(
            mesh_device, ccl_sub_device_crs, 0
        )

        tt_out_tensor = ttnn.experimental.all_gather_async(
            input_tensor_mesh,
            dim,
            multi_device_global_semaphore=all_gather_semaphore_handles,
            num_links=1,
            memory_config=mem_config,
            topology=ttnn.Topology.Ring,
            subdevice_id=worker_sub_device_id,
            enable_persistent_fabric_mode=enable_persistent_fabric,
        )

        logger.info(f"Waiting for op")
        ttnn.synchronize_devices(mesh_device, sub_device_ids=sub_device_stall_group)
        logger.info(f"Done iteration")

        for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
            tt_output_tensor = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
            logger.info(f"Checking for device {t.device().id()}")
            eq, output = comp_equal(tt_output_tensor, output_tensor)
            assert eq, f"{i} FAILED: {output}"
        logger.info(f"Done warmup all gather ccl")
        ############################################################

        ############################################################
        ### Swap tensor ###
        ############################################################
        logger.info(f"Performing swap tensor")
        swap_tensor_d1 = torch.randint(0, 100, shape)
        swap_tensor_d2 = torch.randint(0, 100, shape)

        input_tensor_mesh = create_multi_device_tensors(
            [swap_tensor_d1, swap_tensor_d2], mesh_device, mem_config, layout, ttnn.uint32
        )

        if layout == ttnn.ROW_MAJOR_LAYOUT:
            input_tensor_mesh = ttnn.to_layout(input_tensor_mesh, ttnn.TILE_LAYOUT)

        swapped_tensor_mesh = ttnn.experimental.swap_tensor(
            input_tensor_mesh, multi_device_global_semaphore=all_gather_semaphore_handles, num_links=num_links
        )
        ttnn.synchronize_devices(mesh_device, sub_device_ids=sub_device_stall_group)

        if layout == ttnn.ROW_MAJOR_LAYOUT:
            input_tensor_mesh = ttnn.to_layout(input_tensor_mesh, ttnn.ROW_MAJOR_LAYOUT)
            swapped_tensor_mesh = ttnn.to_layout(swapped_tensor_mesh, ttnn.ROW_MAJOR_LAYOUT)

        swapped_tensors = read_multi_device_tensor(swapped_tensor_mesh)

        # checking swapped tensors
        comp_equal(swapped_tensors[0], swap_tensor_d2)
        comp_equal(swapped_tensors[1], swap_tensor_d1)

        # make sure input tensors are not modified
        before_swap_tensors = read_multi_device_tensor(input_tensor_mesh)
        comp_equal(before_swap_tensors[0], swap_tensor_d1)
        comp_equal(before_swap_tensors[1], swap_tensor_d2)

        logger.info(f"Done swap tensor")
        ############################################################

    except Exception as e:
        logger.error(f"Error during swap tensor: {e}")
        raise e
    finally:
        ############################################################
        ### Teardown persistent fabric ###
        ############################################################
        if enable_persistent_fabric and teardown_persistent_fabric:
            mesh_device.reset_sub_device_stall_group()
            teardown_fabric_interface(mesh_device)
        ############################################################


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.uint32,
        # ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "num_links",
    [1, 2],
)
@pytest.mark.parametrize(
    "shape, layout",
    [
        [(1, 1, 32, 32), ttnn.TILE_LAYOUT],
        # [(1, 1, 1, 32), ttnn.ROW_MAJOR_LAYOUT],
        # [(1, 1, 4, 1), ttnn.ROW_MAJOR_LAYOUT],
    ],
)
@pytest.mark.parametrize("enable_async", [False])
def test_swap_tensor(
    t3k_mesh_device,
    shape,
    layout,
    dtype,
    num_links,
    use_program_cache,
    function_level_defaults,
    enable_async,
):
    run_swap_tensor_impl(
        t3k_mesh_device,
        shape,
        layout,
        dtype,
        num_links,
        enable_async=enable_async,
    )
