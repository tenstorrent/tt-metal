# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import os
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


def run_consolidate_cache_impl(
    mesh_device,
    shape,
    dtype,
    priority_tensors,
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
        ### Consolidate cache ###
        ############################################################
        logger.info(f"Performing consolidate cache")
        cache_d1 = torch.rand(shape).bfloat16()
        cache_d2 = torch.rand(shape).bfloat16()
        all_cache_tensors = [cache_d1, cache_d2]
        other_priority_tensors = priority_tensors[::-1]

        input_tensor_mesh = create_multi_device_tensors(
            all_cache_tensors, mesh_device, mem_config, ttnn.TILE_LAYOUT, ttnn.bfloat16
        )
        priority_tensor_mesh = create_multi_device_tensors(
            priority_tensors, mesh_device, mem_config, ttnn.TILE_LAYOUT, ttnn.uint32
        )
        other_priority_tensor_mesh = create_multi_device_tensors(
            other_priority_tensors, mesh_device, mem_config, ttnn.TILE_LAYOUT, ttnn.uint32
        )

        for _ in range(10):
            consolidated_cache_mesh = ttnn.experimental.consolidate_cache(
                input_tensor_mesh,
                priority_tensor_mesh,
                other_priority_tensor_mesh,
                multi_device_global_semaphore=all_gather_semaphore_handles,
                num_links=num_links,
            )
        ttnn.synchronize_devices(mesh_device, sub_device_ids=sub_device_stall_group)

        consolidated_cache_tensors = read_multi_device_tensor(consolidated_cache_mesh)

        # checking consolidated cache tensors
        min_p_device0 = torch.min(priority_tensors[0].squeeze())
        min_p_device1 = torch.min(priority_tensors[1].squeeze())
        consolidated_cache_idx = 0 if min_p_device0 > min_p_device1 else 1

        comp_equal(consolidated_cache_tensors[0], all_cache_tensors[consolidated_cache_idx])
        comp_equal(consolidated_cache_tensors[1], all_cache_tensors[consolidated_cache_idx])

        logger.info(f"Done consolidate cache")
        ############################################################

    except Exception as e:
        logger.error(f"Error during consolidate cache: {e}")
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
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "num_links",
    [1, 2],
)
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 128),
    ],
)
@pytest.mark.parametrize(
    "priority_tensors",
    [
        [torch.ones(1, 1, 32, 32), torch.zeros(1, 1, 32, 32)],
        [torch.zeros(1, 1, 32, 32), torch.ones(1, 1, 32, 32)],
    ],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("enable_async", [False])
def test_consolidate_cache(
    mesh_device,
    shape,
    dtype,
    num_links,
    priority_tensors,
    use_program_cache,
    function_level_defaults,
    enable_async,
):
    run_consolidate_cache_impl(
        mesh_device,
        shape,
        dtype,
        priority_tensors,
        num_links,
        enable_async=enable_async,
    )
