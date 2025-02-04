# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import math
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull
from tests.ttnn.unit_tests.operations.ccl.test_ccl_common import (
    create_and_load_sub_device_manager_with_fabric_interface,
    teardown_fabric_interface,
    create_global_semaphore_with_same_address,
)

from tests.tt_eager.python_api_testing.unit_testing.misc.test_matmul_1d_gather_in0 import (
    num_cores_to_rectangle_grid,
    round_up,
)


def run_all_reduce_impl(
    mesh_device,
    output_shape,
    cluster_axis,
    input_dtype,
    num_links,
    input_num_cores,
    output_num_cores,
    num_iters=1,
    enable_async=False,
    trace_mode=False,
):
    cluster_shape = (8, 4)

    create_persistent_fabric = True
    teardown_persistent_fabric = True
    enable_persistent_fabric = True
    if num_iters < 1:
        pytest.fail("num_iters must be >= 1")
    # Use Async mode based on test input config
    mesh_device.enable_async(enable_async)

    if enable_async:
        logger.info(f"Using Async Mode for All Gather Op Dispatch")

    ##################################
    ##### Set up fabric stuff
    ##################################
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
            mesh_device, [worker_sub_device], 0, 0, enable_persistent_fabric
        )
        mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    # create global semaphore handles
    ccl_semaphore_handles = create_global_semaphore_with_same_address(mesh_device, ccl_sub_device_crs, 0)

    logger.info(f"Output shape: {output_shape}")

    try:
        ##################################
        ##### Set up input tensors/configs
        ##################################

        ##### FF2 Case #####
        core_offset = 1
        M, N = output_shape[2:]
        N_per_shard = round_up(math.ceil(N / input_num_cores), ttnn.TILE_SIZE)
        output_N_per_shard = round_up(math.ceil(N / output_num_cores), ttnn.TILE_SIZE)
        input_shape = [*cluster_shape, M, N]

        CORE_RANGE = [(x, y) for y in range(compute_grid_size.y) for x in range(compute_grid_size.x)]
        core_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(x, y),
                    ttnn.CoreCoord(x, y),
                )
                for x, y in CORE_RANGE[core_offset : core_offset + input_num_cores]
            ]
        )
        input_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                core_range_set,
                [M, N_per_shard],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        output_core_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(x, y),
                    ttnn.CoreCoord(x, y),
                )
                for x, y in CORE_RANGE[core_offset : core_offset + output_num_cores]
            ]
        )
        output_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                output_core_range_set,
                [M, output_N_per_shard],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        input_tensor = torch.randn(input_shape)
        tt_input_tensor = ttnn.from_torch(
            input_tensor,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=input_dtype,
            memory_config=input_mem_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=cluster_shape),
        )

        # All-Reduce Golden
        output_tensor_goldens_list = [torch.sum(input_tensor, dim=cluster_axis) for _ in range(num_iters)]

        ##################################
        ##### Run the op
        ##################################

        def run_op():
            outs = []
            for i in range(num_iters):
                out = ttnn.experimental.all_reduce_async(
                    tt_input_tensor,
                    cluster_axis=cluster_axis,
                    mesh_device=mesh_device,
                    multi_device_global_semaphore=ccl_semaphore_handles,
                    memory_config=output_mem_config,
                    topology=ttnn.Topology.Linear,
                    num_links=num_links,
                    subdevice_id=worker_sub_device_id,
                )
                for d in mesh_device.get_devices():
                    ttnn.synchronize_device(d)
                outs.append(out)

            return outs

        # ##### Compile Model #####
        # logger.info("Compiling model")
        # tt_outs = run_op()

        # ##### Capture Trace #####
        # logger.info("Capturing trace")

        # trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        # tt_outs = run_op()
        # ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

        # ##### Run Trace #####
        # logger.info("Running trace")
        # ttnn.execute_trace(mesh_device, trace_id, blocking=False)

        tt_outs = run_op()

        ##################################
        ##### Validation
        ##################################
        for tensor_index in range(len(tt_outs)):
            tt_out_tensor = tt_outs[tensor_index]
            output_tensor = output_tensor_goldens_list[tensor_index]
            for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
                # get_device_tensors returns row major, so we need to select the correct golden tensor
                if cluster_axis == 0:
                    output_tensor_ = output_tensor[i % cluster_shape[not (cluster_axis)]].unsqueeze(0).unsqueeze(0)
                else:
                    output_tensor_ = output_tensor[i // cluster_shape[cluster_axis]].unsqueeze(0).unsqueeze(0)

                tt_output_tensor = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
                logger.info(f"Checking for device {t.device().id()}")

                if input_dtype == ttnn.bfloat16:
                    eq, output = comp_pcc(tt_output_tensor, output_tensor_)
                else:
                    eq, output = comp_pcc(tt_output_tensor, output_tensor_)
                logger.info(f"PCC output for {i} is: {output}")
                assert eq, f"{i} FAILED: {output}"
    finally:
        if enable_persistent_fabric and teardown_persistent_fabric:
            mesh_device.reset_sub_device_stall_group()
            teardown_fabric_interface(mesh_device)


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "output_shape, cluster_axis, num_links, input_num_cores, output_num_cores",
    [
        ([1, 1, 32, 1536], 1, 1, 24, 8),  # QKV all reduce
        ([1, 1, 32, 3840], 1, 1, 24, 24),  # FF1 all reduce
        # TODO: Use unpadded shapes and output to 16
        ([1, 1, 32, 2304], 0, 1, 24, 8),  # FF2/DO all reduce
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize("num_iters", [1])
@pytest.mark.parametrize("enable_async", [True])
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 23887872}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
def test_all_reduce(
    mesh_device,
    output_shape,
    cluster_axis,
    input_dtype,
    num_links,
    input_num_cores,
    output_num_cores,
    num_iters,
    enable_async,
    use_program_cache,
    function_level_defaults,
):
    run_all_reduce_impl(
        mesh_device,
        output_shape,
        cluster_axis,
        input_dtype,
        num_links,
        input_num_cores,
        output_num_cores,
        num_iters=num_iters,
        enable_async=enable_async,
    )
