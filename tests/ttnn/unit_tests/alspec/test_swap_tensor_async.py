# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import math
from time import time
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull
from tests.ttnn.unit_tests.operations.ccl.test_ccl_common import (
    create_global_semaphore_with_same_address,
)

from tests.tt_eager.python_api_testing.unit_testing.misc.test_matmul_1d_gather_in0 import (
    round_up,
)
from tracy import signpost


def check_mesh_tensor_alloc(tensor):
    device_tensors = ttnn.get_device_tensors(tensor)
    buffer_addr = device_tensors[0].buffer_address()

    if len(device_tensors) > 1:
        for i in range(1, len(device_tensors)):
            addr = device_tensors[i].buffer_address()
            if not addr == buffer_addr:
                return False
    return True


def run_swap_tensor_impl(
    mesh_device,
    output_shape,
    input_dtype,
    num_links,
    input_num_cores,
    output_num_cores,
    use_priority=False,
    num_iters=1,
    enable_async=False,
    trace_mode=False,
    validate_all=True,
):
    cluster_shape = tuple(mesh_device.shape)
    num_devices = mesh_device.get_num_devices()

    assert num_devices == 2, f"Only 2 devices are supported for swap tensor test, got {num_devices}"
    assert cluster_shape == (2, 1), f"Only (2, 1) cluster shape is supported for swap tensor test, got {cluster_shape}"

    if num_iters < 1:
        pytest.fail("num_iters must be >= 1")
    # Use Async mode based on test input config
    mesh_device.enable_async(enable_async)

    core_grid = mesh_device.compute_with_storage_grid_size()
    total_num_cores = core_grid.x * core_grid.y
    ALL_CORES = ttnn.num_cores_to_corerangeset(total_num_cores, core_grid, row_wise=True)

    if enable_async:
        logger.info(f"Using Async Mode for Swap Tensor Op Dispatch")

    if use_priority:
        logger.info(f"Using Priority Tensors for Swap Tensor Op")

    ##################################
    ##### Set up fabric stuff
    ##################################
    swap_tensor_topology = ttnn.Topology.Linear
    worker_sub_device = ttnn.SubDevice([ALL_CORES])

    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    # create global semaphore handles
    num_buffers = 8
    ccl_semaphore_handles = [
        create_global_semaphore_with_same_address(mesh_device, ALL_CORES, 0) for _ in range(num_buffers)
    ]

    logger.info(f"Output shape: {output_shape}")

    ##################################
    ##### Set up input tensors/configs
    ##################################

    M, N = output_shape[2:]
    N_per_shard = round_up(math.ceil(N / input_num_cores), ttnn.TILE_SIZE)
    output_N_per_shard = round_up(math.ceil(N / output_num_cores), ttnn.TILE_SIZE)
    input_shape = [*cluster_shape, M, N]

    input_core_range_set = ttnn.num_cores_to_corerangeset(input_num_cores, core_grid, row_wise=True)
    input_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            input_core_range_set,
            [M, N_per_shard],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    output_core_range_set = ttnn.num_cores_to_corerangeset(output_num_cores, core_grid, row_wise=True)
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            output_core_range_set,
            [M, output_N_per_shard],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    logger.info(f"Input shape: {input_shape[2:]}, Padded shape: {[M, N_per_shard * input_num_cores]}")
    input_tensor = torch.randn(input_shape)

    tt_input_tensor = ttnn.from_torch(
        input_tensor,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=input_dtype,
        memory_config=input_mem_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=cluster_shape),
    )
    check_mesh_tensor_alloc(tt_input_tensor)

    # Set up the priorities
    priorities = []
    priorities_tt = []
    possible_priorities = [0, 1, 2]
    MAX_PRIORITY_TENSORS = 10
    if use_priority:
        for _ in range(min(MAX_PRIORITY_TENSORS, num_iters)):
            # Randomly select 2 valid priorities from the possible priorities
            temp_priorities = possible_priorities.copy()
            priority_a = temp_priorities.pop(torch.randint(0, len(temp_priorities), (1,)).item())
            priority_b = temp_priorities.pop(torch.randint(0, len(temp_priorities), (1,)).item())

            priorities.append((priority_a, priority_b))

            # Each device has a (ttnn.TILE_SIZE, ttnn.TILE_SIZE) priority tensor
            # priority_tensor_a -> self priority
            # priority_tensor_b -> other device's priority
            priority_a = torch.tensor(priority_a).view(1, 1, 1, 1).repeat(1, 1, ttnn.TILE_SIZE, ttnn.TILE_SIZE)
            priority_b = torch.tensor(priority_b).view(1, 1, 1, 1).repeat(1, 1, ttnn.TILE_SIZE, ttnn.TILE_SIZE)

            # Priority tensor A from the perspective of the first device
            priority_a_tt = ttnn.from_torch(
                torch.concat([priority_a, priority_b], dim=0),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.int32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=cluster_shape),
            )

            # Priority tensor B from the perspective of the second device
            priority_b_tt = ttnn.from_torch(
                torch.concat([priority_b, priority_a], dim=0),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.int32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=cluster_shape),
            )

            priorities_tt.append((priority_a_tt, priority_b_tt))

    # Swap-Tensor Golden
    output_tensor_goldens_list = []
    for i in range(num_iters):
        if use_priority:
            priority = priorities[i % MAX_PRIORITY_TENSORS]
            if priority[0] > priority[1]:
                # Device 0 has higher priority
                output_tensor_golden = input_tensor[0:1, ...].repeat(2, 1, 1, 1)
                output_tensor_goldens_list.append(output_tensor_golden)
            else:
                # Device 1 has higher priority
                output_tensor_golden = input_tensor[1:2, ...].repeat(2, 1, 1, 1)
                output_tensor_goldens_list.append(output_tensor_golden)
        else:
            # Flipping is the same as swapping, when num_devices = 2
            output_tensor_goldens_list.append(torch.flip(input_tensor, [0]))

    ##################################
    ##### Run the op
    ##################################
    def run_op(n_iters, store_all_results=True, use_priority=use_priority):
        outs = []
        for i in range(n_iters):
            if use_priority:
                out = ttnn.experimental.swap_tensor_async(
                    tt_input_tensor,
                    priorities_tt[i % MAX_PRIORITY_TENSORS][0],
                    priorities_tt[i % MAX_PRIORITY_TENSORS][1],
                    multi_device_global_semaphore=ccl_semaphore_handles[i % num_buffers],
                    memory_config=output_mem_config,
                    topology=swap_tensor_topology,
                    num_links=num_links,
                    subdevice_id=worker_sub_device_id,
                )
            else:
                out = ttnn.experimental.swap_tensor_async(
                    tt_input_tensor,
                    multi_device_global_semaphore=ccl_semaphore_handles[i % num_buffers],
                    memory_config=output_mem_config,
                    topology=swap_tensor_topology,
                    num_links=num_links,
                    subdevice_id=worker_sub_device_id,
                )
            if not trace_mode:
                ttnn.synchronize_device(mesh_device)
            if store_all_results:
                outs.append(out)

        if store_all_results:
            return outs
        else:
            return [out]

    if trace_mode:
        ##### Compile Model #####
        logger.info("Compiling model")
        tt_outs = run_op(num_iters, store_all_results=validate_all)

        ##### Capture Trace #####
        logger.info("Capturing trace")

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        tt_outs = run_op(num_iters, store_all_results=validate_all)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        ##### Run Trace #####
        logger.info("Starting Trace perf test...")

        signpost("start")
        ttnn.execute_trace(mesh_device, trace_id, blocking=False)
        ttnn.release_trace(mesh_device, trace_id)
        ttnn.synchronize_device(mesh_device)
        signpost("stop")
    else:
        signpost("start")
        tt_outs = run_op(num_iters, store_all_results=validate_all)
        signpost("stop")

    ##################################
    ##### Validation
    ##################################
    def validate(tt_out_tensor, output_tensor):
        for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
            # get_device_tensors returns row major, so we need to select the correct golden tensor
            output_tensor_ = output_tensor[i].unsqueeze(0).unsqueeze(0)

            tt_output_tensor = t.cpu().to_torch()
            # logger.info(f"Checking for device {t.device().id()}")

            if input_dtype == ttnn.bfloat16:
                eq, output = comp_pcc(tt_output_tensor, output_tensor_)
            else:
                eq, output = comp_pcc(tt_output_tensor, output_tensor_)
            assert eq, f"{i} FAILED: {output}"
        logger.info(f"PCC output is: {output}")

    if validate_all:
        for tensor_index in range(len(tt_outs)):
            tt_out_tensor = tt_outs[tensor_index]
            output_tensor = output_tensor_goldens_list[tensor_index]
            validate(tt_out_tensor, output_tensor)
    else:
        tt_out_tensor = tt_outs[-1]
        output_tensor = output_tensor_goldens_list[-1]
        validate(tt_out_tensor, output_tensor)

    for i in range(mesh_device.get_num_devices()):
        assert (
            mesh_device.get_devices()[i].num_program_cache_entries() == 1
            or mesh_device.get_devices()[i].num_program_cache_entries() == num_iters
        ), f"Device {i} has {mesh_device.get_devices()[i].num_program_cache_entries()} program cache entries"

    mesh_device.reset_sub_device_stall_group()


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.timeout(1500)
@pytest.mark.parametrize(
    "output_shape, num_links, input_num_cores, output_num_cores",
    [
        ([1, 1, 32, 32], 1, 1, 1),
        ([1, 1, 32, 8192], 1, 16, 16),
        ([1, 1, 32, 8192], 1, 32, 16),
        ([1, 1, 32, 4096], 1, 8, 42),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        ttnn.int32,
    ],
)
@pytest.mark.parametrize(
    "use_priority",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "num_iters",
    [
        50,
    ],
)
@pytest.mark.parametrize("enable_async", [True])
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 23887872,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (2, 1),
    ],
    indirect=True,
)
def test_swap_tensor(
    mesh_device,
    output_shape,
    input_dtype,
    num_links,
    input_num_cores,
    output_num_cores,
    use_priority,
    num_iters,
    enable_async,
    trace_mode,
    use_program_cache,
    function_level_defaults,
):
    run_swap_tensor_impl(
        mesh_device,
        output_shape,
        input_dtype,
        num_links,
        input_num_cores,
        output_num_cores,
        use_priority=use_priority,
        num_iters=num_iters,
        enable_async=enable_async,
        trace_mode=trace_mode,
        validate_all=False,
    )
