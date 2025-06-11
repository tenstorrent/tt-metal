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

from tests.tt_eager.python_api_testing.unit_testing.misc.test_matmul_1d_gather_in0 import (
    num_cores_to_rectangle_grid,
    round_up,
)
from models.demos.llama3_subdevices.tt.model_config import (
    PREFETCHER_NOC1_GRID,
)
from tracy import signpost


SUB_DEVICE_CRS = ttnn.CoreRangeSet(
    [
        ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
        ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
    ]
)
QKV_CRS = ttnn.num_cores_to_corerangeset_in_subcoregrids(ttnn.CoreCoord(1, 0), 10, SUB_DEVICE_CRS, row_wise=True)
RING_CRS = ttnn.CoreRangeSet(
    [
        ttnn.CoreRange(
            ttnn.CoreCoord(x, y),
            ttnn.CoreCoord(x, y),
        )
        for x, y in PREFETCHER_NOC1_GRID
    ]
)
FF1_CRS = ttnn.num_cores_to_corerangeset_in_subcoregrids(ttnn.CoreCoord(1, 0), 28, SUB_DEVICE_CRS, row_wise=True)
FF1_CRS_RS_OUT = ttnn.num_cores_to_corerangeset_in_subcoregrids(ttnn.CoreCoord(1, 0), 30, SUB_DEVICE_CRS, row_wise=True)
NORM_CRS = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 7))])
LM_HEAD_CRS = ttnn.num_cores_to_corerangeset_in_subcoregrids(ttnn.CoreCoord(1, 0), 32, SUB_DEVICE_CRS, row_wise=True)
BINARY_MULT_CRS = ttnn.num_cores_to_corerangeset_in_subcoregrids(
    ttnn.CoreCoord(1, 0), 30, SUB_DEVICE_CRS, row_wise=True
)


def run_all_gather_replicate_impl(
    mesh_device,
    input_shape,
    cluster_axis,
    input_dtype,
    num_links,
    input_num_cores,
    input_core_range_set,
    output_num_cores,
    output_core_range_set,
    num_iters=1,
    trace_mode=False,
    validate_all=True,
):
    cluster_shape = (8, 4)

    if num_iters < 1:
        pytest.fail("num_iters must be >= 1")

    ##################################
    ##### Set up fabric stuff
    ##################################

    linear = True
    if linear:
        all_gather_replicate_topology = ttnn.Topology.Linear
        wrap_mesh = False
    else:
        all_gather_replicate_topology = ttnn.Topology.Ring
        wrap_mesh = False

    worker_sub_device = ttnn.SubDevice([SUB_DEVICE_CRS])

    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    # create global semaphore handles
    num_buffers = 8
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, SUB_DEVICE_CRS, 0) for _ in range(num_buffers)]

    logger.info(f"Input shape: {input_shape}")

    ##################################
    ##### Set up input tensors/configs
    ##################################

    # Input shapes
    M, N = input_shape[2:]
    N_per_shard = round_up(math.ceil(N / input_num_cores), ttnn.TILE_SIZE)
    input_shape = [*cluster_shape, M, N]

    # Intermediate shapes
    intermediate_num_cores = cluster_shape[cluster_axis]
    intermediate_core_range_set = ttnn.num_cores_to_corerangeset_in_subcoregrids(
        ttnn.CoreCoord(1, 0), intermediate_num_cores, SUB_DEVICE_CRS, row_wise=True
    )
    intermediate_shape = [*cluster_shape, M, N * cluster_shape[cluster_axis]]
    interemediate_N_per_shard = round_up(math.ceil(intermediate_shape[-1] / intermediate_num_cores), ttnn.TILE_SIZE)

    # Output shapes
    output_shape = intermediate_shape.copy()
    output_shape[-1] *= output_num_cores
    output_N_per_shard = intermediate_shape[-1]

    input_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            input_core_range_set,
            [M, N_per_shard],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    intermediate_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            intermediate_core_range_set,
            [M, interemediate_N_per_shard],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
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

    intermediate_tensor = torch.zeros(intermediate_shape)
    tt_intermediate_tensors = []
    for i in range(num_buffers):
        tt_intermediate_tensor = ttnn.from_torch(
            intermediate_tensor,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=input_dtype,
            memory_config=intermediate_mem_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=cluster_shape),
        )

        tt_intermediate_tensors.append(tt_intermediate_tensor)

    # All Gather Replicate Golden
    output_tensor_goldens_list = []
    for i in range(num_iters):
        # Golden for all gather part
        golden_shape = intermediate_shape
        golden_shape[cluster_axis] = 1
        golden = input_tensor.transpose(-2, cluster_axis).reshape(golden_shape).squeeze(cluster_axis)

        # TODO: Add golden for replicate part

        output_tensor_goldens_list.append(golden)

    ##################################
    ##### Run the op
    ##################################

    def run_op(n_iters, store_all_results=True):
        outs = []
        for i in range(n_iters):
            out = ttnn.experimental.all_gather_replicate_async(
                tt_input_tensor,
                persistent_output_tensor=tt_intermediate_tensors[i % num_buffers],
                dim=3,
                cluster_axis=cluster_axis,
                mesh_device=mesh_device,
                multi_device_global_semaphore=ccl_semaphore_handles[i % num_buffers],
                memory_config=intermediate_mem_config,
                topology=all_gather_replicate_topology,
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
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        tt_outs = run_op(num_iters, store_all_results=validate_all)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

        ##### Run Trace #####
        logger.info("Starting Trace perf test...")
        signpost("start")
        ttnn.execute_trace(mesh_device, trace_id, blocking=False)
        ttnn.release_trace(mesh_device, trace_id)
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
            if cluster_axis == 0:
                output_tensor_ = output_tensor[i % cluster_shape[not (cluster_axis)]]
            else:
                output_tensor_ = output_tensor[i // cluster_shape[cluster_axis]]

            tt_output_tensor = t.cpu().to_torch()

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

    # assert (
    #     mesh_device.num_program_cache_entries() == 1 or mesh_device.num_program_cache_entries() == num_iters
    # ), f"Device has {mesh_device.num_program_cache_entries()} program cache entries"

    mesh_device.reset_sub_device_stall_group()


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.timeout(1500)
@pytest.mark.parametrize(
    "input_shape, cluster_axis, num_links, input_num_cores, input_core_range_set, output_num_cores, output_core_range_set",
    [
        ([1, 1, 32, 960], 1, 3, 30, BINARY_MULT_CRS, 24, RING_CRS),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "num_iters",
    [
        (10),
    ],
)
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 23887872,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
def test_all_gather_replicate(
    mesh_device,
    input_shape,
    cluster_axis,
    input_dtype,
    num_links,
    input_num_cores,
    input_core_range_set,
    output_num_cores,
    output_core_range_set,
    num_iters,
    trace_mode,
    use_program_cache,
    function_level_defaults,
):
    if mesh_device.get_num_devices() != 32:
        pytest.skip("Not TG!")

    run_all_gather_replicate_impl(
        mesh_device,
        input_shape,
        cluster_axis,
        input_dtype,
        num_links,
        input_num_cores,
        input_core_range_set,
        output_num_cores,
        output_core_range_set,
        num_iters=num_iters,
        trace_mode=trace_mode,
        validate_all=False,
    )
