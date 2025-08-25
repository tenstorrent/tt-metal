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
from models.demos.llama3_70b_galaxy.tt.model_config import (
    PREFETCHER_NOC1_GRID,
)
from tracy import signpost


SUB_DEVICE_CRS = ttnn.CoreRangeSet(
    [
        ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
        ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
    ]
)
MCAST_CRS = ttnn.CoreRangeSet(
    [
        ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(6, 9)),
    ]
)
MCAST_NUM_CORES = 60
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
HOP_GRID = ttnn.CoreRangeSet(
    [
        # ttnn.CoreRange(ttnn.CoreCoord(3, 6), ttnn.CoreCoord(3, 6)),
    ]
)
FF1_CRS = ttnn.num_cores_to_corerangeset_in_subcoregrids(ttnn.CoreCoord(1, 0), 28, SUB_DEVICE_CRS, row_wise=True)
FF1_CRS_RS_OUT = ttnn.num_cores_to_corerangeset_in_subcoregrids(ttnn.CoreCoord(1, 0), 30, SUB_DEVICE_CRS, row_wise=True)
NORM_CRS = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 7))])
LM_HEAD_CRS = ttnn.num_cores_to_corerangeset_in_subcoregrids(ttnn.CoreCoord(1, 0), 32, SUB_DEVICE_CRS, row_wise=True)
BINARY_MULT_CRS = ttnn.num_cores_to_corerangeset_in_subcoregrids(
    ttnn.CoreCoord(1, 0), 30, SUB_DEVICE_CRS, row_wise=True
)
MAX_DST_TILES = 8


def num_cores_to_rectangle_grid(num_cores, device):
    """
    Find a rectangular core grid size, given an number of cores.

    Return None if rectangle grid is not possible.
    """
    x = device.compute_with_storage_grid_size().x
    while x > 0 and num_cores % x != 0:
        x -= 1

    if x == 0:
        return None

    y = num_cores // x
    return (x, y)


def gen_input_tensor_per_device(cluster_shape, in_shape, default_value=0, increment=1):
    seed = default_value
    col_tensors = []
    for i in range(cluster_shape[0]):
        row_tensors = []
        for j in range(cluster_shape[1]):
            base = torch.ones(1, 1, in_shape[-2], in_shape[-1]) * seed
            row_tensors.append(base)
            seed += increment
        row_concat = torch.cat(row_tensors, dim=1)
        col_tensors.append(row_concat)
    output = torch.cat(col_tensors, dim=0)
    return output


def run_llama_all_gather_matmul_impl(
    mesh_device,
    # shape params shared by AG and MM
    B_in,
    M_in,
    K_in,
    N_in,
    cluster_axis,
    in0_dtype,
    # MM params for in1
    in1_dtype,
    num_links,
    input_num_cores,
    input_core_range_set,
    output_num_cores,
    output_core_range_set,
    # rest of mm params
    output_dtype,
    fidelity,
    has_bias,
    fp32_acc_mode,
    packer_l1_acc,
    grid,
    in1_is_dram_interleaved,
    # common params
    num_iters=1,
    trace_mode=False,
    validate_all=True,
    global_cb=None,
):
    cluster_shape = (8, 4)

    if num_iters < 1:
        pytest.fail("num_iters must be >= 1")

    if output_num_cores != len(grid):
        pytest.skip("Output num cores does not match grid size")

    # Only run these tests on unharvested TG
    device_grid = (mesh_device.compute_with_storage_grid_size().x, mesh_device.compute_with_storage_grid_size().y)
    if device_grid != (7, 10):
        pytest.skip("Skipping test_run_prefetcher because it only works with a 7x10 grid")

    storage_grid = num_cores_to_rectangle_grid(output_num_cores, mesh_device)
    if storage_grid is None:
        pytest.skip(f"Could not find a rectangle grid for num_cores: {output_num_cores}")

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

    logger.info(f"Input B, M, K, N: {B_in}, {M_in}, {K_in}, {N_in}")

    ##################################
    ##### Set up input tensors/configs
    ##################################

    # Input shapes
    M, K_per_device = M_in, K_in // cluster_shape[cluster_axis]
    K_per_device_per_shard = round_up(math.ceil(K_per_device / input_num_cores), ttnn.TILE_SIZE)
    in0_shape = [*cluster_shape, M, K_per_device]
    in1_shape = [*cluster_shape, K_in, N_in]

    K_per_shard = round_up(math.ceil(K_in / output_num_cores), ttnn.TILE_SIZE)
    K_padded = K_per_shard * output_num_cores
    N_per_shard = round_up(math.ceil(N_in / output_num_cores), ttnn.TILE_SIZE)
    N_per_shard_in_dram = N_per_shard * 2
    N_padded = N_per_shard * output_num_cores

    logger.info(f"K_per_shard {K_per_shard}")
    logger.info(f"K_padded {K_padded}")
    logger.info(f"N_per_shard {N_per_shard}")
    logger.info(f"N_padded {N_padded}")

    in0_block_h = M // ttnn.TILE_SIZE
    in0_block_w = 3  # change this to 4 once padding is removed because 28 is divisible by 40 but not 30
    # in0_block_w = K_in // cluster_shape[cluster_axis] // ttnn.TILE_SIZE
    # while (K_in / ttnn.TILE_SIZE) % in0_block_w != 0:
    #     in0_block_w -= 1

    out_block_h = M // ttnn.TILE_SIZE
    out_block_w = N_padded // output_num_cores // ttnn.TILE_SIZE

    num_blocks_y = (M // ttnn.TILE_SIZE - 1) // out_block_h + 1
    num_blocks_x = (N_padded // ttnn.TILE_SIZE - 1) // out_block_w + 1
    num_blocks_total = num_blocks_y * num_blocks_x

    if num_blocks_total != output_num_cores:
        pytest.skip(f"num_blocks_total {num_blocks_total} != output_num_cores {output_num_cores}")

    out_subblock_h = 1
    out_subblock_w = MAX_DST_TILES
    while out_block_w % out_subblock_w != 0:
        out_subblock_w -= 1

    logger.debug("in0 block h w " + str(in0_block_h) + " " + str(in0_block_w))
    logger.debug("in1 block h w " + str(in0_block_w) + " " + str(out_block_w))
    logger.debug("out block h w " + str(out_block_h) + " " + str(out_block_w))
    logger.debug("out subblock h w " + str(out_subblock_h) + " " + str(out_subblock_w))

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=storage_grid,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
        gather_in0=True,
        hop_cores=HOP_GRID,
        num_global_cb_receivers=24,
        untilize_out=False,
    )
    print(f"program_config: {program_config}\n\n\n")
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=fp32_acc_mode,
        packer_l1_acc=packer_l1_acc,
        dst_full_sync_en=True,
    )
    print(f"compute_kernel_config: {compute_kernel_config}\n\n\n")

    # Intermediate shapes
    intermediate_num_cores = cluster_shape[cluster_axis]
    intermediate_core_range_set = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(3, 0), ttnn.CoreCoord(3, 3)),
        ]
    )
    aggregated_core_range_set = ttnn.num_cores_to_corerangeset_in_subcoregrids(
        ttnn.CoreCoord(1, 0), MCAST_NUM_CORES, MCAST_CRS, row_wise=True
    )
    intermediate_shape = [*cluster_shape, M, K_per_device * cluster_shape[cluster_axis]]
    aggregated_shape = [*cluster_shape, M, K_per_device * intermediate_num_cores * MCAST_NUM_CORES]
    interemediate_N_per_shard = round_up(math.ceil(intermediate_shape[-1] / intermediate_num_cores), ttnn.TILE_SIZE)

    # Output shapes
    output_shape = intermediate_shape.copy()
    output_shape[1] *= output_num_cores
    output_N_per_shard = intermediate_shape[-1]

    input_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            input_core_range_set,
            [M, K_per_device_per_shard],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    print(f"input_mem_config: {input_mem_config}\n\n\n")
    in1_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            output_core_range_set,
            [K_in, N_per_shard],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    print(f"in1_sharded_mem_config: {in1_sharded_mem_config}\n\n\n")
    intermediate_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            intermediate_core_range_set,
            [M, interemediate_N_per_shard],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    print(f"intermediate_mem_config: {intermediate_mem_config}\n\n\n")
    ag_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            intermediate_core_range_set,
            [M, interemediate_N_per_shard],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    print(f"ag_output_mem_config: {ag_output_mem_config}\n\n\n")
    mm_output_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            output_core_range_set,
            [M, N_per_shard],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    print(f"mm_output_sharded_mem_config: {mm_output_sharded_mem_config}\n\n\n")
    logger.info(f"Input shape: {in0_shape[2:]}, Padded shape: {[M, K_per_device_per_shard * input_num_cores]}")
    in0_tensor = torch.randn(in0_shape)
    tt_input_tensor = ttnn.from_torch(
        in0_tensor,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=in0_dtype,
        memory_config=input_mem_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=cluster_shape),
    )

    in1_tensor = torch.randn(in1_shape)
    tt_in1_tensor = ttnn.from_torch(
        in1_tensor,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=in1_dtype,
        memory_config=in1_sharded_mem_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=cluster_shape),
    )

    # Intermediate tensors
    intermediate_tensor = torch.zeros(intermediate_shape)
    tt_intermediate_tensors = []
    for i in range(num_buffers):
        tt_intermediate_tensor = ttnn.from_torch(
            intermediate_tensor,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=in0_dtype,
            memory_config=intermediate_mem_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=cluster_shape),
        )
        tt_intermediate_tensors.append(tt_intermediate_tensor)

    # All Gather Replicate Golden
    output_tensor_goldens_list = []
    for i in range(num_iters):
        # Golden for all gather part
        golden_Ashape = intermediate_shape
        golden_Ashape[cluster_axis] = 1
        golden_A = in0_tensor.transpose(-2, cluster_axis).reshape(golden_Ashape).squeeze(cluster_axis)
        golden_A = golden_A.unsqueeze(cluster_axis).repeat(1, intermediate_num_cores, 1, 1)
        # TODO: Add golden for replicate part

        output_tensor_goldens_list.append(golden_A @ in1_tensor)

    ##################################
    ##### Run the op
    ##################################
    def run_op(n_iters, store_all_results=True):
        outs = []
        for i in range(n_iters):
            out = ttnn.experimental.llama_all_gather_matmul_async(
                tt_input_tensor,
                tt_in1_tensor,
                tt_intermediate_tensors[i % num_buffers],
                dim=3,
                cluster_axis=cluster_axis,
                mesh_device=mesh_device,
                multi_device_global_semaphore=ccl_semaphore_handles[i % num_buffers],
                ag_memory_config=ag_output_mem_config,
                mm_memory_config=mm_output_sharded_mem_config,
                topology=all_gather_replicate_topology,
                num_links=num_links,
                subdevice_id=worker_sub_device_id,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                dtype=output_dtype,
                global_cb=global_cb,
            )

            # TODO: Change when actual output is integrated
            # out = tt_intermediate_tensors[i % num_buffers]

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
            row_index = i // cluster_shape[1]
            col_index = i % cluster_shape[1]
            output_tensor_ = output_tensor[row_index, col_index]
            tt_output_tensor = t.cpu().to_torch().squeeze(0).squeeze(0)

            if in0_dtype == ttnn.bfloat16:
                eq, output = comp_pcc(tt_output_tensor, output_tensor_)
            else:
                eq, output = comp_pcc(tt_output_tensor, output_tensor_)
            assert eq, f"{i} FAILED: {output}"
        logger.info(f"PCC output is: {output}")

    if validate_all:
        for tensor_index in range(len(tt_outs)):
            print(f"validating tensor_index: {tensor_index}")
            tt_out_tensor = tt_outs[tensor_index]
            output_tensor = output_tensor_goldens_list[tensor_index]
            validate(tt_out_tensor, output_tensor)
    else:
        print(f"validating last tensor")
        tt_out_tensor = tt_outs[-1]
        output_tensor = output_tensor_goldens_list[-1]
        validate(tt_out_tensor, output_tensor)

    assert (
        mesh_device.num_program_cache_entries() == 1 or mesh_device.num_program_cache_entries() == num_iters
    ), f"Device has {mesh_device.num_program_cache_entries()} program cache entries"

    mesh_device.reset_sub_device_stall_group()
