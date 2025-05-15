# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import math
from loguru import logger

from ttnn import ReplicateTensorToMesh, ShardTensor2dMesh, ConcatMeshToTensor, ConcatMesh2dToTensor
from models.common.lightweightmodule import LightweightModule
from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_equal,
    comp_pcc,
)
from models.utility_functions import is_grayskull, is_wormhole_b0, is_blackhole

from tests.tt_eager.python_api_testing.unit_testing.misc.test_matmul_1d_gather_in0 import (
    run_multi_core_matmul_1d,
    num_cores_to_rectangle_grid,
    round_up,
)
from tracy import signpost
from models.demos.llama3_subdevices.tt.prefetcher_common import get_core_ranges


def run_prefetcher_mm(
    device,
    num_tensors,
    input_shapes,
    num_layers,
    num_reader_cores,
    dtypes,
    is_functional_test=False,
    enable_performance_mode=False,
    batch_weights=False,
):
    logger.info(f"Running test_run_prefetcher with num_tensors={num_tensors}, num_layers={num_layers}")
    assert len(input_shapes) == len(dtypes)
    assert num_tensors == len(input_shapes)

    num_global_cb_receivers = 2

    K, N = input_shapes[0]

    (
        active_sender_cores,
        dram_cores,
        all_sender_cores,
        active_receiver_cores_list,
        all_receiver_cores,
        worker_cores_range_set,
        mm_optimised_ring_cores,
        hop_grid,
    ) = get_core_ranges(num_reader_cores, num_global_cb_receivers, is_functional_test)

    logger.info(f"active_sender_cores: {active_sender_cores}")
    logger.info(f"all_sender_cores: {all_sender_cores}")
    logger.info(f"active_receiver_cores_list: {active_receiver_cores_list}")
    logger.info(f"all_receiver_cores: {all_receiver_cores}")

    if num_reader_cores != 12:
        mm_optimised_ring_cores = active_receiver_cores_list

    max_tile_size = 0
    for dtype in dtypes:
        if dtype == ttnn.bfloat4_b:
            current_tile_size = 576
        elif dtype == ttnn.bfloat8_b:
            current_tile_size = 1088
        elif dtype == ttnn.bfloat16:
            current_tile_size = 2048

        if current_tile_size > max_tile_size:
            max_tile_size = current_tile_size

    # Set global buffer size to buffer a whole layer at once
    # receiver sizes in tiles
    # FF1: 72 x 5 = 375
    # FF3: 72 x 5 = 375
    # FF2: 120 x 3 = 360
    # QKV: 72 x 2 = 144
    # DO: 72 x 3 = 216
    # Total: 1470

    # global_cb_size = 1000 * max_tile_size # works without profiler, fails with profiler, 900 doesn't provide tracy info
    global_cb_size = 600 * max_tile_size
    sender_receiver_mapping = list(zip(all_sender_cores, all_receiver_cores))
    global_circular_buffer = ttnn.create_global_circular_buffer(device, sender_receiver_mapping, global_cb_size)
    logger.info(f"global cb size {global_cb_size}")

    ##### Set up the input tensors #####
    dram_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(core_coord, core_coord) for core_coord in dram_cores])
    sender_core_range_set = ttnn.CoreRangeSet(
        [ttnn.CoreRange(core_coord, core_coord) for core_coord in active_sender_cores]
    )

    padded_shapes, shard_shapes = [], []
    for K, N in input_shapes:
        num_cores = len(active_receiver_cores_list)
        K_per_shard = round_up(math.ceil(K / num_cores), ttnn.TILE_SIZE)
        K_padded = K_per_shard * num_cores
        N_per_shard = round_up(math.ceil(N / num_cores), ttnn.TILE_SIZE)
        N_padded = N_per_shard * num_cores

        padded_shapes.append((K_padded, N_padded))
        shard_shapes.append((K_per_shard, N_per_shard))

    cluster_shape = None
    mesh_mapper = None
    mesh_composer = None
    if isinstance(device, ttnn._ttnn.multi_device.MeshDevice):
        cluster_shape = device.shape
        mesh_mapper = ReplicateTensorToMesh(device)
        mesh_composer = ConcatMesh2dToTensor(device, dims=(0, 1), mesh_shape=cluster_shape)

    pt_tensors = []
    for l in range(num_layers):
        for t in range(num_tensors):
            pt_tensors.append(torch.randn(input_shapes[t]))

    tt_tensors_all = []
    for tid in range(num_tensors * num_layers):
        K, _ = input_shapes[tid % num_tensors]
        _, N = padded_shapes[tid % num_tensors]
        input_sharded_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(
                dram_core_range_set,
                [K, N // len(dram_cores)],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        tt_tensor = ttnn.as_tensor(
            pt_tensors[tid],
            device=device,
            dtype=dtypes[tid % num_tensors],
            memory_config=input_sharded_mem_config,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mesh_mapper,
        )
        tt_tensors_all.append(tt_tensor)
    tt_tensors = tt_tensors_all[:num_tensors]

    # Set up the tensor addrs
    tensor_addrs = torch.tensor([x.buffer_address() for x in tt_tensors_all])
    tensor_addrs = tensor_addrs.repeat(len(dram_cores), 1)
    tensor_addrs_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            sender_core_range_set,
            [tensor_addrs.shape[0] // len(dram_cores), tensor_addrs.shape[1]],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    tt_tensor_addrs = ttnn.as_tensor(
        tensor_addrs,
        device=device,
        dtype=ttnn.uint32,
        memory_config=tensor_addrs_mem_config,
        mesh_mapper=mesh_mapper,
    )
    tt_tensors.append(tt_tensor_addrs)

    ##### Setup up sub devices #####
    prefetcher_sub_device = ttnn.SubDevice([sender_core_range_set])
    worker_sub_device = ttnn.SubDevice([worker_cores_range_set])
    sub_device_manager = device.create_sub_device_manager([prefetcher_sub_device, worker_sub_device], 0)
    device.load_sub_device_manager(sub_device_manager)
    worker_sub_device_id = ttnn.SubDeviceId(1)  # Can we parameterize this?

    max_dst_tiles = 8
    grid = active_receiver_cores_list
    num_cores = grid[0] * grid[1] if isinstance(grid, tuple) else len(grid)
    storage_grid = num_cores_to_rectangle_grid(num_cores, device)
    M = 32

    in0_shapes = []
    out_shapes = []
    block_dims = []
    for tid in range(num_tensors):
        K, N = input_shapes[tid]
        _, N_padded = padded_shapes[tid]
        K_per_shard, N_per_shard = shard_shapes[tid]

        in0_shape = [1, 1, M, K]
        in0_shapes.append(in0_shape)

        out_shape = [1, 1, M, N_per_shard]
        out_shapes.append(out_shape)

        in0_block_h = M // ttnn.TILE_SIZE
        in0_block_w = K // num_cores // ttnn.TILE_SIZE
        while (K / ttnn.TILE_SIZE) % in0_block_w != 0:
            in0_block_w -= 1

        out_block_h = M // ttnn.TILE_SIZE
        out_block_w = N_padded // num_cores // ttnn.TILE_SIZE

        out_subblock_h = 1
        out_subblock_w = max_dst_tiles
        while out_block_w % out_subblock_w != 0:
            out_subblock_w -= 1

        logger.debug("in0 block h w " + str(in0_block_h) + " " + str(in0_block_w))
        logger.debug("in1 block h w " + str(in0_block_w) + " " + str(out_block_w))
        logger.debug("out block h w " + str(out_block_h) + " " + str(out_block_w))
        logger.debug("out subblock h w " + str(out_subblock_h) + " " + str(out_subblock_w))

        block_dim = [in0_block_w, out_subblock_h, out_subblock_w, out_block_h, out_block_w]
        block_dims.append(block_dim)
    # x, y
    if isinstance(grid, tuple):  # Generate random grid
        CORE_RANGE = [(x, y) for y in range(storage_grid[1]) for x in range(storage_grid[0])]
        random.shuffle(CORE_RANGE)
    else:
        CORE_RANGE = grid

    input_core_range_set = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(x, y),
                ttnn.CoreCoord(x, y),
            )
            for x, y in mm_optimised_ring_cores
        ]
    )

    output_core_range_set = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(x, y),
                ttnn.CoreCoord(x, y),
            )
            for x, y in CORE_RANGE
        ]
    )

    hop_core_range_set = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(x, y),
                ttnn.CoreCoord(x, y),
            )
            for x, y in hop_grid
        }
    )

    output_mem_configs = []
    for shape in out_shapes:
        _, _, M, N_per_shard = shape

        output_sharded_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                output_core_range_set,
                [M, N_per_shard],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        output_mem_configs.append(output_sharded_mem_config)

    in0_tensors = []
    in0_t_tensors = []
    prev_shape = in0_shapes[0]
    prev_in0 = torch.randn(prev_shape)
    for shape, shard_shape in zip(in0_shapes, shard_shapes):
        in0 = torch.randn(shape)
        if batch_weights and prev_shape == shape:
            in0 = prev_in0
            prev_shape = shape
        in0_tensors.append(in0)

        _, _, M, _ = shape
        K_per_shard, _ = shard_shape

        in0_sharded_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                input_core_range_set,
                [M, K_per_shard],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        in0_t = ttnn.from_torch(
            in0,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=in0_sharded_mem_config,
            mesh_mapper=mesh_mapper,
        )
        in0_t_tensors.append(in0_t)

    program_configs = []
    for block_dim in block_dims:
        in0_block_w, out_subblock_h, out_subblock_w, out_block_h, out_block_w = block_dim
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
            num_global_cb_receivers=num_global_cb_receivers,
            hop_cores=hop_core_range_set,  # Only use with noc1 grid
        )
        program_configs.append(program_config)

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
        dst_full_sync_en=True,
    )

    def run_op():
        ttnn.dram_prefetcher(
            tt_tensors,
            num_layers,
            global_cb=global_circular_buffer,
            enable_performance_mode=enable_performance_mode,
        )
        device.set_sub_device_stall_group([worker_sub_device_id])

        outputs_dram = []
        for l in range(num_layers):
            outputs_l1 = []
            t = 0
            while t < num_tensors:
                idx = l * num_tensors + t
                if batch_weights and t < num_tensors - 1 and in0_t_tensors[t].shape == in0_t_tensors[t + 1].shape:
                    logger.info(f"running matmul_batched_weights for layer {l}, tensor {t} and tensor {t+1}")
                    [output_t1, output_t2] = ttnn.matmul_batched_weights(
                        in0_t_tensors[t],
                        [tt_tensors_all[idx], tt_tensors_all[idx + 1]],
                        program_config=program_configs[t],
                        memory_config=output_mem_configs[t],
                        compute_kernel_config=compute_kernel_config,
                        global_cb=global_circular_buffer,
                        sub_device_id=worker_sub_device_id,
                    )
                    outputs_l1.append(output_t1)
                    outputs_l1.append(output_t2)
                    t += 2
                else:
                    logger.info(f"running normal matmul for layer {l}, tensor {t}")
                    output_t = ttnn.matmul(
                        in0_t_tensors[t],
                        tt_tensors_all[idx],
                        program_config=program_configs[t],
                        memory_config=output_mem_configs[t],
                        compute_kernel_config=compute_kernel_config,
                        global_cb=global_circular_buffer,
                        sub_device_id=worker_sub_device_id,
                    )
                    outputs_l1.append(output_t)
                    t += 1
            # Send outputs to DRAM to so that we don't run out of L1 memory when testing for large number of layers
            for t in range(num_tensors):
                outputs_dram.append(ttnn.to_memory_config(outputs_l1[t], ttnn.DRAM_MEMORY_CONFIG))
        device.reset_sub_device_stall_group()
        return outputs_dram

    ##### Compile Model #####
    logger.info("Compiling model")
    outputs_t = run_op()

    ##### Capture Trace #####
    logger.info("Capturing trace")

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    outputs_t = run_op()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)

    ##### Run Trace #####
    logger.info("Running trace")
    signpost("start")
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    signpost("stop")

    ##### Check Results #####
    all_passing = True
    for l in range(num_layers):
        for t in range(num_tensors):
            idx = l * num_tensors + t
            logger.info(f"Checking matmul for layer {l}, tensor {t}")
            tt_out = ttnn.to_torch(
                outputs_t[idx],
                mesh_composer=mesh_composer,
            )[:1, :1, ...]
            pt_out = in0_tensors[t] @ pt_tensors[idx]

            dtype = dtypes[t]
            if dtype == ttnn.bfloat4_b:
                pcc_threshold = 0.99
            elif dtype == ttnn.bfloat8_b:
                pcc_threshold = 0.999
            elif dtype == ttnn.bfloat16:
                pcc_threshold = 0.999

            passing, output = comp_pcc(pt_out, tt_out, pcc_threshold)
            logger.info(output)
            all_passing = passing and all_passing

    device.clear_loaded_sub_device_manager()
    device.remove_sub_device_manager(sub_device_manager)

    assert all_passing
