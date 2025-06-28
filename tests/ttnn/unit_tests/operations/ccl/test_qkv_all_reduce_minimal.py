# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import math
from time import time
from loguru import logger
import ttnn
import os

is_RING_6U = os.environ.get("RING_6U", "0") == "1"
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from tracy import signpost

from tests.ttnn.utils_for_testing import assert_with_pcc

from tests.tt_eager.python_api_testing.unit_testing.misc.test_matmul_1d_gather_in0 import (
    round_up,
)
from models.demos.llama3_subdevices.tt.model_config import (
    PREFETCHER_NOC1_GRID,
)
from models.demos.llama3_subdevices.tt.model_config import set_tg_attention_config


RING_CRS = ttnn.CoreRangeSet(
    [
        ttnn.CoreRange(
            ttnn.CoreCoord(x, y),
            ttnn.CoreCoord(x, y),
        )
        for x, y in PREFETCHER_NOC1_GRID
    ]
)


def run_all_reduce_qkv_heads_fuse_perf_impl(
    mesh_device,
    output_shape,
    cluster_axis,
    input_dtype,
    output_dtype,
    num_links,
    input_num_cores,
    output_num_cores,
    num_iters=1,
    warmup_iters=0,
    trace_mode=True,
    validate_all=True,
    profiler=BenchmarkProfiler(),
    linear=True,
):
    if linear:
        ALL_GATHER_TOPOLOGY = ttnn.Topology.Linear
        WRAP_MESH = False
    else:
        ALL_GATHER_TOPOLOGY = ttnn.Topology.Ring
        WRAP_MESH = True
    cluster_shape = (8, 4)

    if num_iters < 1:
        pytest.fail("num_iters must be >= 1")

    ##################################
    ##### Set up fabric stuff
    ##################################
    model_config = {}
    model_config = set_tg_attention_config(model_config, 4096)
    ccl_sub_device_crs = model_config["CREATE_HEAD_OUTPUT_MEMCFG"].shard_spec.grid
    worker_sub_device = ttnn.SubDevice(
        [
            ccl_sub_device_crs,
        ]
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    # create global semaphore handles
    num_buffers = 2
    ccl_semaphore_handles = [
        ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, i) for i in range(num_buffers)
    ]

    logger.info(f"Output shape: {output_shape}")

    try:
        ##################################
        ##### Set up input tensors/configs
        ##################################

        ##### FF2 Case #####
        M, N = output_shape[2:]
        N_per_shard = round_up(math.ceil(N / input_num_cores), ttnn.TILE_SIZE)
        output_N_per_shard = round_up(math.ceil(N / output_num_cores), ttnn.TILE_SIZE)
        input_shape = [*cluster_shape, M, N]  # [8, 4, 32, 1280]
        intermediate_shape = [*input_shape[:-1], N * cluster_shape[cluster_axis]]

        input_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                RING_CRS,
                [M, N_per_shard],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        ar_core_range_set = ttnn.num_cores_to_corerangeset_in_subcoregrids(
            ttnn.CoreCoord(1, 0),
            output_num_cores,
            model_config["CREATE_HEAD_OUTPUT_MEMCFG"].shard_spec.grid,
            row_wise=False,
        )
        output_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ar_core_range_set,
                [M, output_N_per_shard],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        intermediate_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ar_core_range_set,
                [M, output_N_per_shard * cluster_shape[cluster_axis]],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        logger.info(f"Input shape: {input_shape[2:]}, Padded shape: {[M, N_per_shard * input_num_cores]}")
        input_tensor = torch.randn(input_shape)

        # Prepare input tensors
        tt_qkv = ttnn.from_torch(
            input_tensor,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=input_dtype,
            memory_config=input_mem_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=cluster_shape),
        )  # [1, 1, 32, 1280]

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

        head_dim = N // (8 + 2 * 1)
        qkv_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                model_config["CREATE_HEAD_OUTPUT_MEMCFG"].shard_spec.grid,
                [M, head_dim],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        # Select batch_offset with create_qkv_heads_decode instead of selection matmul
        batch_offset = [0, 8, 16, 24]
        batch_offset_tt_tensor = ttnn.as_tensor(
            torch.tensor(batch_offset, dtype=torch.int32).reshape(4, 1),
            dtype=ttnn.int32,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device=mesh_device, dims=(None, 0), mesh_shape=list(mesh_device.shape)
            ),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ##################################
        ##### Run the op
        ##################################

        def run_op(n_iters, store_all_results=True):
            outs = {}
            outs["tt_qkv_reduced"] = []
            outs["q_heads_pre_rot_1BQD"] = []
            outs["k_heads_pre_rot_1BKD"] = []
            outs["v_heads_1BKD"] = []
            for i in range(n_iters):
                (
                    tt_qkv_reduced,
                    q_heads_pre_rot_1BQD,
                    k_heads_pre_rot_1BKD,
                    v_heads_1BKD,
                ) = ttnn.experimental.all_reduce_create_qkv_heads(
                    tt_qkv,
                    tt_intermediate_tensors[i % num_buffers],
                    cluster_axis=cluster_axis,
                    mesh_device=mesh_device,
                    multi_device_global_semaphore=ccl_semaphore_handles[i % num_buffers],
                    num_heads=8,
                    memory_config=output_mem_config,
                    topology=ALL_GATHER_TOPOLOGY,
                    num_links=num_links,
                    subdevice_id=worker_sub_device_id,
                    num_kv_heads=1,
                    final_memory_config=qkv_mem_config,
                    batch_offset=batch_offset_tt_tensor,
                    slice_size=8,
                    dtype=output_dtype,
                )
                if not trace_mode:
                    ttnn.synchronize_device(mesh_device)
                if store_all_results:
                    outs["tt_qkv_reduced"].append(tt_qkv_reduced)
                    outs["q_heads_pre_rot_1BQD"].append(q_heads_pre_rot_1BQD)
                    outs["k_heads_pre_rot_1BKD"].append(k_heads_pre_rot_1BKD)
                    outs["v_heads_1BKD"].append(v_heads_1BKD)

            if store_all_results:
                return outs
            else:
                return [tt_qkv_reduced, q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD, v_heads_1BKD]

        if trace_mode:
            # Compile Run
            logger.info("Compiling model")
            tt_outs = run_op(1, store_all_results=validate_all)

            logger.info("Capturing Warmup")
            print("Warmup iteration: ", warmup_iters)
            if warmup_iters > 0:
                trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                run_op(warmup_iters, store_all_results=validate_all)
                ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=0)
                ttnn.synchronize_device(mesh_device)

            logger.info("Capturing Trace")
            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            tt_outs = run_op(num_iters, store_all_results=validate_all)
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            ttnn.synchronize_device(mesh_device)

            logger.info("Starting Trace perf test...")
            profiler.start("all-reduce-qkv-heads-trace-warmup")
            if warmup_iters > 0:
                ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
                ttnn.release_trace(mesh_device, trace_id_warmup)
                ttnn.synchronize_device(mesh_device)
            profiler.end("all-reduce-qkv-heads-trace-warmup")

            signpost("start")
            profiler.start("all-reduce-qkv-heads-trace")
            ttnn.execute_trace(mesh_device, trace_id, blocking=False)
            ttnn.release_trace(mesh_device, trace_id)
            ttnn.synchronize_device(mesh_device)
            profiler.end("all-reduce-qkv-heads-trace")
            signpost("stop")

            time_taken = profiler.get_duration("all-reduce-qkv-heads-trace") - profiler.get_duration(
                "all-reduce-qkv-heads-trace-warmup"
            )
            effective_iter = num_iters - warmup_iters
            logger.info(f"Time taken e2e: {time_taken} s")
            logger.info(f"Time per iter e2e: {time_taken / effective_iter} s")
            logger.info(f"Time per iter e2e: {time_taken / effective_iter * 1e6} us")
        else:
            signpost("start")
            tt_outs = run_op(num_iters, store_all_results=validate_all)
            signpost("stop")

        # Get non-distributed tensors
        reduced_input_tensor = input_tensor.sum(dim=1)

        reduced_input_tensor_reshaped = reduced_input_tensor.reshape(8, 32, 10, 128)
        q_output_tensor = reduced_input_tensor_reshaped[:, :, :8, :]
        k_output_tensor = reduced_input_tensor_reshaped[:, :, 8:9, :]
        v_output_tensor = reduced_input_tensor_reshaped[:, :, 9:10, :]

        def run_validate(qkv_heads):
            q_heads_pre_rot_1BQD = qkv_heads[0]
            k_heads_pre_rot_1BKD = qkv_heads[1]
            v_heads_1BKD = qkv_heads[2]
            q_non_distributed = ttnn.to_torch(
                q_heads_pre_rot_1BQD,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    mesh_device,
                    dims=(0, 1),
                    mesh_shape=cluster_shape,
                ),
            )
            k_non_distributed = ttnn.to_torch(
                k_heads_pre_rot_1BKD,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    mesh_device,
                    dims=(0, 1),
                    mesh_shape=cluster_shape,
                ),
            )
            v_non_distributed = ttnn.to_torch(
                v_heads_1BKD,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    mesh_device,
                    dims=(0, 1),
                    mesh_shape=cluster_shape,
                ),
            )
            # Compare results
            assert_with_pcc(q_output_tensor, q_non_distributed, 0.9999)
            assert_with_pcc(k_output_tensor, k_non_distributed, 0.9999)
            assert_with_pcc(v_output_tensor, v_non_distributed, 0.9999)

        if validate_all:
            for i in range(num_iters):
                tt_out = [
                    tt_outs["q_heads_pre_rot_1BQD"][i],
                    tt_outs["k_heads_pre_rot_1BKD"][i],
                    tt_outs["v_heads_1BKD"][i],
                ]
                run_validate(tt_out)
        else:
            run_validate(tt_outs)

    finally:
        mesh_device.reset_sub_device_stall_group()


# Test 1: test_all_reduce_create_qkv_heads_fuse
@pytest.mark.skipif(is_RING_6U, reason="This test is not for 6U devices")
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("num_iters, warmup_iters", [[1, 0]])
@pytest.mark.parametrize("trace_mode", [False])
@pytest.mark.parametrize("validate_all", [True])
@pytest.mark.parametrize(
    "output_shape, cluster_axis, num_links, input_num_cores, output_num_cores",
    [
        ([1, 1, 32, 1280], 1, 3, 24, 10),  # QKV all reduce
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "output_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
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
def test_all_reduce_qkv_heads_fuse(
    mesh_device,
    output_shape,
    cluster_axis,
    input_dtype,
    output_dtype,
    num_links,
    input_num_cores,
    output_num_cores,
    num_iters,
    warmup_iters,
    trace_mode,
    validate_all,
):
    if mesh_device.get_num_devices() != 32:
        pytest.skip("Not TG!")
    profiler = BenchmarkProfiler()
    run_all_reduce_qkv_heads_fuse_perf_impl(
        mesh_device,
        output_shape,
        cluster_axis,
        input_dtype,
        output_dtype,
        num_links,
        input_num_cores,
        output_num_cores,
        num_iters=num_iters,
        warmup_iters=warmup_iters,
        trace_mode=trace_mode,
        profiler=profiler,
        validate_all=validate_all,
    )


# Test 2: test_all_reduce_create_qkv_heads_fuse_perf
@pytest.mark.skipif(is_RING_6U, reason="This test is not for 6U devices")
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("num_iters, warmup_iters", [[30, 10]])
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize("validate_all", [True])
@pytest.mark.parametrize(
    "output_shape, cluster_axis, num_links, input_num_cores, output_num_cores",
    [
        ([1, 1, 32, 1280], 1, 3, 24, 10),  # QKV all reduce
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "output_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
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
def test_all_reduce_qkv_heads_fuse_perf(
    mesh_device,
    output_shape,
    cluster_axis,
    input_dtype,
    output_dtype,
    num_links,
    input_num_cores,
    output_num_cores,
    num_iters,
    warmup_iters,
    trace_mode,
    validate_all,
):
    if mesh_device.get_num_devices() != 32:
        pytest.skip("Not TG!")
    profiler = BenchmarkProfiler()
    run_all_reduce_qkv_heads_fuse_perf_impl(
        mesh_device,
        output_shape,
        cluster_axis,
        input_dtype,
        output_dtype,
        num_links,
        input_num_cores,
        output_num_cores,
        num_iters=num_iters,
        warmup_iters=warmup_iters,
        trace_mode=trace_mode,
        profiler=profiler,
        validate_all=validate_all,
    )


# Test 2: test_all_reduce_create_qkv_heads_fuse_perf
@pytest.mark.skipif(not is_RING_6U, reason="This test is only for 6U devices")
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("num_iters, warmup_iters", [[30, 10]])
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize("validate_all", [True])
@pytest.mark.parametrize(
    "output_shape, cluster_axis, num_links, input_num_cores, output_num_cores",
    [
        ([1, 1, 32, 1280], 1, 3, 24, 10),  # QKV all reduce
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "output_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 23887872,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        }
    ],
    indirect=True,
)
def test_all_reduce_qkv_heads_fuse_perf_6U(
    mesh_device,
    output_shape,
    cluster_axis,
    input_dtype,
    output_dtype,
    num_links,
    input_num_cores,
    output_num_cores,
    use_program_cache,
    num_iters,
    warmup_iters,
    trace_mode,
    validate_all,
):
    if mesh_device.get_num_devices() != 32:
        pytest.skip("Not TG!")
    profiler = BenchmarkProfiler()
    run_all_reduce_qkv_heads_fuse_perf_impl(
        mesh_device,
        output_shape,
        cluster_axis,
        input_dtype,
        output_dtype,
        num_links,
        input_num_cores,
        output_num_cores,
        use_program_cache,
        num_iters=num_iters,
        warmup_iters=warmup_iters,
        trace_mode=trace_mode,
        profiler=profiler,
        validate_all=validate_all,
        linear=False,
    )
