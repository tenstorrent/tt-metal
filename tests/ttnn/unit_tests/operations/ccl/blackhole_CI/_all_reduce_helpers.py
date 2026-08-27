# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
#  SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the test_new_all_reduce.py test files under
box/all_post_commit and box/nightly.

The two copies had drifted only in how program-cache entries are counted;
``cache_counter_mode`` preserves each call site's pre-refactor behavior:

- ``"device_attr_sections"`` (box/all_post_commit): a CacheEntriesCounter is
  attached to ``mesh_device.cache_entries_counter`` and every ``run_op`` call
  (compile, warmup capture, trace capture, eager run) is measured as its own
  section; trace execution is not measured.
- ``"local_wrap_dispatch"`` (box/nightly): a local CacheEntriesCounter wraps
  the whole dispatch section (trace or eager) in a single measure().
"""

import torch
import pytest
import math
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tests_common.cache_entries_counter import CacheEntriesCounter

from tests.ttnn.nightly.unit_tests.operations.matmul.test_matmul_1d_gather_in0 import (
    round_up,
)
from models.demos.llama3_70b_galaxy.tt.model_config import (
    PREFETCHER_NOC1_GRID,
)
from models.perf.benchmarking_utils import BenchmarkProfiler
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
NORM_CRS_QWEN = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 4))])

LM_HEAD_CRS = ttnn.num_cores_to_corerangeset_in_subcoregrids(ttnn.CoreCoord(1, 0), 32, SUB_DEVICE_CRS, row_wise=True)


def run_all_reduce_impl(
    mesh_device,
    output_shape,
    cluster_axis,
    input_dtype,
    num_links,
    input_num_cores,
    input_core_range_set,
    output_num_cores,
    output_core_range_set,
    output_dtype=None,
    loopback_size=1,
    num_iters=1,
    warmup_iters=0,
    trace_mode=False,
    validate_all=True,
    profiler=BenchmarkProfiler(),
    linear=True,
    cluster_shape=(4, 1),
    cache_counter_mode="local_wrap_dispatch",
):
    assert cache_counter_mode in ("device_attr_sections", "local_wrap_dispatch")

    if output_dtype is None:
        output_dtype = input_dtype

    if num_iters < 1:
        pytest.fail("num_iters must be >= 1")

    ##################################
    ##### Set up fabric stuff
    ##################################
    if linear:
        all_reduce_topology = ttnn.Topology.Linear
        wrap_mesh = False
    else:
        all_reduce_topology = ttnn.Topology.Ring
        wrap_mesh = False

    if cache_counter_mode == "device_attr_sections":
        mesh_device.cache_entries_counter = CacheEntriesCounter(mesh_device)

    worker_sub_device = ttnn.SubDevice([SUB_DEVICE_CRS])

    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    # create global semaphore handles
    num_buffers = 8
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, SUB_DEVICE_CRS, 0) for _ in range(num_buffers)]

    logger.info(f"Output shape: {output_shape}")

    ##################################
    ##### Set up input tensors/configs
    ##################################

    ##### FF2 Case #####
    M, N = output_shape[2:]
    N_per_shard = round_up(math.ceil(N / input_num_cores), ttnn.TILE_SIZE)
    output_N_per_shard = round_up(math.ceil(N / output_num_cores), ttnn.TILE_SIZE)
    input_shape = [*cluster_shape, M, N]
    intermediate_shape = [*input_shape[:-1], N * cluster_shape[cluster_axis]]

    input_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            input_core_range_set,
            [M, N_per_shard],
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
    intermediate_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            output_core_range_set,
            [M, output_N_per_shard * cluster_shape[cluster_axis]],
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

    # All-Reduce Golden
    # Inputs reduce sequentially for 10 iters
    output_tensor_goldens_list = []
    for i in range(num_iters):
        if i % loopback_size == 0:
            ar_input_tensor = input_tensor

        output_tensor_goldens_list.append(torch.sum(ar_input_tensor, dim=cluster_axis))
        ar_input_tensor = torch.concat(
            [output_tensor_goldens_list[-1].unsqueeze(cluster_axis)] * cluster_shape[cluster_axis], dim=cluster_axis
        )

    ##################################
    ##### Run the op
    ##################################

    if cache_counter_mode == "local_wrap_dispatch":
        cache_entries_counter = CacheEntriesCounter(mesh_device)
    else:
        cache_entries_counter = None

    def run_op_body(n_iters, store_all_results):
        outs = []
        for i in range(n_iters):
            if i % loopback_size == 0:
                tt_input = tt_input_tensor

            out = ttnn.experimental.all_reduce_async(
                tt_input,
                tt_intermediate_tensors[i % num_buffers],
                cluster_axis=cluster_axis,
                mesh_device=mesh_device,
                multi_device_global_semaphore=ccl_semaphore_handles[i % num_buffers],
                memory_config=output_mem_config,
                dtype=output_dtype,
                topology=all_reduce_topology,
                num_links=num_links,
                subdevice_id=worker_sub_device_id,
            )
            if not trace_mode:
                ttnn.synchronize_device(mesh_device)
            if store_all_results:
                outs.append(out)

            # Loop back the output to the input
            if loopback_size != 1:
                tt_input = ttnn.reshard(out, input_mem_config)

        return outs, out

    def run_op(n_iters, store_all_results=True):
        if cache_counter_mode == "device_attr_sections":
            with mesh_device.cache_entries_counter.measure():
                outs, out = run_op_body(n_iters, store_all_results)
        else:
            outs, out = run_op_body(n_iters, store_all_results)

        if store_all_results:
            return outs
        else:
            return [out]

    def dispatch():
        if trace_mode:
            ##### Compile Model #####
            logger.info("Compiling model")
            tt_outs = run_op(num_iters, store_all_results=validate_all)

            ##### Capture Trace #####
            logger.info("Capturing trace")
            if warmup_iters > 0:
                trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                tt_outs = run_op(warmup_iters, store_all_results=validate_all)
                ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=0)
                ttnn.synchronize_device(mesh_device)

            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            tt_outs = run_op(num_iters, store_all_results=validate_all)
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            ttnn.synchronize_device(mesh_device)

            ##### Run Trace #####
            logger.info("Starting Trace perf test...")
            profiler.start("all-reduce-async-trace-warmup")
            if warmup_iters > 0:
                ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
                ttnn.release_trace(mesh_device, trace_id_warmup)
                ttnn.synchronize_device(mesh_device)
            profiler.end("all-reduce-async-trace-warmup")

            signpost("start")
            profiler.start("all-reduce-async-trace")
            ttnn.execute_trace(mesh_device, trace_id, blocking=False)
            ttnn.release_trace(mesh_device, trace_id)
            ttnn.synchronize_device(mesh_device)
            profiler.end("all-reduce-async-trace")
            signpost("stop")
            time_taken = profiler.get_duration("all-reduce-async-trace") - profiler.get_duration(
                "all-reduce-async-trace-warmup"
            )
            effective_iter = num_iters - warmup_iters
            logger.info(f"Time taken e2e: {time_taken} s")
            logger.info(f"Time per iter e2e: {time_taken / effective_iter} s")
            logger.info(f"Time per iter e2e: {time_taken / effective_iter * 1e6} us")

        else:
            signpost("start")
            tt_outs = run_op(num_iters, store_all_results=validate_all)
            signpost("stop")

        return tt_outs

    if cache_counter_mode == "local_wrap_dispatch":
        with cache_entries_counter.measure():
            tt_outs = dispatch()
    else:
        tt_outs = dispatch()

    ##################################
    ##### Validation
    ##################################
    def validate(tt_out_tensor, output_tensor):
        for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
            # get_device_tensors returns row major, so we need to select the correct golden tensor
            if cluster_axis == 0:
                output_tensor_ = output_tensor[i % cluster_shape[not (cluster_axis)]].unsqueeze(0).unsqueeze(0)
            else:
                output_tensor_ = output_tensor[i // cluster_shape[cluster_axis]].unsqueeze(0).unsqueeze(0)

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

    reshard_op_cnt = 1 if loopback_size > 1 else 0
    if cache_counter_mode == "device_attr_sections":
        assert (
            mesh_device.cache_entries_counter.total == 1 + reshard_op_cnt
            or mesh_device.cache_entries_counter.total == num_iters + reshard_op_cnt
        ), f"Device has {mesh_device.cache_entries_counter.total} program cache entries"
    else:
        assert (
            cache_entries_counter.total == 1 + reshard_op_cnt
            or cache_entries_counter.total == num_iters + reshard_op_cnt
        ), f"Device has {cache_entries_counter.total} program cache entries"

    mesh_device.reset_sub_device_stall_group()
