# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from tt_py_test_utils_common.all_to_all_combine_ref import (
    gen_tensors,
    get_batch_cluster_idxr,
    get_cluster_dims,
    get_experts_on_device,
    get_input_sparse_contribs,
    get_output_combined_contribs,
)

from time import sleep

import torch
import pytest
from loguru import logger
import random

from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
import ttnn
from tracy import signpost
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc

from tests.nightly.t3000.ccl.test_all_to_all_dispatch import (
    gen_tokens,
    gen_expert_mapping,
    get_metadata_tensor,
    get_expert_indices,
    get_output_tensor as get_sparse_tokens,
)


def trace_all_to_all_combine(
    mesh_device,
    mesh_shape,
    axis,
    batch,
    seq,
    local_reduce,
    experts,
    select_experts_k,
    hidden_size,
    num_iters,
    warmup_iters,
    num_links,
    scheme="random",
    dtype=ttnn.bfloat16,
    topology=None,
    input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    profiler=BenchmarkProfiler(),
    test_skew=False,
):
    devices = mesh_shape[0] * mesh_shape[1]
    # input, output, interm core range set
    compute_grid = (mesh_device.compute_with_storage_grid_size().x, mesh_device.compute_with_storage_grid_size().y)
    subdevice_shard_cores_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(compute_grid[0] - 1, compute_grid[1] - 1),
            ),
        }
    )

    _, input_contrib, expert_mapping, metadata_tensor, output_contrib_tensor, data_map = gen_tensors(
        batch,
        experts,
        select_experts_k,
        hidden_size,
        seq,
        mesh_shape,
        axis,
        devices,
        scheme=scheme,
        local_reduce=local_reduce,
    )

    tt_input_contribs = ttnn.from_torch(
        input_contrib,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=dtype,
        memory_config=input_memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    tt_expert_mapping = ttnn.from_torch(
        expert_mapping,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=input_memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=mesh_shape),
    )

    tt_metadata = ttnn.from_torch(
        metadata_tensor,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=input_memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    if test_skew:
        delays = []
        for i in range(mesh_shape[0]):
            delay_at_i = []
            for j in range(mesh_shape[1]):
                delay_at_i.append(0)
            delays.append(delay_at_i)
        delays[0][0] = 800000

    def run_op(n):
        if test_skew:
            ttnn.apply_device_delay(mesh_device, delays)
        for i in range(n):
            tt_out_tensor = ttnn.all_to_all_combine(
                tt_input_contribs,
                tt_metadata,
                tt_expert_mapping,
                local_reduce=local_reduce,
                num_links=num_links,
                topology=topology,
                memory_config=output_memory_config,
                cluster_axis=axis,
            )

    # compile run:
    logger.info("Compiling model")
    tt_out_tensor_list = run_op(1)

    logger.info("Capturing Warmup")

    if warmup_iters > 0:
        logger.info(f"Capturing Warmup {warmup_iters} iterations")
        trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        run_op(warmup_iters)
        ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=0)
        ttnn.synchronize_device(mesh_device)

    logger.info("Capturing Trace")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    run_op(num_iters)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    logger.info("Starting Trace perf test...")
    profiler.start("all-to-all-combine-trace-warmup")
    if warmup_iters > 0:
        ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
        ttnn.release_trace(mesh_device, trace_id_warmup)
        ttnn.synchronize_device(mesh_device)
    profiler.end("all-to-all-combine-trace-warmup")

    signpost("start")
    profiler.start("all-to-all-combine-trace")
    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.synchronize_device(mesh_device)
    profiler.end("all-to-all-combine-trace")
    signpost("stop")

    time_taken = profiler.get_duration("all-to-all-combine-trace") - profiler.get_duration(
        "all-to-all-combine-trace-warmup"
    )
    logger.info(f"Time taken e2e: {time_taken} s")


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "trace_region_size": 500000,
        },
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 500000,
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((2, 4), (2, 4), id="2x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize("axis", [0])
@pytest.mark.parametrize("batches_per_device", [8])
@pytest.mark.parametrize("experts_per_device", [8])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7000])
@pytest.mark.parametrize("seq", [2])
@pytest.mark.parametrize("local_reduce", [False, True])
@pytest.mark.parametrize("scheme", ["random"])
@pytest.mark.parametrize("num_iters", [10])
@pytest.mark.parametrize("warmup_iters", [5])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("topology", [None])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_all_to_all_combine_trace(
    mesh_device,
    mesh_shape,
    axis,
    batches_per_device,
    experts_per_device,
    select_experts_k,
    hidden_size,
    seq,
    local_reduce,
    num_iters,
    warmup_iters,
    scheme,
    input_memory_config,
    output_memory_config,
    num_links,
    topology,
    dtype,
):
    devices = mesh_shape[0] * mesh_shape[1]
    batch = batches_per_device * devices
    experts = experts_per_device * devices

    trace_all_to_all_combine(
        mesh_device,
        mesh_shape,
        axis,
        batch,
        seq,
        local_reduce,
        experts,
        select_experts_k,
        hidden_size,
        num_iters,
        warmup_iters,
        num_links,
        scheme,
        dtype,
        topology,
        input_memory_config,
        output_memory_config,
    )


# we import and use this function in tests.sweeps_framework.sweeps.ccl.generality.all_to_all_combine.py
# so be sure to carry over interface changes!
def run_all_to_all_combine_test(
    mesh_device,
    mesh_shape,
    axis,
    batch,
    seq,
    local_reduce,
    experts,
    select_experts_k,
    hidden_size,
    num_iters,
    num_links,
    scheme="random",
    use_regular_grid=False,
    input_grid=None,
    output_grid=None,
    dtype=ttnn.bfloat16,
    topology=None,
    input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    test_skew=False,
    check_program_cache_entries=None,
):
    if test_skew and local_reduce:
        pytest.skip("Skip skew test for local reduce")
    # When asserting cache behavior, run with the program cache ENABLED and reallocate inputs per
    # iter (done below) so a cache hit exercises the stale-runtime-arg path. Regression guard for
    # the address-in-hash workaround removed from AllToAllCombineDeviceOperation::compute_program_hash:
    # if a baked semaphore/buffer address were to go stale on a hit the correctness check fails, and
    # if keying regressed to per-realloc misses the entry-count assertion fails.
    if check_program_cache_entries is not None:
        mesh_device.enable_program_cache()
        from tests.tests_common.cache_entries_counter import CacheEntriesCounter

        mesh_device.cache_entries_counter = CacheEntriesCounter(mesh_device)
    devices = mesh_shape[0] * mesh_shape[1]
    # input, output, interm core range set
    compute_grid = (mesh_device.compute_with_storage_grid_size().x, mesh_device.compute_with_storage_grid_size().y)
    subdevice_shard_cores_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(compute_grid[0] - 1, compute_grid[1] - 1),
            ),
        }
    )

    expert_mapping_tensors = []
    input_tensors = []
    metadata_tensors = []
    output_tensor_goldens_list = []

    for iter in range(num_iters):
        _, input_contrib, expert_mapping, metadata_tensor, output_contrib_tensor, data_map = gen_tensors(
            batch,
            experts,
            select_experts_k,
            hidden_size,
            seq,
            mesh_shape,
            axis,
            devices,
            scheme=scheme,
            local_reduce=local_reduce,
        )

        output_tensor_goldens_list.append((output_contrib_tensor, data_map))

        tt_input_contribs = ttnn.from_torch(
            input_contrib,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dtype,
            memory_config=input_memory_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )

        tt_expert_mapping = ttnn.from_torch(
            expert_mapping,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=input_memory_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=mesh_shape),
        )

        tt_metadata = ttnn.from_torch(
            metadata_tensor,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=input_memory_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )

        input_tensors.append(tt_input_contribs)
        expert_mapping_tensors.append(tt_expert_mapping)
        metadata_tensors.append(tt_metadata)

    ccl_sub_device_crs = subdevice_shard_cores_grid
    worker_sub_device = ttnn.SubDevice(
        [
            ccl_sub_device_crs,
        ]
    )

    tt_out_tensor_list = []

    if test_skew:
        delays = []
        for i in range(mesh_shape[0]):
            delay_at_i = []
            for j in range(mesh_shape[1]):
                delay_at_i.append(0)
            delays.append(delay_at_i)
        delays[0][0] = 400000

    def run_op(n_iters, store_all_results=True):
        tt_output_list = []

        for i in range(n_iters):
            if test_skew:
                ttnn.apply_device_delay(mesh_device, delays)
            tt_out_tensor = ttnn.all_to_all_combine(
                input_tensors[i],
                metadata_tensors[i],
                expert_mapping_tensors[i],
                num_links=num_links,
                topology=topology,
                memory_config=output_memory_config,
                local_reduce=local_reduce,
                cluster_axis=axis,
            )

            ttnn.synchronize_device(mesh_device)
            if store_all_results:
                tt_output_list.append(tt_out_tensor)
        if store_all_results:
            return tt_output_list
        else:
            return [tt_out_tensor]

    if check_program_cache_entries is not None:
        with mesh_device.cache_entries_counter.measure():
            tt_out_tensor_list = run_op(num_iters, store_all_results=True)
    else:
        tt_out_tensor_list = run_op(num_iters, store_all_results=True)

    failed = False
    for tt_out, (ref, data_map) in zip(tt_out_tensor_list, output_tensor_goldens_list):
        if axis == 0:
            # need to roll my own mesh composer here for the transposed ordering
            device_shards = [ttnn.to_torch(ittout, mesh_composer=None) for ittout in ttnn.get_device_tensors(tt_out)]
            ordered_shards = []
            for ir in range(mesh_shape[1]):
                for ic in range(mesh_shape[0]):
                    ordered_shards.append(device_shards[ic * mesh_shape[1] + ir])
            tt_out_agg = torch.cat(ordered_shards, dim=1)

        else:
            tt_out_agg = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))
        check_results(tt_out_agg, ref, data_map)

    if check_program_cache_entries is not None:
        logger.info(f"Device has {mesh_device.cache_entries_counter.total} program cache entries")
        assert mesh_device.cache_entries_counter.total == check_program_cache_entries, (
            f"Expected {check_program_cache_entries} program cache entry/entries, got "
            f"{mesh_device.cache_entries_counter.total} -- program cache is missing on cache hit "
            f"(stale-address hash workaround regression?)"
        )


def check_results(test_tensor, ref_tensor, data_map):
    for k in range(ref_tensor.shape[0]):
        for b in range(ref_tensor.shape[1]):
            for s in range(ref_tensor.shape[2]):
                if data_map[k, b, s].item() == 1:
                    assert torch.equal(
                        test_tensor[k, b, s, :], ref_tensor[k, b, s, :]
                    ), f"Equal check failed for k={k}, b={b}, s={s} with test_tensor {test_tensor[k, b, s, :]} and ref_tensor {ref_tensor[k, b, s, :]}"


@pytest.mark.parametrize(
    "device_params, mesh_shape, mesh_device, axis, num_links, test_skew",
    [
        # FABRIC_2D tests with both axis=0 and axis=1
        pytest.param(
            {
                "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
                "fabric_config": ttnn.FabricConfig.FABRIC_2D,
                "trace_region_size": 500000,
            },
            (2, 4),
            (2, 4),
            0,
            2,
            False,
            id="fabric_2d_axis_0",
        ),
        pytest.param(
            {
                "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
                "fabric_config": ttnn.FabricConfig.FABRIC_2D,
                "trace_region_size": 500000,
            },
            (2, 4),
            (2, 4),
            1,
            1,
            False,
            id="fabric_2d_axis_1",
        ),
        # FABRIC_1D tests with both axis=0 and axis=1
        pytest.param(
            {
                "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "trace_region_size": 500000,
            },
            (2, 4),
            (2, 4),
            0,
            2,
            False,
            id="fabric_1d_line_axis_0",
        ),
        pytest.param(
            {
                "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "trace_region_size": 500000,
            },
            (2, 4),
            (2, 4),
            1,
            1,
            False,
            id="fabric_1d_line_axis_1",
        ),
        # FABRIC_1D_RING tests with only axis=1 (excluding axis=0)
        pytest.param(
            {
                "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 500000,
            },
            (1, 8),
            (1, 8),
            1,
            1,
            False,
            id="fabric_1d_ring_axis_1",
        ),
        # FABRIC_1D_RING tests with only axis=1 (excluding axis=0)
        pytest.param(
            {
                "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 500000,
            },
            (1, 8),
            (1, 8),
            1,
            1,
            True,
            id="fabric_1d_ring_axis_1_skew",
        ),
    ],
    indirect=["device_params", "mesh_device"],
)
@pytest.mark.parametrize("batches_per_device", [8])
@pytest.mark.parametrize("experts_per_device", [8])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7000])
@pytest.mark.parametrize("seq", [2])
@pytest.mark.parametrize("local_reduce", [False, True])
@pytest.mark.parametrize("scheme", ["random"])
@pytest.mark.parametrize("num_iters", [2])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("topology", [None])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_all_to_all_combine_no_trace(
    mesh_device,
    mesh_shape,
    axis,
    batches_per_device,
    seq,
    local_reduce,
    experts_per_device,
    select_experts_k,
    hidden_size,
    num_iters,
    scheme,
    input_memory_config,
    output_memory_config,
    num_links,
    topology,
    dtype,
    test_skew,
):
    devices = mesh_shape[0] * mesh_shape[1]
    batch = batches_per_device * devices
    experts = experts_per_device * devices

    # Program cache stays enabled (see run_all_to_all_combine_test): with num_iters=2 and inputs
    # reallocated each iter, iter 2 must be a cache HIT so the count reflects distinct programs, not
    # dispatches. The measured (device-wide) delta counts every program-cached op run in the window:
    #   - all_to_all_combine itself (1 entry; hits on iter 2 -- this is what the guard protects).
    #   - the zeroed output it allocates internally via ttnn::moreh_full when no output tensor is
    #     passed (all_to_all_combine.cpp) -- moreh_full is itself a cached device op (+1, hits on iter 2).
    # A genuine keying regression (rebuild every realloc) would make all_to_all_combine miss twice,
    # pushing the total to 3 (or 4 with skew) -- so this assertion still catches it. test_skew adds a
    # separate apply_device_delay program (+1). Unlike the dispatch sibling, combine's output goes
    # through moreh_full, which is why its baseline is one higher.
    expected_cache_entries = 3 if test_skew else 2

    run_all_to_all_combine_test(
        mesh_device,
        mesh_shape,
        axis,
        batch,
        seq,
        local_reduce,
        experts,
        select_experts_k,
        hidden_size,
        num_iters,
        num_links=num_links,
        scheme=scheme,
        topology=topology,
        input_memory_config=input_memory_config,
        output_memory_config=output_memory_config,
        test_skew=test_skew,
        check_program_cache_entries=expected_cache_entries,
    )


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 500000,
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((2, 4), (2, 4), id="2x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("batches_per_device", [8])
@pytest.mark.parametrize("experts_per_device", [8])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize(
    "seq_len, num_iters, warmup_iters",
    [(1, 40, 10), (128, 10, 5)],
    ids=["decode", "prefill"],
)
@pytest.mark.parametrize("local_reduce", [True])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("topology", [None])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_perf(
    mesh_device,
    mesh_shape,
    cluster_axis,
    batches_per_device,
    experts_per_device,
    select_experts_k,
    hidden_size,
    seq_len,
    local_reduce,
    num_iters,
    warmup_iters,
    num_links,
    topology,
    dtype,
    input_memory_config,
    output_memory_config,
):
    if cluster_axis is None:
        dispatch_devices = mesh_shape[0] * mesh_shape[1]
    else:
        dispatch_devices = mesh_shape[cluster_axis]

    batch = batches_per_device * dispatch_devices
    experts = experts_per_device * dispatch_devices

    trace_all_to_all_combine(
        mesh_device,
        mesh_shape,
        cluster_axis,
        batch,
        seq_len,
        local_reduce,
        experts,
        select_experts_k,
        hidden_size,
        num_iters,
        warmup_iters,
        num_links,
        "random",  # scheme TODO worst_perf
        dtype,
        topology,
        input_memory_config,
        output_memory_config,
    )


@pytest.mark.parametrize(
    "device_params, mesh_shape, mesh_device, axis, num_links, test_skew",
    [
        pytest.param(
            {
                "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "trace_region_size": 500000,
            },
            (2, 4),
            (2, 4),
            1,
            1,
            False,
            id="fabric_1d_line_axis_1",
        ),
    ],
    indirect=["device_params", "mesh_device"],
)
@pytest.mark.parametrize("batches_per_device", [8])
@pytest.mark.parametrize("experts_per_device", [8])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7000])
@pytest.mark.parametrize("seq", [2])
@pytest.mark.parametrize("local_reduce", [False])
@pytest.mark.parametrize("scheme", ["random"])
@pytest.mark.parametrize("num_iters", [2])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("topology", [None])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_all_to_all_combine_no_trace_submesh(
    mesh_device,
    mesh_shape,
    axis,
    batches_per_device,
    seq,
    local_reduce,
    experts_per_device,
    select_experts_k,
    hidden_size,
    num_iters,
    scheme,
    input_memory_config,
    output_memory_config,
    num_links,
    topology,
    dtype,
    test_skew,
):
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape((1, 4)))
    devices = mesh_shape[0] * mesh_shape[1]
    batch = batches_per_device * devices
    experts = experts_per_device * devices

    mesh_device.disable_and_clear_program_cache()

    run_all_to_all_combine_test(
        submesh_device,
        submesh_device.shape,
        axis,
        batch,
        seq,
        local_reduce,
        experts,
        select_experts_k,
        hidden_size,
        num_iters,
        num_links=num_links,
        scheme=scheme,
        topology=topology,
        input_memory_config=input_memory_config,
        output_memory_config=output_memory_config,
        test_skew=test_skew,
    )


@pytest.mark.parametrize(
    "device_params, mesh_shape, mesh_device, axis",
    [
        pytest.param(
            {
                "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "trace_region_size": 500000,
            },
            (2, 4),
            (2, 4),
            1,
            id="fabric_1d_line_axis_1",
        ),
    ],
    indirect=["device_params", "mesh_device"],
)
@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("experts", [64])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("seq", [8])
@pytest.mark.parametrize("local_reduce", [False])
@pytest.mark.parametrize("scheme", ["random"])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_all_to_all_combine_batch_1(
    mesh_device,
    mesh_shape,
    axis,
    batch,
    seq,
    local_reduce,
    experts,
    select_experts_k,
    hidden_size,
    scheme,
    input_memory_config,
    output_memory_config,
    dtype,
):
    mesh_device.disable_and_clear_program_cache()
    devices = mesh_shape[0] * mesh_shape[1]
    _, input_contrib, expert_mapping, metadata_tensor, output_contrib_tensor, data_map = gen_tensors(
        batch,
        experts,
        select_experts_k,
        hidden_size,
        seq,
        mesh_shape,
        axis,
        devices,
        scheme=scheme,
        local_reduce=local_reduce,
    )
    input_contrib.reshape(devices, -1, seq, hidden_size)

    logger.info(f"Batch = 1 input_contrib shape: {input_contrib.shape}")
    logger.info(f"Batch = 1 metadata_tensor shape: {metadata_tensor.shape}")

    tt_input_contribs = ttnn.from_torch(
        input_contrib,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=dtype,
        memory_config=input_memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    tt_expert_mapping = ttnn.from_torch(
        expert_mapping,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=input_memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=mesh_shape),
    )

    tt_metadata = ttnn.from_torch(
        metadata_tensor,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=input_memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    logger.info(f"Batch = 1 tt_input_contribs shape: {tt_input_contribs.shape}")
    logger.info(f"Batch = 1 tt_metadata shape: {tt_metadata.shape}")

    tt_output_tensor = ttnn.all_to_all_combine(
        tt_input_contribs,
        tt_metadata,
        tt_expert_mapping,
        local_reduce=local_reduce,
        cluster_axis=axis,
        output_shard_dim=2,
        memory_config=output_memory_config,
    )

    logger.info(f"Batch = 1 tt_output_tensor shape: {tt_output_tensor.shape}")
    torch_tt_output_tensor = ttnn.to_torch(tt_output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2))
    logger.info(f"Batch = 1 torch_tt_output_tensor shape: {torch_tt_output_tensor.shape}")

    torch_tt_output_tensor = torch_tt_output_tensor.reshape(-1, torch_tt_output_tensor.shape[-1])

    torch_ref_output_tensor = output_contrib_tensor.reshape(-1, output_contrib_tensor.shape[-1])

    torch.allclose(torch_tt_output_tensor, torch_ref_output_tensor)
