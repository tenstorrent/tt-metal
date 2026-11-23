# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


from tt_py_test_utils_common.selective_combine import (
    ELM_SIZE_BYTES,
    FABRIC_PACKET_SIZE_BYTES,
    _active_token_core_split_counts_even,
    _get_tt_dense_metadata,
    _get_tt_dense_token_maps,
    _get_tt_sharded_dense_input,
    _unpack_dense_metadata_entry,
    device_mesh_iterator,
    effective_data_parallel_core_dim,
    gen_dense_input_contribs,
    gen_dense_metadata,
    gen_output_ref,
    gen_tensors,
    get_sharded_dense_input,
)

from loguru import logger
import math
import os
import random

import pytest
import torch
import ttnn

# Force torch CPU ops to use all cores. bf16 matmul on CPU is single-thread on
# some torch builds even with OMP_NUM_THREADS set; this is the runtime knob that
# keeps host-side golden compute fast. See test_moe_compute_6U.py for context.
torch.set_num_threads(os.cpu_count() or 1)

# Mesh graph descriptor paths for different mesh configurations
MESH_GRAPH_DESC_1x16 = (
    "tests/tt_metal/tt_fabric/custom_mesh_descriptors/single_galaxy_1x16_torus_graph_descriptor.textproto"
)
MESH_GRAPH_DESC_1x8 = (
    "tests/tt_metal/tt_fabric/custom_mesh_descriptors/single_galaxy_1x8_torus_graph_descriptor.textproto"
)


def is_mesh_graph_descriptor_set(expected_path):
    """Check if TT_MESH_GRAPH_DESC_PATH is set to the expected path."""
    return os.environ.get("TT_MESH_GRAPH_DESC_PATH") == expected_path


random.seed(42)

from tests.nightly.t3000.ccl.test_all_to_all_combine import (
    get_batch_cluster_idxr,
    get_experts_on_device,
    get_cluster_dims,
    gen_tensors as gen_tensors_combine,
)

from models.perf.benchmarking_utils import BenchmarkProfiler

from tracy import signpost

torch.set_printoptions(threshold=float("inf"))


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 16384,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 16)], indirect=True)
@pytest.mark.parametrize("batch", [512])
@pytest.mark.parametrize("experts", [32])
@pytest.mark.parametrize("select_experts_k", [4])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("seq", [1])
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("worker_core_range", [((0, 0), (3, 3))])
@pytest.mark.parametrize("mux_core_range", [((0, 1), (0, 2))])
@pytest.mark.parametrize("token_parallel_core_dim", [4])
@pytest.mark.parametrize("data_parallel_core_dim", [4])
def test_gen_tensors(
    mesh_device,
    batch,
    experts,
    select_experts_k,
    hidden_size,
    seq,
    cluster_axis,
    worker_core_range,
    mux_core_range,
    token_parallel_core_dim,
    data_parallel_core_dim,
):
    mesh_shape = tuple(mesh_device.shape)
    devices = math.prod(mesh_shape)

    (
        dense_metadata_tensor,
        dense_token_maps,
        dense_token_counts_tensor,
        dense_contribs_tensor,
        output_ref,
        output_data_map,
    ) = gen_tensors(
        batch,
        experts,
        select_experts_k,
        hidden_size,
        seq,
        mesh_shape,
        cluster_axis,
        devices,
        token_parallel_core_dim,
        scheme="sequential",
    )

    worker_cores = ttnn.CoreRangeSet([ttnn.CoreRange(*[ttnn.CoreCoord(c) for c in worker_core_range])])

    _get_tt_sharded_dense_input(
        dense_contribs_tensor,
        worker_cores,
        token_parallel_core_dim,
        data_parallel_core_dim,
        mesh_device,
        cluster_axis,
    )

    _get_tt_dense_token_maps(dense_token_maps, mesh_device)


def _check_ref(tt_out, output_ref, output_data_map, mesh_device, axis):
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

    assert tt_out_agg.shape == output_ref.shape
    for k in range(tt_out_agg.shape[0]):
        for t in range(tt_out_agg.shape[1]):
            if output_data_map[k, t].item() == 1:
                assert torch.equal(
                    tt_out_agg[k, t, :], output_ref[k, t, :]
                ), f"Equal check failed for {t=}, {k=} with {tt_out_agg[k,t, :]=} and {output_ref[k,t, :]=}"


def _run_op_with_trace(num_iters, op_func, mesh_device, profiler):
    # compile run:
    logger.info("Compiling model")
    tt_out = op_func(1)
    ttnn.synchronize_device(mesh_device)

    logger.info("Capturing Warmup")

    logger.info(f"Capturing Warmup iterations")
    trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    tt_out = op_func(max(1, num_iters // 4))
    ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    logger.info("Warmup done")

    logger.info("Capturing Trace")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    tt_out = op_func(num_iters)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    logger.info("Starting Trace perf test...")
    profiler.start("selective-reduce-combine-trace-warmup")
    ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
    ttnn.release_trace(mesh_device, trace_id_warmup)
    ttnn.synchronize_device(mesh_device)
    profiler.end("selective-reduce-combine-trace-warmup")

    signpost("start")
    profiler.start("selective-reduce-combine-trace")
    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.synchronize_device(mesh_device)
    profiler.end("selective-reduce-combine-trace")
    signpost("stop")

    time_taken = profiler.get_duration("selective-reduce-combine-trace") - profiler.get_duration(
        "selective-reduce-combine-trace-warmup"
    )
    time_taken = time_taken / num_iters * 1000000
    logger.info(f"Time taken e2e: {time_taken} us")

    return tt_out


def _run_test(
    batch,
    experts,
    select_experts_k,
    hidden_size,
    seq,
    cluster_axis,
    worker_cores,
    data_parallel_core_dim,
    token_parallel_core_dim,
    num_links,
    mux_cores,
    mesh_device,
    num_test_iters,
    trace_mode,
    profiler=None,
    scheme="random",
    topology=ttnn.Topology.Ring,
    metadata_memory_config=None,
):
    mesh_shape = tuple(mesh_device.shape)
    devices = math.prod(mesh_shape)
    assert experts % devices == 0

    (
        dense_metadata_tensor,
        dense_token_maps,
        dense_token_counts_tensor,
        dense_contribs_tensor,
        output_ref,
        output_data_map,
    ) = gen_tensors(
        batch,
        experts,
        select_experts_k,
        hidden_size,
        seq,
        mesh_shape,
        cluster_axis,
        devices,
        token_parallel_core_dim,
        scheme=scheme,
    )

    tt_dense_contribs = _get_tt_sharded_dense_input(
        dense_contribs_tensor,
        worker_cores,
        token_parallel_core_dim,
        data_parallel_core_dim,
        mesh_device,
        cluster_axis,
    )
    tt_dense_metadata = _get_tt_dense_metadata(dense_metadata_tensor, mesh_device, metadata_memory_config)
    tt_dense_token_maps = _get_tt_dense_token_maps(dense_token_maps, mesh_device)

    tt_token_counts = ttnn.from_torch(
        dense_token_counts_tensor,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    output_tensor = torch.zeros([select_experts_k, batch * seq, hidden_size], dtype=torch.bfloat16)
    tt_output_tensor = ttnn.from_torch(
        output_tensor,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
    )

    barrier_semaphore = ttnn.create_global_semaphore(mesh_device, worker_cores, 0)

    core_list = list(ttnn.corerange_to_cores(worker_cores))

    def _run_op(num_iters):
        for _ in range(num_iters):
            if not trace_mode:
                delays = [[random.randint(0, 10) * 1000 for _ in range(mesh_shape[1])] for _ in range(mesh_shape[0])]
                ttnn.apply_device_delay(mesh_device, delays)

            tt_out = ttnn.experimental.selective_reduce_combine(
                tt_dense_contribs,
                tt_dense_metadata,
                tt_dense_token_maps,
                tt_token_counts,
                hidden_size,
                batch,
                seq,
                select_experts_k,
                cluster_axis,
                topology=topology,
                num_links=num_links,
                token_parallel_core_dim=token_parallel_core_dim,
                data_parallel_core_dim=data_parallel_core_dim,
                worker_cores=core_list,
                mux_core_range_set=mux_cores,
                output_tensor=tt_output_tensor,
                optional_cross_device_semaphore=barrier_semaphore,
            )
            # subsequent op can catch correctness errors that may get covered up by a delay prior to validation
            if not trace_mode:
                tt_out = ttnn.to_layout(tt_out, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return tt_out

    if trace_mode:
        tt_out = _run_op_with_trace(num_test_iters, _run_op, mesh_device, profiler)

    else:
        tt_out = _run_op(num_test_iters)
        ttnn.synchronize_device(mesh_device)

    _check_ref(tt_out, output_ref, output_data_map, mesh_device, cluster_axis)


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 16384,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device, mesh_shape",
    [
        pytest.param(
            (1, 8),
            (1, 8),
            marks=pytest.mark.skipif(
                not is_mesh_graph_descriptor_set(MESH_GRAPH_DESC_1x8),
                reason=f"1x8 mesh requires TT_MESH_GRAPH_DESC_PATH={MESH_GRAPH_DESC_1x8}",
            ),
            id="1x8",
        ),
        pytest.param(
            (1, 16),
            (1, 16),
            marks=pytest.mark.skipif(
                not is_mesh_graph_descriptor_set(MESH_GRAPH_DESC_1x16),
                reason=f"1x16 mesh requires TT_MESH_GRAPH_DESC_PATH={MESH_GRAPH_DESC_1x16}",
            ),
            id="1x16",
        ),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("batch", [512, 128, 64, 48])
@pytest.mark.parametrize("select_experts_k", [1, 2, 8])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("seq", [1])
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("experts_per_device", [2])
@pytest.mark.parametrize("worker_core_range", [((0, 0), (3, 3))])
@pytest.mark.parametrize("token_parallel_core_dim", [4])
@pytest.mark.parametrize("data_parallel_core_dim", [4])
@pytest.mark.parametrize("num_links", [4])
@pytest.mark.parametrize("mux_core_range", [((4, 0), (5, 7))])
@pytest.mark.parametrize("num_test_iters", [5])
@pytest.mark.parametrize("num_inner_iters", [1])
def test_decode(
    mesh_device,
    mesh_shape,
    batch,
    select_experts_k,
    hidden_size,
    seq,
    cluster_axis,
    experts_per_device,
    worker_core_range,
    token_parallel_core_dim,
    data_parallel_core_dim,
    num_links,
    mux_core_range,
    num_test_iters,
    num_inner_iters,
):
    experts = experts_per_device * mesh_shape[cluster_axis]
    mesh_device.disable_and_clear_program_cache()

    worker_cores = ttnn.CoreRangeSet([ttnn.CoreRange(*[ttnn.CoreCoord(c) for c in worker_core_range])])
    mux_cores = ttnn.CoreRangeSet([ttnn.CoreRange(*[ttnn.CoreCoord(c) for c in mux_core_range])])

    assert worker_cores.num_cores() == token_parallel_core_dim * data_parallel_core_dim

    for i in range(num_test_iters):
        _run_test(
            batch,
            experts,
            select_experts_k,
            hidden_size,
            seq,
            cluster_axis,
            worker_cores,
            data_parallel_core_dim,
            token_parallel_core_dim,
            num_links,
            mux_cores,
            mesh_device,
            num_inner_iters,
            False,
            scheme="random",
        )
        logger.info(f"Decode iter: {i} success")


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 16384,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "trace_region_size": 500000,
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 16)], indirect=True)
@pytest.mark.parametrize("batch", [512])
@pytest.mark.parametrize("experts", [32])
@pytest.mark.parametrize("select_experts_k", [2])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("seq", [1])
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("worker_core_range", [((0, 0), (3, 3))])
@pytest.mark.parametrize("token_parallel_core_dim", [4])
@pytest.mark.parametrize("data_parallel_core_dim", [4])
@pytest.mark.parametrize("num_links", [4])
@pytest.mark.parametrize("mux_core_range", [((4, 0), (5, 7))])
@pytest.mark.parametrize("num_outer_test_iters", [1])
@pytest.mark.parametrize("num_test_iters", [40])
@pytest.mark.parametrize("scheme", ["sequential", "best_congestion"])
def test_deepseek_perf(
    mesh_device,
    batch,
    experts,
    select_experts_k,
    hidden_size,
    seq,
    cluster_axis,
    worker_core_range,
    token_parallel_core_dim,
    data_parallel_core_dim,
    num_links,
    mux_core_range,
    num_outer_test_iters,
    num_test_iters,
    scheme,
):
    worker_cores = ttnn.CoreRangeSet([ttnn.CoreRange(*[ttnn.CoreCoord(c) for c in worker_core_range])])
    mux_cores = ttnn.CoreRangeSet([ttnn.CoreRange(*[ttnn.CoreCoord(c) for c in mux_core_range])])

    assert worker_cores.num_cores() >= token_parallel_core_dim * data_parallel_core_dim
    assert mux_cores.num_cores() >= num_links * 2

    profiler = BenchmarkProfiler()

    for i in range(num_outer_test_iters):
        _run_test(
            batch,
            experts,
            select_experts_k,
            hidden_size,
            seq,
            cluster_axis,
            worker_cores,
            data_parallel_core_dim,
            token_parallel_core_dim,
            num_links,
            mux_cores,
            mesh_device,
            num_test_iters,
            trace_mode=True,
            profiler=profiler,
            scheme=scheme,
        )
        logger.info(f"Decode iter: {i} success")


# Regression test for #56769. With l1_small_size = 0 the carried barrier semaphore is allocated in general L1, and
# the op must keep its count intact across iterations.
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 0,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        }
    ],
    indirect=True,
    ids=["l1_small_0"],
)
@pytest.mark.parametrize(
    "mesh_device, mesh_shape",
    [
        pytest.param(
            (1, 8),
            (1, 8),
            marks=pytest.mark.skipif(
                not is_mesh_graph_descriptor_set(MESH_GRAPH_DESC_1x8),
                reason=f"1x8 mesh requires TT_MESH_GRAPH_DESC_PATH={MESH_GRAPH_DESC_1x8}",
            ),
            id="1x8",
        ),
        pytest.param(
            (1, 16),
            (1, 16),
            marks=pytest.mark.skipif(
                not is_mesh_graph_descriptor_set(MESH_GRAPH_DESC_1x16),
                reason=f"1x16 mesh requires TT_MESH_GRAPH_DESC_PATH={MESH_GRAPH_DESC_1x16}",
            ),
            id="1x16",
        ),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("batch", [48])
@pytest.mark.parametrize("select_experts_k", [1])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("seq", [1])
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("experts_per_device", [2])
@pytest.mark.parametrize("worker_core_range", [((0, 0), (3, 3))])
@pytest.mark.parametrize("token_parallel_core_dim", [4])
@pytest.mark.parametrize("data_parallel_core_dim", [4])
@pytest.mark.parametrize("num_links", [4])
@pytest.mark.parametrize("mux_core_range", [((4, 0), (5, 7))])
@pytest.mark.parametrize("num_test_iters", [5])
def test_zero_l1_small(
    mesh_device,
    mesh_shape,
    batch,
    select_experts_k,
    hidden_size,
    seq,
    cluster_axis,
    experts_per_device,
    worker_core_range,
    token_parallel_core_dim,
    data_parallel_core_dim,
    num_links,
    mux_core_range,
    num_test_iters,
):
    experts = experts_per_device * mesh_shape[cluster_axis]

    worker_cores = ttnn.CoreRangeSet([ttnn.CoreRange(*[ttnn.CoreCoord(c) for c in worker_core_range])])
    mux_cores = ttnn.CoreRangeSet([ttnn.CoreRange(*[ttnn.CoreCoord(c) for c in mux_core_range])])

    _run_test(
        batch,
        experts,
        select_experts_k,
        hidden_size,
        seq,
        cluster_axis,
        worker_cores,
        data_parallel_core_dim,
        token_parallel_core_dim,
        num_links,
        mux_cores,
        mesh_device,
        num_test_iters,
        False,
        scheme="random",
    )
