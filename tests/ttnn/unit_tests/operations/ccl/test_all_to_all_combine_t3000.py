# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from time import sleep

import torch
import pytest
from loguru import logger

from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
import ttnn
from tracy import signpost
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc

from tests.ttnn.unit_tests.operations.ccl.test_all_to_all_dispatch_t3000 import (
    gen_tokens,
    gen_expert_mapping,
    get_metadata_tensor,
    get_expert_indices,
    get_output_tensor as get_sparse_tokens,
)


def _get_experts_on_device(num_experts, expert_mapping, device):
    return [e for e in range(num_experts) if expert_mapping[0, 0, e, device] == 1]


def _get_replication_dims(replication_axis, mesh_shape):
    if replication_axis == 1:
        replication_dim = mesh_shape[0]
        replication_group = mesh_shape[1]
    elif replication_axis == 0:
        replication_dim = mesh_shape[1]
        replication_group = mesh_shape[0]
    else:
        assert replication_axis == -1
        replication_dim = 1
        replication_group = mesh_shape[0] * mesh_shape[1]
    return replication_dim, replication_group


def _get_batch_rep_idxr(replication_axis, batch):
    def _idxr(m0, m1, b):
        if replication_axis == 0:
            return m1 * batch + b
        elif replication_axis == 1:
            return m0 * batch + b
        else:
            return b

    return _idxr


def get_input_sparse_contribs(
    sparse_tokens, expert_indices, expert_mapping, mesh_shape, axis, apply_fake_expert=True, local_reduce=False
):
    # sparse tokens is [devices, batch, seq, hidden_size]
    # note, in the actual op batch*=replication_dim but the reference `sparse_tokens` is not doing that here
    # desired expert contributions tensor is [experts[/devices], batch*replicate_dim, seq, hidden_size]
    # we'll multiply the tokens by the index of their assigned expert to mock expert application.

    batch = expert_indices.shape[0]
    devices = expert_mapping.shape[-1]
    experts = expert_mapping.shape[-2]
    hidden_size = sparse_tokens.shape[-1]
    selected_experts_k = expert_indices.shape[-1]
    seq = sparse_tokens.shape[-2]

    assert experts % devices == 0
    experts_per_device = experts // devices

    if local_reduce:
        expert_dim = devices
        expert_idxr = lambda d, _: d

    else:
        expert_dim = experts
        expert_idxr = lambda d, local_idx: d * experts_per_device + local_idx

    input_contribs_tensor = torch.zeros([expert_dim, batch, seq, hidden_size])
    batch_idxr = _get_batch_rep_idxr(axis, batch)

    token_expert_count = 0
    for d in range(devices):
        experts_on_device = _get_experts_on_device(experts, expert_mapping, d)
        assert len(experts_on_device) == experts_per_device
        for b in range(batch):
            for k in range(selected_experts_k):
                for s in range(seq):
                    expert_idx = expert_indices[b, 0, s, k].item()
                    if expert_idx not in experts_on_device:
                        continue

                    local_expert_idx = expert_idxr(d, experts_on_device.index(expert_idx))

                    # multiply by expert index to mock application of expert

                    if apply_fake_expert:
                        contrib = sparse_tokens[d, b, s, :] * (-1 if expert_idx == 0 else expert_idx)
                    else:
                        contrib = sparse_tokens[d, b, s, :]
                    input_contribs_tensor[local_expert_idx, b, s, :] += contrib

                    token_expert_count += 1

    assert token_expert_count == batch * seq * selected_experts_k
    return input_contribs_tensor


def get_output_combined_contribs(
    sparse_contribs, expert_indices, expert_mapping, mesh_shape, replication_axis, local_reduce=False
):
    # sparse_contribs is [E[/devices], b, seq, hidden]
    # output recalled contribs is [K, batch * replicate_dim [/devices], seq, hidden]
    batch = expert_indices.shape[0]
    experts = expert_mapping.shape[-2]
    selected_experts_k = expert_indices.shape[-1]
    hidden = sparse_contribs.shape[-1]
    seq = sparse_contribs.shape[-2]

    devices = mesh_shape[0] * mesh_shape[1]

    assert experts % devices == 0
    experts_per_device = experts // devices

    replication_dim, replication_group = _get_replication_dims(replication_axis, mesh_shape)
    batch_rep_idxr = _get_batch_rep_idxr(replication_axis, batch)

    if local_reduce:
        local_contrib_idx_func = lambda d, _: d
    else:
        local_contrib_idx_func = lambda d, local_idx: d * experts_per_device + local_idx

    output_combined_contribs_tensor = torch.zeros(selected_experts_k, batch * replication_dim, seq, hidden)
    real_data_map = torch.zeros(output_combined_contribs_tensor.shape[:-1])

    total_token_expert_count = 0
    for m0 in range(mesh_shape[0]):
        for m1 in range(mesh_shape[1]):
            d = m0 * mesh_shape[1] + m1
            device_expert_list = _get_experts_on_device(experts, expert_mapping, d)

            for b in range(batch):
                for s in range(seq):
                    token_experts = expert_indices[b, 0, s, :].tolist()
                    for eg in device_expert_list:
                        if eg in token_experts:
                            k = token_experts.index(eg)
                        else:
                            continue

                        axis_batch_idx = batch_rep_idxr(m0, m1, b)
                        local_contrib_idx = local_contrib_idx_func(d, device_expert_list.index(eg))

                        sc = sparse_contribs[local_contrib_idx, b, s, :]
                        output_combined_contribs_tensor[k, axis_batch_idx, s, :] = sc

                        real_data_map[k, axis_batch_idx, s] = 1
                        total_token_expert_count += 1

                        if local_reduce:
                            break
    # assert total_token_expert_count == batch * (devices if local_reduce else selected_experts_k) * seq
    return output_combined_contribs_tensor, real_data_map


def gen_tensors(
    batch,
    experts,
    selected_experts_k,
    hidden_size,
    seq,
    mesh_shape,
    replication_axis,
    devices,
    scheme="random",
    local_reduce=False,
):
    torch.manual_seed(20)
    # create input tokens
    assert batch % devices == 0
    assert experts % devices == 0
    assert selected_experts_k < experts

    input_tokens = gen_tokens(batch, hidden_size, seq, mesh_shape, devices, scheme)
    expert_mapping = gen_expert_mapping(experts, devices, scheme)
    expert_indices = get_expert_indices(batch, experts, selected_experts_k, seq, mesh_shape, scheme)

    sparse_dispatched_tokens = get_sparse_tokens(input_tokens, expert_indices, expert_mapping, seq, mesh_shape)
    input_sparse_contribs_tensor = get_input_sparse_contribs(
        sparse_dispatched_tokens,
        expert_indices,
        expert_mapping,
        mesh_shape,
        replication_axis,
        local_reduce=local_reduce,
    )

    output_tensor, data_map = get_output_combined_contribs(
        input_sparse_contribs_tensor,
        expert_indices,
        expert_mapping,
        mesh_shape,
        replication_axis,
        local_reduce=local_reduce,
    )

    metadata_tensor = get_metadata_tensor(expert_indices, expert_mapping, mesh_shape)

    # create expert indices
    return (
        sparse_dispatched_tokens,
        input_sparse_contribs_tensor,
        expert_mapping,
        metadata_tensor,
        output_tensor,
        data_map,
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
    topology=ttnn.Topology.Linear,
    input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    profiler=BenchmarkProfiler(),
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

    ccl_semaphore_handle = ttnn.create_global_semaphore(mesh_device, subdevice_shard_cores_grid, 0)

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

    def run_op(n):
        for _ in range(n):
            tt_out_tensor = ttnn.all_to_all_combine(
                tt_input_contribs,
                tt_expert_mapping,
                tt_metadata,
                local_reduce=local_reduce,
                num_links=num_links,
                topology=topology,
                memory_config=output_memory_config,
                global_semaphore=ccl_semaphore_handle,
                axis=axis,
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
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
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
    topology=ttnn.Topology.Linear,
    input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
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

    # create global semaphore handles
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)]

    tt_out_tensor_list = []

    def run_op(n_iters, store_all_results=True):
        tt_output_list = []

        for i in range(n_iters):
            tt_out_tensor = ttnn.all_to_all_combine(
                input_tensors[i],
                expert_mapping_tensors[i],
                metadata_tensors[i],
                num_links=num_links,
                topology=topology,
                memory_config=output_memory_config,
                global_semaphore=ccl_semaphore_handles[i],
                local_reduce=local_reduce,
                axis=axis,
            )

            ttnn.synchronize_device(mesh_device)
            if store_all_results:
                tt_output_list.append(tt_out_tensor)
        if store_all_results:
            return tt_output_list
        else:
            return [tt_out_tensor]

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


def check_results(test_tensor, ref_tensor, data_map):
    for k in range(ref_tensor.shape[0]):
        for b in range(ref_tensor.shape[1]):
            for s in range(ref_tensor.shape[2]):
                if data_map[k, b, s].item() == 1:
                    assert_with_pcc(test_tensor[k, b, s, :], ref_tensor[k, b, s, :])


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
@pytest.mark.parametrize("axis", [0, 1])
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
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
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
):
    devices = mesh_shape[0] * mesh_shape[1]
    batch = batches_per_device * devices
    experts = experts_per_device * devices

    mesh_device.disable_and_clear_program_cache()

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
    )


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
@pytest.mark.parametrize("local_reduce", [False, True])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
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
    "mesh_shape, mesh_device", [pytest.param((2, 4), (2, 4), id="2x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize("local_reduce", [True, False])
def test_simple_tensor_gen(mesh_device, mesh_shape, local_reduce):
    torch.set_printoptions(threshold=10000)
    devices = mesh_shape[0] * mesh_shape[1]
    batch = 8 * devices
    experts = 8 * devices
    select_experts_k = 8
    hidden_size = 7000
    axis = 1
    seq = 2
    replicate_dim = mesh_shape[0 if axis == 1 else 1]
    (
        sparse_dispatched_tokens,
        input_sparse_contribs_tensor,
        expert_mapping,
        metadata_tensor,
        output_tensor,
        _,
    ) = gen_tensors(
        batch,
        experts,
        select_experts_k,
        hidden_size,
        seq,
        mesh_shape,
        axis,
        devices,
        scheme="random",
        local_reduce=local_reduce,
    )

    if local_reduce:
        sparse_expert_dim = devices
    else:
        sparse_expert_dim = experts

    assert sparse_dispatched_tokens.shape == (devices, batch, seq, hidden_size)
    assert input_sparse_contribs_tensor.shape == (sparse_expert_dim, batch, seq, hidden_size)
    assert output_tensor.shape == (select_experts_k, batch * replicate_dim, seq, hidden_size)
    assert expert_mapping.shape == (1, 1, experts, devices)
    assert metadata_tensor.shape == (devices, batch, seq, select_experts_k)
