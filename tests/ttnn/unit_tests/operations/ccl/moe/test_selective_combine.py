# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


from loguru import logger
import math
import pdb

import random

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import ttnn

from tests.nightly.t3000.ccl.test_all_to_all_combine import (
    check_results,
    get_batch_cluster_idxr,
    get_experts_on_device,
    get_expert_indices,
    get_cluster_dims,
    gen_tensors as gen_tensors_combine,
    get_output_combined_contribs as gen_reference_reference,
)

from tests.nightly.t3000.ccl.test_all_to_all_dispatch import get_mesh_mapper
from models.perf.benchmarking_utils import BenchmarkProfiler

from tracy import signpost

torch.set_printoptions(threshold=float("inf"))


def _device_mesh_iterator(mesh_shape):
    for m0 in range(mesh_shape[0]):
        for m1 in range(mesh_shape[1]):
            device = m0 * mesh_shape[1] + m1
            yield m0, m1, device


def _unpack_dense_metadata_entry(entry, num_local_experts, seq):
    token_id = entry[0].item()
    k_entries = entry[1 : num_local_experts + 1]
    b = token_id // seq
    s = token_id % seq

    return k_entries, b, s


def gen_dense_metadata(batch, seq, experts, select_experts_k, mesh_shape, cluster_axis, metadata, expert_mapping):
    _, _, devices = get_cluster_dims(cluster_axis, mesh_shape)

    assert experts % devices == 0

    num_local_experts = experts // devices

    """
    struct Header {
       uint32_t token_id; // which token in source device's buffer
       uint32_t k[2]; // k+1 if not activated
       uint32_t expert_weight[2]; // bfloat16 scores
       ... 16 byte padding
    }
    """

    metadata_entry_size = 2 * num_local_experts + 1
    dense_metadata_buffer = torch.ones([devices, batch * seq, metadata_entry_size], dtype=torch.uint32) * (
        select_experts_k + 1
    )
    dense_metadata_len = torch.zeros([devices], dtype=torch.uint32)
    active_token_counts = torch.zeros([experts], dtype=torch.int32)

    for m0, m1, rec_d in _device_mesh_iterator(mesh_shape):
        device_expert_list = get_experts_on_device(experts, expert_mapping, rec_d)

        for b in range(batch):
            for s in range(seq):
                k_entries = [select_experts_k + 1] * num_local_experts
                for k in range(select_experts_k):
                    ek = metadata[0, b, s, k].item()
                    if ek in device_expert_list:
                        local_e_idx = device_expert_list.index(ek)
                        k_entries[local_e_idx] = k
                        active_token_counts[ek] += 1
                if any(map(lambda x: x != select_experts_k + 1, k_entries)):
                    metadata_entry = torch.zeros([metadata_entry_size], dtype=torch.uint32)
                    metadata_entry[0] = b * seq + s
                    metadata_entry[1 : num_local_experts + 1] = torch.tensor(k_entries)
                    token_count = dense_metadata_len[rec_d].item()
                    dense_metadata_buffer[rec_d, token_count] = metadata_entry
                    dense_metadata_len[rec_d] = token_count + 1

    return dense_metadata_buffer, dense_metadata_len, active_token_counts


def gen_dense_input_contribs(
    batch,
    seq,
    experts,
    select_experts_k,
    sparse_contribs_tensor,
    expert_mapping,
    dense_metadata_tensor,
    dense_metadata_len,
    mesh_shape,
):
    num_local_experts = experts // expert_mapping.shape[-1]
    hidden_size = sparse_contribs_tensor.shape[-1]
    seq = sparse_contribs_tensor.shape[-2]

    dense_input_contribs_tensor = torch.zeros([experts, batch * seq, hidden_size]).bfloat16()

    for m0, m1, rec_d in _device_mesh_iterator(mesh_shape):
        device_expert_list = get_experts_on_device(experts, expert_mapping, rec_d)
        device_dense_idxs = [0] * num_local_experts
        for dt in range(dense_metadata_len[rec_d]):
            dense_metadata_entry = dense_metadata_tensor[rec_d, dt]
            k_entries, b, s = _unpack_dense_metadata_entry(dense_metadata_entry, num_local_experts, seq)
            for local_e_idx, k_entry in enumerate(k_entries):
                if k_entry == select_experts_k + 1:
                    continue

                global_e_id = rec_d * num_local_experts + local_e_idx

                contrib = sparse_contribs_tensor[global_e_id, b, s]
                assert not (contrib == torch.zeros([hidden_size])).all()

                dense_input_contribs_tensor[global_e_id, device_dense_idxs[local_e_idx]] = contrib.bfloat16()
                device_dense_idxs[local_e_idx] += 1

    return dense_input_contribs_tensor


def gen_output_ref(
    batch,
    seq,
    experts,
    select_experts_k,
    dense_input_contribs_tensor,
    dense_metadata_tensor,
    dense_metadata_len,
    mesh_shape,
    cluster_axis,
    local_reduce=False,
):
    cluster_factor, cluster_size, devices = get_cluster_dims(cluster_axis, mesh_shape)

    num_local_experts = experts // devices
    hidden_size = dense_input_contribs_tensor.shape[-1]
    output_ref_tensor = torch.zeros(batch * seq * cluster_factor, select_experts_k, hidden_size).bfloat16()
    output_data_map = torch.zeros(output_ref_tensor.shape[:-1])

    batch_rep_idxr = get_batch_cluster_idxr(cluster_axis, batch)

    for m0, m1, rec_d in _device_mesh_iterator(mesh_shape):
        edt = [0] * num_local_experts
        for dt in range(dense_metadata_len[rec_d]):
            dense_metadata_entry = dense_metadata_tensor[rec_d, dt]
            k_entries, b, s = _unpack_dense_metadata_entry(dense_metadata_entry, num_local_experts, seq)
            global_batch = batch_rep_idxr(m0, m1, b)

            reduction_buffer = torch.zeros([hidden_size]).bfloat16() if local_reduce else None
            for local_e_idx, k in enumerate(k_entries):
                if k == select_experts_k + 1:
                    continue

                global_e_idx = rec_d * num_local_experts + local_e_idx

                if local_reduce:
                    reduction_buffer += dense_input_contribs_tensor[global_e_idx, edt[local_e_idx]]
                else:
                    output_ref_tensor[global_batch * seq + s, k] = dense_input_contribs_tensor[
                        global_e_idx, edt[local_e_idx]
                    ]
                    output_data_map[global_batch * seq + s, k] = 1
                edt[local_e_idx] += 1

            if local_reduce:
                local_reduction_k = next(
                    filter(k_entries, lambda x: x != select_experts_k + 1)
                )  # somewhat arbitrary placement
                output_ref_tensor[global_batch * seq + s, local_reduction_k] = reduction_buffer
                output_data_map[global_batch * seq + s, local_reduction_k] = 1

    return output_ref_tensor, output_data_map


def gen_tensors(
    batch,
    experts,
    select_experts_k,
    hidden_size,
    seq,
    mesh_shape,
    cluster_axis,
    devices,
    scheme="random",
    local_reduce=False,
):
    _, input_sparse_contribs_tensor, expert_mapping, metadata_tensor, _, _ = gen_tensors_combine(
        batch,
        experts,
        select_experts_k,
        hidden_size,
        seq,
        mesh_shape,
        cluster_axis,
        devices,
        scheme,
        local_reduce=False,
    )

    dense_metadata_tensor, dense_metadata_len, active_token_counts = gen_dense_metadata(
        batch, seq, experts, select_experts_k, mesh_shape, cluster_axis, metadata_tensor, expert_mapping
    )
    dense_contribs_tensor = gen_dense_input_contribs(
        batch,
        seq,
        experts,
        select_experts_k,
        input_sparse_contribs_tensor,
        expert_mapping,
        dense_metadata_tensor,
        dense_metadata_len,
        mesh_shape,
    )
    output_ref, output_data_map = gen_output_ref(
        batch,
        seq,
        experts,
        select_experts_k,
        dense_contribs_tensor,
        dense_metadata_tensor,
        dense_metadata_len,
        mesh_shape,
        cluster_axis,
        local_reduce,
    )

    return dense_metadata_tensor, active_token_counts, dense_contribs_tensor, output_ref, output_data_map


NUM_DEVICES = 8


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("batch", [16])
@pytest.mark.parametrize("experts", [16])
@pytest.mark.parametrize("select_experts_k", [4])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("seq", [1])
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("devices", [NUM_DEVICES])
@pytest.mark.parametrize("worker_core_range", [((0, 0), (1, 0))])
@pytest.mark.parametrize("mux_core_range", [((0, 1), (0, 2))])
@pytest.mark.parametrize("num_token_parallel_cores", [2])
@pytest.mark.parametrize("num_data_parallel_cores", [1])
def test_gen_tensors(
    mesh_device,
    batch,
    experts,
    select_experts_k,
    hidden_size,
    seq,
    cluster_axis,
    devices,
    worker_core_range,
    mux_core_range,
    num_token_parallel_cores,
    num_data_parallel_cores,
):
    mesh_shape = tuple(mesh_device.shape)

    (
        dense_metadata_tensor,
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
        scheme="random",
    )

    worker_cores = ttnn.CoreRangeSet([ttnn.CoreRange(*[ttnn.CoreCoord(c) for c in worker_core_range])])

    _get_tt_sharded_dense_input(
        dense_contribs_tensor,
        worker_cores,
        num_token_parallel_cores,
        num_data_parallel_cores,
        mesh_device,
        cluster_axis,
    )


FABRIC_PACKET_SIZE_BYTES = 4096


def _get_tt_sharded_dense_input(
    dense_contribs_tensor, core_range, num_token_parallel_cores, num_data_parallel_cores, device, cluster_axis
):
    hidden0 = dense_contribs_tensor.shape[-1]

    packet_size = min(FABRIC_PACKET_SIZE_BYTES, hidden0 * 2)

    packet_multiple = packet_size // 2
    packet_padded_hidden = math.ceil(hidden0 / packet_multiple) * packet_multiple
    packet_padding = (0, packet_padded_hidden - hidden0)
    dense_contribs_tensor = F.pad(dense_contribs_tensor, packet_padding)

    tt_dense_contribs = ttnn.from_torch(
        dense_contribs_tensor,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0),
    )

    shape0 = dense_contribs_tensor.shape

    local_experts = shape0[0] // ttnn.get_num_devices()
    num_tokens = shape0[1]
    hidden = shape0[2]

    assert shape0[2] % num_data_parallel_cores == 0

    # want [num_token_parallel_cores, num_data_parallel_cores, local_experts, num_tokens//num_token_parallel_cores, hidden//num_data_parallel_cores]

    shape1 = [
        local_experts,
        num_token_parallel_cores,
        num_tokens // num_token_parallel_cores,
        num_data_parallel_cores,
        hidden // num_data_parallel_cores,
    ]

    tt_dense_contribs = ttnn.reshape(tt_dense_contribs, shape1)
    tt_dense_contribs = ttnn.permute(tt_dense_contribs, [3, 1, 0, 2, 4])

    shard_shape = (local_experts * num_tokens // num_token_parallel_cores, hidden // num_data_parallel_cores)
    shard_spec = ttnn.ShardSpec(core_range, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    return ttnn.interleaved_to_sharded(tt_dense_contribs, mem_config)


def _get_tt_dense_metadata(dense_metadata_tensor, mesh_device):
    metadata_shape = dense_metadata_tensor.shape
    metadata_reshape = (metadata_shape[0], metadata_shape[1] * metadata_shape[2])
    return ttnn.from_torch(
        dense_metadata_tensor.reshape(metadata_reshape),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )


def _check_ref(tt_out, output_ref, output_data_map, mesh_device, axis):
    if axis == 0:
        # need to roll my own mesh composer here for the transposed ordering
        device_shards = [ttnn.to_torch(ittout, mesh_composer=None) for ittout in ttnn.get_device_tensors(tt_out)]
        ordered_shards = []
        for ir in range(mesh_shape[1]):
            for ic in range(mesh_shape[0]):
                ordered_shards.append(device_shards[ic * mesh_shape[1] + ir])
        tt_out_agg = torch.cat(ordered_shards, dim=0)

    else:
        tt_out_agg = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    # print(f"{tt_out_agg=}")

    assert tt_out_agg.shape == output_ref.shape
    for t in range(tt_out_agg.shape[0]):
        for k in range(tt_out_agg.shape[1]):
            if output_data_map[t, k].item() == 1:
                assert torch.equal(
                    tt_out_agg[t, k, :], output_ref[t, k, :]
                ), f"Equal check failed for {t=}, {k=} with {tt_out_agg[t,k, :]=} and {output_ref[t,k, :]=}"


def _run_op_with_trace(num_iters, op_func, mesh_device, profiler):
    # compile run:
    logger.info("Compiling model")
    tt_out = tt_scores_out_list = op_func(1)
    ttnn.synchronize_device(mesh_device)

    logger.info("Capturing Warmup")

    logger.info(f"Capturing Warmup iterations")
    trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    tt_out = op_func(num_iters // 4)
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
    logger.info(f"Time taken e2e: {time_taken} s")

    return tt_out


def _run_test(
    batch,
    experts,
    select_experts_k,
    hidden_size,
    seq,
    cluster_axis,
    worker_cores,
    num_data_parallel_cores,
    num_token_parallel_cores,
    num_links,
    mux_cores,
    mesh_device,
    num_test_iters,
    trace_mode,
    profiler=None,
    scheme="random",
):
    mesh_shape = tuple(mesh_device.shape)
    devices = math.prod(mesh_shape)
    assert experts % devices == 0
    experts_per_device = experts // devices

    (
        dense_metadata_tensor,
        dense_token_counts_tensor,
        dense_contribs_tensor,
        output_ref,
        output_data_map,
    ) = gen_tensors(
        batch, experts, select_experts_k, hidden_size, seq, mesh_shape, cluster_axis, devices, scheme=scheme
    )

    # print(f"{dense_metadata_tensor=}")
    #     print(f"{dense_token_counts_tensor=}")

    tt_dense_contribs = _get_tt_sharded_dense_input(
        dense_contribs_tensor,
        worker_cores,
        num_token_parallel_cores,
        num_data_parallel_cores,
        mesh_device,
        cluster_axis,
    )
    tt_dense_metadata = _get_tt_dense_metadata(dense_metadata_tensor, mesh_device)

    # TODO figure out how to set different semaphore values for different devices
    max_active_token_count = max(dense_token_counts_tensor.tolist())
    active_token_semaphores = [
        ttnn.create_global_semaphore(mesh_device, worker_cores, max_active_token_count)
        for _ in range(experts_per_device)
    ]

    def _run_op(num_iters):
        for _ in range(num_iters):
            tt_out = ttnn.selective_reduce_combine(
                tt_dense_contribs,
                tt_dense_metadata,
                hidden_size,
                batch,
                seq,
                select_experts_k,
                experts,
                cluster_axis,
                topology=ttnn.Topology.Linear,
                num_links=num_links,
                num_token_parallel_cores=num_token_parallel_cores,
                num_data_parallel_cores=num_data_parallel_cores,
                worker_core_range_set=worker_cores,
                mux_core_range_set=mux_cores,
                active_token_count_semaphores=active_token_semaphores,
            )
        return tt_out

    if trace_mode:
        tt_out = _run_op_with_trace(num_test_iters, _run_op, mesh_device, profiler)

    else:
        tt_out = _run_op(num_test_iters)
        ttnn.synchronize_device(mesh_device)

    _check_ref(tt_out, output_ref, output_data_map, mesh_device, cluster_axis)


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("batch", [16])
@pytest.mark.parametrize("experts", [16])
@pytest.mark.parametrize("select_experts_k", [4])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("seq", [1])
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("worker_core_range", [((0, 0), (3, 3))])
@pytest.mark.parametrize("num_token_parallel_cores", [4])
@pytest.mark.parametrize("num_data_parallel_cores", [4])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("mux_core_range", [((4, 0), (4, 3))])
@pytest.mark.parametrize("num_test_iters", [1])
@pytest.mark.parametrize("num_inner_iters", [1])
def test_decode(
    mesh_device,
    batch,
    experts,
    select_experts_k,
    hidden_size,
    seq,
    cluster_axis,
    worker_core_range,
    num_token_parallel_cores,
    num_data_parallel_cores,
    num_links,
    mux_core_range,
    num_test_iters,
    num_inner_iters,
):
    mesh_device.disable_and_clear_program_cache()

    worker_cores = ttnn.CoreRangeSet([ttnn.CoreRange(*[ttnn.CoreCoord(c) for c in worker_core_range])])
    mux_cores = ttnn.CoreRangeSet([ttnn.CoreRange(*[ttnn.CoreCoord(c) for c in mux_core_range])])

    assert worker_cores.num_cores() == num_token_parallel_cores * num_data_parallel_cores

    for i in range(num_test_iters):
        _run_test(
            batch,
            experts,
            select_experts_k,
            hidden_size,
            seq,
            cluster_axis,
            worker_cores,
            num_data_parallel_cores,
            num_token_parallel_cores,
            num_links,
            mux_cores,
            mesh_device,
            num_inner_iters,
            False,
            scheme="sequential",
        )
        logger.info(f"Decode iter: {i} success")


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 500000}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("batch", [16])
@pytest.mark.parametrize("experts", [16])
@pytest.mark.parametrize("select_experts_k", [4])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("seq", [1])
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("worker_core_range", [((0, 0), (3, 3))])
@pytest.mark.parametrize("num_token_parallel_cores", [4])
@pytest.mark.parametrize("num_data_parallel_cores", [4])
@pytest.mark.parametrize("num_links", [2])
@pytest.mark.parametrize("mux_core_range", [((4, 0), (4, 3))])
@pytest.mark.parametrize("num_outer_test_iters", [1])
@pytest.mark.parametrize("num_test_iters", [4])
def test_decode_trace(
    mesh_device,
    batch,
    experts,
    select_experts_k,
    hidden_size,
    seq,
    cluster_axis,
    worker_core_range,
    num_token_parallel_cores,
    num_data_parallel_cores,
    num_links,
    mux_core_range,
    num_outer_test_iters,
    num_test_iters,
):
    worker_cores = ttnn.CoreRangeSet([ttnn.CoreRange(*[ttnn.CoreCoord(c) for c in worker_core_range])])
    mux_cores = ttnn.CoreRangeSet([ttnn.CoreRange(*[ttnn.CoreCoord(c) for c in mux_core_range])])

    assert worker_cores.num_cores() == num_token_parallel_cores * num_data_parallel_cores

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
            num_data_parallel_cores,
            num_token_parallel_cores,
            num_links,
            mux_cores,
            mesh_device,
            num_test_iters,
            trace_mode=True,
            profiler=profiler,
        )
        logger.info(f"Decode iter: {i} success")
