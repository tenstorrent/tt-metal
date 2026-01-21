import pdb

import random

import numpy as np
import pytest
import torch
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


torch.set_printoptions(threshold=float("inf"))


def _device_mesh_iterator(mesh_shape):
    for m0 in range(mesh_shape[0]):
        for m1 in range(mesh_shape[1]):
            device = m0 * mesh_shape[1] + m1
            yield m0, m1, device


def _unpack_dense_metadata_entry(entry, num_local_experts, seq):
    k_entries = entry[:num_local_experts]
    token_id = entry[2 * num_local_experts].item()
    b = token_id // seq
    s = token_id % seq

    return k_entries, b, s


def gen_dense_metadata(batch, seq, experts, select_experts_k, mesh_shape, cluster_axis, metadata, expert_mapping):
    _, _, devices = get_cluster_dims(cluster_axis, mesh_shape)

    assert experts % devices == 0

    num_local_experts = experts // devices

    """
    struct Header {
       uint16_t k[2]; // k+1 if not activated
       uint16_t expert_weight[2]; // bfloat16 scores
       uint32_t token_id; // which token in source device's buffer
    }
    """

    metadata_entry_size = 2 * num_local_experts + 1
    dense_metadata_buffer = torch.zeros([devices, batch * seq, metadata_entry_size], dtype=torch.uint32)
    dense_token_counts = torch.zeros([devices], dtype=torch.uint32)

    for m0, m1, rec_d in _device_mesh_iterator(mesh_shape):
        device_expert_list = get_experts_on_device(experts, expert_mapping, rec_d)

        k_entries = [select_experts_k + 1] * num_local_experts
        for b in range(batch):
            for s in range(seq):
                for k in range(select_experts_k):
                    ek = metadata[0, b, s, k].item()
                    if ek in device_expert_list:
                        local_e_idx = device_expert_list.index(ek)
                        k_entries[local_e_idx] = k
                if any(map(lambda x: x != select_experts_k + 1, k_entries)):
                    metadata_entry = torch.zeros([metadata_entry_size], dtype=torch.uint32)
                    metadata_entry[:num_local_experts] = torch.tensor(k_entries)
                    metadata_entry[2 * num_local_experts] = b * seq + s
                    token_count = dense_token_counts[rec_d].item()
                    dense_metadata_buffer[rec_d, token_count] = metadata_entry
                    dense_token_counts[rec_d] = token_count + 1

    dense_metadata_buffer.reshape(devices, batch * seq * metadata_entry_size)
    return dense_metadata_buffer, dense_token_counts


def gen_dense_input_contribs(
    batch,
    seq,
    experts,
    select_experts_k,
    sparse_contribs_tensor,
    expert_mapping,
    dense_metadata_tensor,
    dense_token_counts,
    mesh_shape,
):
    num_local_experts = experts // expert_mapping.shape[-1]
    hidden_size = sparse_contribs_tensor.shape[-1]
    seq = sparse_contribs_tensor.shape[-2]

    dense_input_contribs_tensor = torch.zeros([experts, batch * seq, hidden_size]).bfloat16()

    for m0, m1, rec_d in _device_mesh_iterator(mesh_shape):
        device_expert_list = get_experts_on_device(experts, expert_mapping, rec_d)
        device_dense_idxs = [0] * num_local_experts
        for dt in range(dense_token_counts[rec_d]):
            dense_metadata_entry = dense_metadata_tensor[rec_d, dt]
            k_entries, b, s = _unpack_dense_metadata_entry(dense_metadata_entry, num_local_experts, seq)
            for local_e_idx, (global_e_id, k_entry) in enumerate(zip(device_expert_list, k_entries)):
                if k_entry == select_experts_k + 1:
                    continue
                contrib = sparse_contribs_tensor[global_e_id, b, s]
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
    dense_token_counts,
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
        for dt in range(dense_token_counts[rec_d]):
            dense_metadata_entry = dense_metadata_tensor[rec_d, dt]
            k_entries, b, s = _unpack_dense_metadata_entry(dense_metadata_entry, num_local_experts, seq)
            global_batch = batch_rep_idxr(m0, m1, b)

            reduction_buffer = torch.zeros([hidden_size]).bfloat16() if local_reduce else None
            for local_e_idx, k in enumerate(k_entries):
                if k == select_experts_k + 1:
                    continue

                if local_reduce:
                    reduction_buffer += dense_input_contribs_tensor[local_e_idx, dt]
                else:
                    output_ref_tensor[global_batch * seq + s, k] = dense_input_contribs_tensor[local_e_idx, dt]
                    output_data_map[global_batch * seq + s, k] = 1

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

    dense_metadata_tensor, dense_token_counts_tensor = gen_dense_metadata(
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
        dense_token_counts_tensor,
        mesh_shape,
    )
    output_ref, output_data_map = gen_output_ref(
        batch,
        seq,
        experts,
        select_experts_k,
        dense_contribs_tensor,
        dense_metadata_tensor,
        dense_token_counts_tensor,
        mesh_shape,
        cluster_axis,
        local_reduce,
    )

    return dense_metadata_tensor, dense_token_counts_tensor, dense_contribs_tensor, output_ref, output_data_map


NUM_DEVICES = 8


@pytest.mark.parametrize("batch", [64])
@pytest.mark.parametrize("experts", [16])
@pytest.mark.parametrize("selected_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [4])
@pytest.mark.parametrize("seq", [1])
@pytest.mark.parametrize("mesh_shape", [(2, 4)])
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("devices", [NUM_DEVICES])
def test_gen_tensors(batch, experts, selected_experts_k, hidden_size, seq, mesh_shape, cluster_axis, devices):
    inputs = gen_tensors(
        batch,
        experts,
        selected_experts_k,
        hidden_size,
        seq,
        mesh_shape,
        cluster_axis,
        devices,
        scheme="sequential",
    )


# experts, batch * seq, hidden_size
def _get_tt_sharded_dense_input(dense_contribs_tensor, core_range, device, cluster_axis):
    shape = dense_contribs_tensor.shape
    core_grid = core_range.ranges()[0].grid_size()

    assert shape[-1] % core_grid.y == 0

    sharded_shape = [shape[0] * shape[1] * core_grid.y, shape[-1] // core_grid.y]
    dense_contribs_tensor = dense_contribs_tensor.reshape(sharded_shape)

    mem_config = ttnn.create_sharded_memory_config(
        sharded_shape, core_range, ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True
    )

    return ttnn.from_torch(
        dense_contribs_tensor,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem_config,
        mesh_mapper=get_mesh_mapper(device, device.shape, cluster_axis, 0),
    )


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("batch", [16])
@pytest.mark.parametrize("experts", [16])
@pytest.mark.parametrize("select_experts_k", [4])
@pytest.mark.parametrize("hidden_size", [16])
@pytest.mark.parametrize("seq", [1])
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("devices", [NUM_DEVICES])
@pytest.mark.parametrize("worker_core_range", [((0, 0), (0, 0))])
@pytest.mark.parametrize("mux_core_range", [((0, 1), (0, 2))])
@pytest.mark.parametrize("num_iters", [(8, 4)])
def test_decode(
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
    num_iters,
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
    )
    assert experts % devices == 0
    experts_per_device = experts // devices

    worker_cores = ttnn.CoreRangeSet([ttnn.CoreRange(*[ttnn.CoreCoord(c) for c in worker_core_range])])
    mux_cores = ttnn.CoreRangeSet([ttnn.CoreRange(*[ttnn.CoreCoord(c) for c in mux_core_range])])

    tt_dense_contribs = _get_tt_sharded_dense_input(dense_contribs_tensor, worker_cores, mesh_device, cluster_axis)
    tt_metadata = ttnn.from_torch(
        dense_metadata_tensor,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # TODO figure out how to set different semaphore values for different devices
    max_active_token_count = max(dense_token_counts_tensor.tolist())
    active_token_semaphores = [
        ttnn.create_global_semaphore(mesh_device, worker_cores, max_active_token_count)
        for _ in range(experts_per_device)
    ]

    tt_out = ttnn.selective_reduce_combine(
        tt_dense_contribs,
        tt_metadata,
        hidden_size,
        batch,
        seq,
        select_experts_k,
        experts,
        cluster_axis,
        topology=ttnn.Topology.Linear,
        num_links=1,
        num_token_parallel_cores=1,
        num_data_parallel_cores=1,
        worker_core_range_set=worker_cores,
        mux_core_range_set=mux_cores,
        active_token_count_semaphores=active_token_semaphores,
    )
