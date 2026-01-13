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

torch.set_printoptions(threshold=float("inf"))


def _device_mesh_iterator(mesh_shape):
    for m0 in range(mesh_shape[0]):
        for m1 in range(mesh_shape[1]):
            device = m0 * mesh_shape[1] + m1
            yield m0, m1, device


def _pack_uint16_to_uint32_torch(first, second):
    low = np.uint16(first)
    high = np.uint16(second)

    packed = (high.astype(np.uint32) << 16) | low.astype(np.uint32)
    return torch.from_numpy(np.array(packed, dtype=np.uint32))


def _unpack_uint16_from_uint32_torch(packed):
    p = packed.to(torch.int64)

    low = (p & 0xFFFF).to(torch.uint16)
    high = (p >> 16).to(torch.uint16)

    return high, low


def _unpack_dense_metadata_entry(entry, seq):
    k_entries = _unpack_uint16_from_uint32_torch(entry[0])
    token_id = entry[2].item()
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

    metadata_entry_size_bytes = 2 * num_local_experts + 2 * num_local_experts + 4
    assert metadata_entry_size_bytes % 4 == 0
    metadata_entry_size_uint32 = metadata_entry_size_bytes // 4

    dense_metadata_buffer = torch.zeros([devices, batch * seq, metadata_entry_size_uint32], dtype=torch.uint32)
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
                    metadata_entry = torch.zeros([metadata_entry_size_uint32], dtype=torch.uint32)
                    metadata_entry[0] = _pack_uint16_to_uint32_torch(*k_entries)
                    metadata_entry[2] = b * seq + s
                    token_count = dense_token_counts[rec_d].item()
                    dense_metadata_buffer[rec_d, token_count] = metadata_entry
                    dense_token_counts[rec_d] = token_count + 1

    dense_metadata_buffer.reshape(devices, batch * seq * metadata_entry_size_uint32)
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
            k_entries, b, s = _unpack_dense_metadata_entry(dense_metadata_entry, seq)
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

    hidden_size = dense_input_contribs_tensor.shape[-1]
    output_ref_tensor = torch.zeros(batch * seq * cluster_factor, select_experts_k, hidden_size).bfloat16()
    output_data_map = torch.zeros(output_ref_tensor.shape[:-1])

    batch_rep_idxr = get_batch_cluster_idxr(cluster_axis, batch)

    for m0, m1, rec_d in _device_mesh_iterator(mesh_shape):
        for dt in range(dense_token_counts[rec_d]):
            dense_metadata_entry = dense_metadata_tensor[rec_d, dt]
            k_entries, b, s = _unpack_dense_metadata_entry(dense_metadata_entry, seq)
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
