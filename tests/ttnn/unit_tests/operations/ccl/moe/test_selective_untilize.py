import pdb

import random

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

CHUNK_COMPLETE_VAL = torch.iinfo(torch.uint32).max


def gen_combine_semaphore_table(batch, seq, experts, mesh_shape, cluster_index, ready_device_chunk_counts: list[int]):
    _, cluster_size, devices = get_cluster_dims(cluster_index, mesh_shape)

    assert experts % devices == 0
    assert max(ready_device_chunk_counts) <= cluster_size
    assert min(ready_device_chunk_counts) >= 0
    assert len(ready_device_chunk_counts) == devices

    num_experts_per_device = experts // devices

    # initialize this to the complete signal, so the Op doesn't wait for additional batches
    # shard over experts -> recievers.
    combine_semaphore_table = torch.full([experts, cluster_size], CHUNK_COMPLETE_VAL, dtype=torch.uint32)

    # randomly assign chunk entries on different devices to be in a completed state
    # the op waits for whole columns of this table to be ready before proceeding so
    # need to set whole columns as ready
    chunk_indexes = list(range(sum(ready_device_chunk_counts) * num_experts_per_device))
    random.shuffle(chunk_indexes)
    for dest_d, ready_cols in zip(range(devices), ready_device_chunk_counts):
        ready_col_list = list(range(cluster_size))
        random.shuffle(ready_col_list)
        for src_d in ready_col_list[:ready_cols]:
            for local_e_idx in range(num_experts_per_device):
                e_idx = dest_d * num_experts_per_device + local_e_idx
                combine_semaphore_table[e_idx, src_d] = chunk_indexes.pop()

    return combine_semaphore_table


def gen_dense_input_contribs(
    sparse_contribs_tensor, expert_mapping, metadata_tensor, combine_semaphore_table, mesh_shape, cluster_axis
):
    _, cluster_size, devices = get_cluster_dims(cluster_axis, mesh_shape)

    batch = metadata_tensor.shape[1]
    devices = expert_mapping.shape[-1]
    experts = expert_mapping.shape[-2]
    hidden_size = sparse_contribs_tensor.shape[-1]
    seq = sparse_contribs_tensor.shape[-2]
    experts_per_device = experts // devices

    batch_per_device = batch // cluster_size
    chunk_size = batch_per_device * seq

    dense_input_contribs_tensor = torch.zeros([experts, cluster_size, chunk_size, hidden_size]).bfloat16()

    for m0 in range(mesh_shape[0]):
        for m1 in range(mesh_shape[1]):
            rec_d = m0 * mesh_shape[1] + m1
            device_expert_list = get_experts_on_device(experts, expert_mapping, rec_d)
            for local_e_idx in range(experts_per_device):
                global_e_idx = rec_d * experts_per_device + local_e_idx
                expert_id = device_expert_list[local_e_idx]

                for send_d in range(cluster_size):
                    chunk_idx = 0
                    if combine_semaphore_table[global_e_idx, send_d] == CHUNK_COMPLETE_VAL:
                        continue
                    assert not (
                        combine_semaphore_table[rec_d * experts_per_device : (rec_d + 1) * experts_per_device, send_d]
                        == CHUNK_COMPLETE_VAL
                    ).any()
                    for b in range(send_d * batch_per_device, (send_d + 1) * batch_per_device):
                        for s in range(seq):
                            if expert_id not in metadata_tensor[0, b, s]:
                                continue
                            sparse_contrib = sparse_contribs_tensor[global_e_idx, b, s, :]
                            assert not (sparse_contrib == 0.0).all()
                            dense_input_contribs_tensor[global_e_idx, send_d, chunk_idx, :] = sparse_contrib
                            chunk_idx += 1

    return dense_input_contribs_tensor


def gen_output_ref(
    dense_input_contribs_tensor, expert_mapping, metadata_tensor, combine_semaphore_table, mesh_shape, cluster_axis
):
    cluster_factor, cluster_size, devices = get_cluster_dims(cluster_axis, mesh_shape)

    batch = metadata_tensor.shape[1]
    devices = expert_mapping.shape[-1]
    experts = expert_mapping.shape[-2]
    hidden_size = dense_input_contribs_tensor.shape[-1]
    seq = metadata_tensor.shape[2]

    # selected_experts_k = metadata_tensor.shape[-1] # not needed?

    num_experts_per_device = experts // devices
    batch_per_device = batch // cluster_size

    output_ref_tensor = torch.zeros(cluster_size, batch * cluster_factor, seq, hidden_size).bfloat16()
    output_data_map = torch.zeros(output_ref_tensor.shape[:-1])

    batch_rep_idxr = get_batch_cluster_idxr(cluster_axis, batch)

    for m0 in range(mesh_shape[0]):
        for m1 in range(mesh_shape[1]):
            d = m0 * mesh_shape[1] + m1
            device_expert_list = get_experts_on_device(experts, expert_mapping, d)
            local_combine_semaphores = combine_semaphore_table[
                d * num_experts_per_device : (d + 1) * num_experts_per_device, :
            ]

            rec_cluster_device = d if cluster_axis is None else (m0, m1)[cluster_axis]

            chunk_idx_table = torch.zeros([cluster_size, num_experts_per_device]).to(torch.int32)
            for sender_d in range(cluster_size):
                if (local_combine_semaphores[:, sender_d] == CHUNK_COMPLETE_VAL).all():
                    assert len(set(local_combine_semaphores[:, sender_d].tolist())) == 1
                    continue
                for cb in range(batch_per_device):
                    b = sender_d * batch_per_device + cb
                    for s in range(seq):
                        axis_batch_idx = batch_rep_idxr(m0, m1, b)
                        expert_contribs_accum = torch.zeros((hidden_size)).bfloat16()
                        for local_e_idx, e in enumerate(device_expert_list):
                            if e not in metadata_tensor[0, b, s, :]:
                                continue

                            chunk_idx = chunk_idx_table[sender_d, local_e_idx].item()
                            chunk_idx_table[sender_d, local_e_idx] += 1

                            global_e_idx = d * num_experts_per_device + local_e_idx

                            expert_contrib = dense_input_contribs_tensor[global_e_idx, sender_d, chunk_idx, :]
                            # print(f"rec_d: {d} sender_d {sender_d} cb: {cb} local_e: {local_e_idx} chunk_idx: {chunk_idx}")
                            # print(expert_contrib)

                            expert_contribs_accum += expert_contrib
                            output_data_map[rec_cluster_device, axis_batch_idx, s] = 1

                        if output_data_map[rec_cluster_device, axis_batch_idx, s] == 1:
                            # print(f"d: {d} axis_batch_idx: {axis_batch_idx} {expert_contribs_accum=}")
                            output_ref_tensor[rec_cluster_device, axis_batch_idx, s, :] = expert_contribs_accum

    return output_ref_tensor, output_data_map


def gen_tensors(
    batch,
    experts,
    selected_experts_k,
    hidden_size,
    seq,
    mesh_shape,
    cluster_axis,
    devices,
    ready_device_chunks,  # op waits for all D of a given E set of chunks to proceed
    scheme="random",
):
    _, input_sparse_contribs_tensor, expert_mapping, metadata_tensor, og_combine_output_tensor, _ = gen_tensors_combine(
        batch,
        experts,
        selected_experts_k,
        hidden_size,
        seq,
        mesh_shape,
        cluster_axis,
        devices,
        scheme,
        local_reduce=False,
    )

    combine_semaphore_table = gen_combine_semaphore_table(
        batch, seq, experts, mesh_shape, cluster_axis, ready_device_chunks
    )
    dense_contribs_tensor = gen_dense_input_contribs(
        input_sparse_contribs_tensor, expert_mapping, metadata_tensor, combine_semaphore_table, mesh_shape, cluster_axis
    )
    output_ref, output_data_map = gen_output_ref(
        dense_contribs_tensor, expert_mapping, metadata_tensor, combine_semaphore_table, mesh_shape, cluster_axis
    )

    expert_indices = get_expert_indices(batch, experts, selected_experts_k, seq, mesh_shape, scheme)
    output_ref_ref, _ = gen_reference_reference(
        input_sparse_contribs_tensor, expert_indices, expert_mapping, mesh_shape, cluster_axis, local_reduce=False
    )

    #  print(f"Output ref ref: {output_ref_ref}")
    #     print(f"Output ref: {output_ref}")
    #
    #     print("Dense contribs")
    #     for e in range(experts):
    #         for d in range(mesh_shape[cluster_axis]):
    #             print(f"e: {e} d: {d} {dense_contribs_tensor[e,d,:,:]}")

    return dense_contribs_tensor, combine_semaphore_table, output_ref, output_data_map, output_ref_ref, expert_mapping


NUM_DEVICES = 8


@pytest.mark.parametrize("batch", [64])
@pytest.mark.parametrize("experts", [16])
@pytest.mark.parametrize("selected_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [4])
@pytest.mark.parametrize("seq", [1])
@pytest.mark.parametrize("mesh_shape", [(2, 4)])
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("devices", [NUM_DEVICES])
@pytest.mark.parametrize("ready_device_chunks", [[4 for _ in range(NUM_DEVICES)]])
def test_gen_tensors(
    batch, experts, selected_experts_k, hidden_size, seq, mesh_shape, cluster_axis, devices, ready_device_chunks
):
    (
        dense_contribs_tensor,
        combine_semaphore_table,
        output_ref,
        output_data_map,
        output_ref_ref,
        expert_mapping,
    ) = gen_tensors(
        batch,
        experts,
        selected_experts_k,
        hidden_size,
        seq,
        mesh_shape,
        cluster_axis,
        devices,
        ready_device_chunks,
        scheme="sequential",
    )
