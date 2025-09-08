# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import itertools
from math import prod, ceil
import random

import pytest
import torch

import ttnn
from tests.ttnn.unit_tests.operations.ccl.test_all_to_all_combine_t3000 import get_experts_on_device
from tests.ttnn.unit_tests.operations.ccl.test_all_to_all_dispatch_t3000 import (
    gen_expert_mapping,
    get_expert_indices,
    get_metadata_tensor,
)
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal

REDUCTION_SIZE = 16


def gen_topk(experts, indices_tensor, devices):
    # indices_tensor is [batch, 1, seq, k] . Entries are expert indices
    # we want  [1, batch, seq, global_experts] . Here, entries are topk weights for expert at given index
    # (replicated over devices on dim 0)

    indices_shape = indices_tensor.shape
    batch = indices_shape[0]
    seq = indices_shape[2]
    selected_experts_k = indices_shape[3]

    topk_tensor = torch.zeros([1, batch, seq, experts], dtype=torch.bfloat16)

    for b in range(batch):
        for s in range(seq):
            low = random.random()
            # I think these are expected to be descending...
            scores = reversed([low + i / (low + selected_experts_k - 1) for i in range(selected_experts_k)])

            for e, weight in zip(indices_tensor[b, 0, s, :].tolist(), scores):
                topk_tensor[0, b, s, e] = weight

    return topk_tensor.repeat([devices, 1, 1, 1])


def gen_output_expert_token_activation(devices, topk_tensor, indices_tensor, expert_mapping_tensor):
    experts = expert_mapping_tensor.shape[2]

    indices_shape = indices_tensor.shape
    batch = indices_shape[0]
    seq = indices_shape[2]
    selected_experts_k = indices_shape[3]

    assert experts % devices == 0
    experts_per_device = experts // devices

    expert_token_activation_tensor = torch.zeros([devices, batch, seq, experts_per_device])

    for d in range(devices):
        local_expert_list = get_experts_on_device(experts, expert_mapping_tensor, d)
        for b in range(batch):
            for s in range(seq):
                for k in range(selected_experts_k):
                    expert_index = indices_tensor[b, 0, s, k].item()
                    if expert_index in local_expert_list:
                        local_expert_index = local_expert_list.index(expert_index)
                    else:
                        continue

                    expert_token_activation_tensor[d, b, s, local_expert_index] = topk_tensor[0, b, s, expert_index]

    return expert_token_activation_tensor


def gen_reduced_expert_token_activation(expert_token_activation_tensor, reduction_size):
    devices, batch, seq, experts_per_device = expert_token_activation_tensor.shape

    batch_seq = batch * seq
    reduced_batch_seq = ceil(batch_seq / reduction_size)

    rs_expert_token_activation_tensor = expert_token_activation_tensor.reshape((devices, batch_seq, experts_per_device))

    reduced_tensor = torch.zeros((devices, 1, reduced_batch_seq, experts_per_device), dtype=torch.int16)
    for d in range(devices):
        for rbs in range(reduced_batch_seq):
            ridx_start, ridx_end = rbs * reduction_size, (rbs + 1) * reduction_size
            for e in range(experts_per_device):
                reduced_tensor[d, 0, rbs, e] = rs_expert_token_activation_tensor[d, ridx_start:ridx_end, e].any().item()

    return reduced_tensor


def gen_tensors(devices, experts, batch, seq, selected_experts_k, mesh_shape, reduction_size, scheme):
    expert_mapping = gen_expert_mapping(experts, devices, scheme)
    expert_indices = get_expert_indices(batch, experts, selected_experts_k, seq, mesh_shape, scheme)
    topk_tensor = gen_topk(experts, expert_indices, prod(mesh_shape))

    output = gen_output_expert_token_activation(devices, topk_tensor, expert_indices, expert_mapping)
    reduced_output = gen_reduced_expert_token_activation(output, reduction_size)

    metadata_tensor = get_metadata_tensor(expert_indices, expert_mapping, mesh_shape)

    return expert_mapping, metadata_tensor, topk_tensor, output, reduced_output


@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((2, 4), (2, 4), id="2x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize("batches_per_device", [8])
@pytest.mark.parametrize("num_iters", [2])
@pytest.mark.parametrize("seq", [1, 2])
@pytest.mark.parametrize("experts_per_device", [8])
@pytest.mark.parametrize("selected_experts_k", [8])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("scheme", ["sequential", "random"])
def test_moe_expert_token_remaps(
    mesh_device,
    mesh_shape,
    num_iters,
    experts_per_device,
    batches_per_device,
    seq,
    selected_experts_k,
    input_memory_config,
    scheme,
):
    devices = prod(mesh_shape)
    batch = devices * batches_per_device
    experts = devices * experts_per_device

    expert_mapping_tensors = []
    expert_metadata_tensors = []
    topk_tensors = []
    output_tensor_goldens_list = []

    for _ in range(num_iters):
        expert_mapping, metadata_tensor, topk_tensor, output_mapping, output_reduced = gen_tensors(
            devices, experts, batch, seq, selected_experts_k, mesh_shape, REDUCTION_SIZE, scheme
        )

        output_tensor_goldens_list.append((output_mapping, output_reduced))

        tt_topk = ttnn.from_torch(
            topk_tensor,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=input_memory_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )

        assert len(ttnn.get_device_tensors(tt_topk)) == devices

        tt_expert_mapping = ttnn.from_torch(
            expert_mapping,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=input_memory_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        assert len(ttnn.get_device_tensors(tt_expert_mapping)) == devices

        tt_metadata = ttnn.from_torch(
            metadata_tensor,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=input_memory_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )

        assert len(ttnn.get_device_tensors(tt_metadata)) == devices

        topk_tensors.append(tt_topk)
        expert_mapping_tensors.append(tt_expert_mapping)
        expert_metadata_tensors.append(tt_metadata)

    out_tensor_list = []
    for topk, mapping, metadata in zip(topk_tensors, expert_mapping_tensors, expert_metadata_tensors):
        tt_op_out = ttnn.moe_expert_token_remap(topk, mapping, metadata)
        out_tensor_list.append(tt_op_out)

    for (mapping_ref, reduced_ref), (mapping_test, reduced_test) in zip(output_tensor_goldens_list, out_tensor_list):
        mapping_test_torch = ttnn.to_torch(mapping_test, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        assert_with_pcc(mapping_test_torch, mapping_ref)

        reduced_test_torch = ttnn.to_torch(reduced_test, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        assert_equal(reduced_test_torch, reduced_ref)


@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((2, 4), (2, 4), id="2x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize("batches_per_device", [8])
@pytest.mark.parametrize("seq", [1, 2])
@pytest.mark.parametrize("experts_per_device", [8])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("scheme", ["random", "sequential"])
def test_gen_tensors(mesh_device, mesh_shape, experts_per_device, batches_per_device, seq, select_experts_k, scheme):
    devices = prod(mesh_shape)
    batch = batches_per_device * devices
    experts = experts_per_device * devices

    expert_mapping, metadata_tensor, topk_tensor, output, reduced_output = gen_tensors(
        devices, experts, batch, seq, select_experts_k, mesh_shape, REDUCTION_SIZE, scheme
    )

    assert expert_mapping.shape == (1, 1, experts, devices)
    assert metadata_tensor.shape == (devices, batch, seq, select_experts_k)
    assert topk_tensor.shape == (devices, batch, seq, experts)
    assert output.shape == (devices, batch, seq, experts_per_device)
    assert reduced_output.shape == (devices, 1, ceil(batch * seq / REDUCTION_SIZE), experts_per_device)


@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((2, 4), (2, 4), id="2x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize("batches_per_device", [4])
@pytest.mark.parametrize("seq", [1, 2])
@pytest.mark.parametrize("experts_per_device", [8])
@pytest.mark.parametrize("selected_experts_k", [8])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("scheme", ["sequential", "random"])
@pytest.mark.parametrize("kn", [(128, 512), (7 * 1024, 2 * 1024)])
@pytest.mark.parametrize("tile_h", [REDUCTION_SIZE])
@pytest.mark.parametrize("tile_w", [32])
@pytest.mark.parametrize("in1_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("reduction_size", [REDUCTION_SIZE])
def test_moe_expert_token_remap_matmul(
    mesh_device,
    mesh_shape,
    experts_per_device,
    batches_per_device,
    seq,
    selected_experts_k,
    input_memory_config,
    scheme,
    kn,
    tile_h,
    tile_w,
    in1_dtype,
    reduction_size,
):
    devices = prod(mesh_shape)
    batch = devices * batches_per_device
    experts = devices * experts_per_device

    expert_mapping, metadata_tensor, topk_tensor, _output_mapping, _output_reduced = gen_tensors(
        devices, experts, batch, seq, selected_experts_k, mesh_shape, reduction_size, scheme
    )

    tt_topk = ttnn.from_torch(
        topk_tensor,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=input_memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    tt_expert_mapping = ttnn.from_torch(
        expert_mapping,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=input_memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_metadata = ttnn.from_torch(
        metadata_tensor,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=input_memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    m, (k, n) = reduction_size, kn
    b, s = batch, seq
    in0 = torch.randn((1, b * s // reduction_size, m, k), dtype=torch.bfloat16)
    in1 = torch.randn(
        (1, experts_per_device, k, n), dtype=torch.float32 if in1_dtype == ttnn.float32 else torch.bfloat16
    )

    in0_t = ttnn.from_torch(
        in0,
        tile=ttnn.Tile((tile_h, tile_w)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=input_memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    in1_t = ttnn.from_torch(
        in1,
        dtype=in1_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=input_memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    _, sparsity_t = ttnn.moe_expert_token_remap(tt_topk, tt_expert_mapping, tt_metadata)

    sparsity_t_torch = ttnn.to_torch(sparsity_t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    sparsity_t = ttnn.from_torch(
        sparsity_t_torch,
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=input_memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    output_tile = ttnn.Tile([tile_h, tile_w])
    matmul_out = ttnn.sparse_matmul(
        in0_t,
        in1_t,
        sparsity=sparsity_t,
        memory_config=input_memory_config,
        output_tile=output_tile,
    )

    matmul_out_torch = ttnn.to_torch(matmul_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    # Compute matmul using torch for each batch and concatenate the results
    for dev, b, e in itertools.product(
        range(prod(mesh_shape)), range(b * s // reduction_size), range(experts_per_device)
    ):
        if sparsity_t_torch[dev, 0, b, e] == 0.0:
            continue
        in0_batch = in0[0, b, :, :]
        in1_batch = in1[0, e, :, :]
        pt_out = torch.matmul(in0_batch, in1_batch)

        # Compare with output tensor
        expected_pcc = 0.999
        assert_with_pcc(pt_out, matmul_out_torch[dev, b, 0, e, :, :], expected_pcc)
