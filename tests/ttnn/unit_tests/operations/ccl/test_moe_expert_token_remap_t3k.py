from math import prod
import random

import pytest
import torch

from tests.ttnn.unit_tests.operations.ccl.test_all_to_all_combine_t3000 import get_experts_on_device
from tests.ttnn.unit_tests.operations.ccl.test_all_to_all_dispatch_t3000 import (
    gen_expert_mapping,
    get_expert_indices,
    get_metadata_tensor,
)


def gen_topk(experts, indices_tensor):
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

    return topk_tensor


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


def gen_tensors(devices, experts, batch, seq, selected_experts_k, mesh_shape, scheme):
    expert_mapping = gen_expert_mapping(experts, devices, scheme)
    expert_indices = get_expert_indices(batch, experts, selected_experts_k, seq, mesh_shape, scheme)
    topk_tensor = gen_topk(experts, expert_indices)

    output = gen_output_expert_token_activation(devices, topk_tensor, expert_indices, expert_mapping)

    metadata_tensor = get_metadata_tensor(expert_indices, expert_mapping, mesh_shape)

    return expert_mapping, metadata_tensor, topk_tensor, output


def test_moe_expert_token_remaps(num_iters, devices, experts, batch, seq, selected_experts_k, mesh_shape, scheme):
    expert_mapping_tensors = []
    expert_metadata_tensors = []
    topk_tensors = []
    output_tensor_goldens_list = []

    for _ in range(num_iters):
        expert_mapping, expert_metadata, topk_tensor, output = gen_tensors(
            devices, experts, batch, seq, selected_experts_k, mesh_shape, scheme
        )

        output_tensor_goldens_list.append(output)

        tt_topk = ttnn.from_torch(
            topk_tensor,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dtype,
            memory_config=input_memory_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device, dim=0),
        )

        tt_expert_mapping = ttnn.from_torch(
            expert_mapping,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=input_memory_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )

        tt_metadata = ttnn.from_torch(
            metadata_tensor,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=input_memory_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )

        topk_tensors.append(tt_input_contribs)
        expert_mapping_tensors.append(tt_expert_mapping)
        metadata_tensors.append(tt_metadata)

    out_tensor_list = []
    for topk, mapping, metadata in zip(topk_tensors, expert_mapping_tensors, expert_metadata_tensors):
        tt_op_out = ttnn.moe_expert_token_remap(topk, mapping, metadata, mesh_device)
        out_tensor_list.append(out_tensor_list)

    for ref, test in zip(out_tensor_list, output_tensor_goldens_list):
        assert_with_pcc(ref, test)


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

    expert_mapping, metadata_tensor, topk_tensor, output = gen_tensors(
        devices, experts, batch, seq, select_experts_k, mesh_shape, scheme
    )

    assert expert_mapping.shape == (1, 1, experts, devices)
    assert metadata_tensor.shape == (devices, batch, seq, select_experts_k)
    assert topk_tensor.shape == (1, batch, seq, experts)
    assert output.shape == (devices, batch, seq, experts_per_device)
