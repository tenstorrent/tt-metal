# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import random
from math import prod

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal


def _gen_input_routing_weights(non_zero_size, num_cluster_experts):
    assert non_zero_size <= num_cluster_experts
    routing_weights = torch.zeros((1, num_cluster_experts)).bfloat16()
    for i in random.sample(range(num_cluster_experts), k=non_zero_size):
        routing_weights[0, i] = random.random()

    return routing_weights


def _dev_idx(mesh_shape, cluster_axis, cluster, member):
    if cluster_axis == 0:
        return member * mesh_shape[1] + cluster
    if cluster_axis == 1:
        return member + mesh_shape[1] * cluster


def _get_reference_outputs(routing_weights, non_zero_size, expert_parallel_size, cluster_axis, mesh_shape):
    # TODO check this assumption
    assert expert_parallel_size == mesh_shape[cluster_axis]
    num_clusters = mesh_shape[1 - cluster_axis]

    assert non_zero_size % expert_parallel_size == 0
    nnz_per_device = non_zero_size // expert_parallel_size

    outputs = [torch.zeros_like(routing_weights) for _ in range(prod(mesh_shape))]

    for c in range(num_clusters):
        member_idx = 0
        member_nnz = 0
        for i in range(routing_weights.shape[1]):
            if routing_weights[0, i] != 0:
                device_idx = _dev_idx(mesh_shape, cluster_axis, c, member_idx)
                outputs[device_idx][0, i] = routing_weights[0, i]
                member_nnz += 1
            if member_nnz == nnz_per_device:
                member_nnz = 0
                member_idx += 1

    return outputs


def _gen_tensors(non_zero_size, expert_parallel_size, num_cluster_experts, mesh_shape, cluster_axis):
    input_routing_weights = _gen_input_routing_weights(non_zero_size, num_cluster_experts)
    reference_outputs = _get_reference_outputs(
        input_routing_weights, non_zero_size, expert_parallel_size, cluster_axis, mesh_shape
    )

    return input_routing_weights, reference_outputs


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=["mesh_device"])
@pytest.mark.parametrize("non_zero_size", [8])
@pytest.mark.parametrize("expert_parallel_size", [2, 4])
@pytest.mark.parametrize("num_cluster_experts", [32])
@pytest.mark.parametrize("cluster_axis", [0, 1])
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("num_iters", [1])
def test_moe_routing_remap(
    mesh_device,
    non_zero_size,
    expert_parallel_size,
    num_cluster_experts,
    cluster_axis,
    num_iters,
    dtype,
    input_memory_config,
):
    mesh_shape = tuple(mesh_device.shape)

    if expert_parallel_size != mesh_shape[cluster_axis]:
        pytest.skip("expert parallel size should match cluster size")

    for _ in range(num_iters):
        routing_weights_torch, reference_outputs_torch = _gen_tensors(
            non_zero_size, expert_parallel_size, num_cluster_experts, mesh_shape, cluster_axis
        )

        tt_routing_weights = ttnn.from_torch(
            routing_weights_torch,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dtype,
            memory_config=input_memory_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        tt_output = ttnn.moe_routing_remap(tt_routing_weights, non_zero_size, expert_parallel_size, cluster_axis)

        tt_outputs = ttnn.get_device_tensors(tt_output)
        assert len(tt_outputs) == len(reference_outputs_torch)
        for i, (test, ref) in enumerate(zip(tt_outputs, reference_outputs_torch)):
            test_torch = ttnn.to_torch(test)
            assert_with_pcc(test_torch, ref)


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=["mesh_device"])
@pytest.mark.parametrize("non_zero_size", [8])
@pytest.mark.parametrize("expert_parallel_size", [2, 4])
@pytest.mark.parametrize("num_cluster_experts", [32])
@pytest.mark.parametrize("cluster_axis", [0, 1])
def test_gen_tensors(mesh_device, non_zero_size, expert_parallel_size, num_cluster_experts, cluster_axis):
    mesh_shape = tuple(mesh_device.shape)
    # TODO check that this restriction is firm
    if expert_parallel_size != mesh_shape[cluster_axis]:
        pytest.skip("expert parallel size should match cluster size")

    (routing_weights_torch, reference_outputs_torch) = _gen_tensors(
        non_zero_size, expert_parallel_size, num_cluster_experts, mesh_shape, cluster_axis
    )

    assert len(reference_outputs_torch) == prod(mesh_shape)
    assert len(torch.nonzero(routing_weights_torch)) == non_zero_size
    for o in reference_outputs_torch:
        assert len(torch.nonzero(o)) == non_zero_size // expert_parallel_size

    for m in range(mesh_shape[cluster_axis]):
        idxs = set()
        for c in range(mesh_shape[1 - cluster_axis]):
            device_idx = _dev_idx(mesh_shape, cluster_axis, c, m)
            idx = torch.nonzero(reference_outputs_torch[device_idx])
            idx = tuple(i[1] for i in idx.tolist())
            idxs.add(idx)
        assert len(idxs) == 1
