# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


def _expected_moe_routing_outputs(
    routing_weights, non_zero_weight_size, expert_parallel_size, cluster_axis, mesh_shape
):
    non_zero_indices = torch.nonzero(routing_weights.flatten(), as_tuple=False).flatten()
    local_non_zero_size = non_zero_weight_size // expert_parallel_size
    outputs = [torch.zeros_like(routing_weights) for _ in range(mesh_shape[0] * mesh_shape[1])]

    for cluster_index in range(mesh_shape[1 - cluster_axis]):
        for member_index in range(mesh_shape[cluster_axis]):
            coordinate = [0, 0]
            coordinate[cluster_axis] = member_index
            coordinate[1 - cluster_axis] = cluster_index
            device_index = coordinate[0] * mesh_shape[1] + coordinate[1]
            local_start = member_index * local_non_zero_size
            local_indices = non_zero_indices[local_start : local_start + local_non_zero_size]
            outputs[device_index].flatten()[local_indices] = routing_weights.flatten()[local_indices]
    return torch.cat(outputs, dim=0)


def test_point_to_point_golden_selects_nonzero_receiver_shard():
    input_tensors = [torch.full((1, 4), index, dtype=torch.bfloat16) for index in range(4)]
    golden_function = ttnn.get_golden_function(ttnn.point_to_point)

    output = golden_function(
        input_tensors,
        sender_coord=(0, 1),
        receiver_coord=(1, 0),
        _ttnn_golden_mesh_shape=(2, 2),
    )

    assert torch.equal(output, input_tensors[1])
    assert output._ttnn_mesh_index == 2


@pytest.mark.parametrize("cluster_axis, expert_parallel_size", [(0, 2), (1, 4)])
def test_moe_routing_remap_golden_partitions_each_mesh_member(cluster_axis, expert_parallel_size):
    routing_weights = torch.zeros((1, 32), dtype=torch.bfloat16)
    routing_weights[0, [2, 4, 10, 13, 14, 18, 22, 24]] = torch.arange(1, 9, dtype=torch.bfloat16)
    mesh_shape = (2, 4)
    non_zero_weight_size = 8
    golden_function = ttnn.get_golden_function(ttnn.moe_routing_remap)

    output = golden_function(
        routing_weights,
        non_zero_weight_size,
        expert_parallel_size,
        cluster_axis,
        _ttnn_golden_mesh_shape=mesh_shape,
    )
    expected = _expected_moe_routing_outputs(
        routing_weights,
        non_zero_weight_size,
        expert_parallel_size,
        cluster_axis,
        mesh_shape,
    )

    assert torch.equal(output, expected)
    first_next_member = 4 if cluster_axis == 0 else 1
    assert not torch.equal(output[0], output[first_next_member])
