# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


def test_all_to_all_dispatch_golden_masks_placeholder_rows():
    input_tensor = torch.arange(1, 13, dtype=torch.bfloat16).reshape(2, 1, 2, 3)
    expert_indices = torch.tensor([[[[0], [1]]], [[[1], [0]]]], dtype=torch.uint16)
    expert_mapping = torch.tensor([[[[1, 0], [0, 1]]]], dtype=torch.bool)
    golden_function = ttnn.get_golden_function(ttnn.all_to_all_dispatch)

    output_tokens, output_metadata = golden_function(input_tensor, expert_indices, expert_mapping)

    expected_mask = torch.tensor(
        [
            [[True, False], [False, True]],
            [[False, True], [True, False]],
        ]
    )
    expected_tokens = torch.zeros(2, 2, 2, 3, dtype=torch.bfloat16)
    expanded_input = input_tensor[:, 0].unsqueeze(0).expand(2, -1, -1, -1)
    expected_tokens[expected_mask] = expanded_input[expected_mask]
    comparison_config = output_tokens._ttnn_comparison_config

    assert torch.equal(output_tokens, expected_tokens)
    assert torch.equal(output_metadata, expert_indices[:, 0].unsqueeze(0).expand(2, -1, -1, -1))
    assert torch.equal(comparison_config.mask, expected_mask)


def test_all_to_all_combine_golden_masks_duplicate_device_slots():
    input_tensor = torch.tensor([[[[10.0, 11.0]]], [[[20.0, 21.0]]]], dtype=torch.bfloat16)
    expert_metadata = torch.tensor([[[[0, 1, 2]]]], dtype=torch.uint16)
    expert_mapping = torch.tensor([[[[1, 0], [1, 0], [0, 1]]]], dtype=torch.bool)
    golden_function = ttnn.get_golden_function(ttnn.all_to_all_combine)

    output = golden_function(input_tensor, expert_metadata, expert_mapping, local_reduce=True)

    expected = torch.zeros(3, 1, 1, 2, dtype=torch.bfloat16)
    expected[0, 0, 0] = input_tensor[0, 0, 0]
    expected[2, 0, 0] = input_tensor[1, 0, 0]
    expected_mask = torch.tensor([[[True]], [[False]], [[True]]])

    assert torch.equal(output, expected)
    assert torch.equal(output._ttnn_comparison_config.mask, expected_mask)


def test_reduce_to_root_golden_reduces_four_device_states():
    input_tensors_l = [torch.full((1, 1, 1, 32), value, dtype=torch.float32) for value in (1.0, 2.0, 3.0, 4.0)]
    input_tensors_s = [torch.ones((1, 1, 1, 32), dtype=torch.float32) for _ in range(4)]
    input_tensors_m = [torch.zeros((1, 1, 1, 32), dtype=torch.float32) for _ in range(4)]
    golden_function = ttnn.get_golden_function(ttnn.reduce_to_root)

    output_l, output_s, output_m = golden_function(
        input_tensors_l,
        input_tensors_s,
        input_tensors_m,
        root_coord=(1, 0),
        _ttnn_golden_mesh_shape=(2, 2),
    )

    assert torch.equal(output_l, torch.full_like(output_l, 2.5))
    assert torch.equal(output_s, torch.full_like(output_s, 4.0))
    assert torch.equal(output_m, torch.zeros_like(output_m))
    assert output_l._ttnn_mesh_index == 2
    assert output_s._ttnn_mesh_index == 2
    assert output_m._ttnn_mesh_index == 2


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
