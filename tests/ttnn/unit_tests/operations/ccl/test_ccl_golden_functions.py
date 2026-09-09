# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import itertools

import pytest
import torch

import ttnn


def _tensor_topology(*, mesh_shape=(2, 2), shard_dims=(0, 1)):
    mesh_coords = tuple(
        ttnn.MeshCoordinate(*coordinate) for coordinate in itertools.product(*(range(d) for d in mesh_shape))
    )
    return ttnn.TensorTopologySnapshot(
        distribution_shape=mesh_shape,
        placements=tuple(
            ttnn.PlacementReplicate() if shard_dim is None else ttnn.PlacementShard(shard_dim)
            for shard_dim in shard_dims
        ),
        mesh_coords=mesh_coords,
    )


def _distributed_golden(shards, *, mesh_shape=(2, 2), shard_dims=(0, 1), global_value=None):
    topology = _tensor_topology(mesh_shape=mesh_shape, shard_dims=shard_dims)
    return ttnn.DistributedGolden(
        topology=topology,
        global_value=global_value,
        shards=dict(zip(topology.mesh_coords, shards)),
    )


def test_mesh_value_round_trip_with_uneven_shards():
    topology = _tensor_topology()
    mesh_coords = topology.mesh_coords
    global_value = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    shard_shapes = {
        mesh_coords[0]: (1, 2),
        mesh_coords[1]: (1, 3),
        mesh_coords[2]: (2, 2),
        mesh_coords[3]: (2, 3),
    }
    expected_slices = {
        mesh_coords[0]: (slice(0, 1), slice(0, 2)),
        mesh_coords[1]: (slice(0, 1), slice(2, 5)),
        mesh_coords[2]: (slice(1, 3), slice(0, 2)),
        mesh_coords[3]: (slice(1, 3), slice(2, 5)),
    }

    shards = ttnn.decompose_mesh_value(
        global_value,
        topology=topology,
        shard_shapes_by_mesh_coord=shard_shapes,
    )

    assert set(shards) == set(mesh_coords)
    for mesh_coord, shard_slice in expected_slices.items():
        assert torch.equal(shards[mesh_coord], global_value[shard_slice])
    global_value[0, 0] = -1
    assert shards[mesh_coords[0]][0, 0].item() == -1
    assert torch.equal(ttnn.compose_mesh_value(shards_by_mesh_coord=shards, topology=topology), global_value)


def test_mesh_value_round_trip_when_mesh_axes_shard_same_dimension():
    topology = _tensor_topology(shard_dims=(0, 0))
    mesh_coords = topology.mesh_coords
    global_value = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    shard_extents = (1, 3, 2, 4)
    shard_shapes = {mesh_coord: (shard_extent, 2) for mesh_coord, shard_extent in zip(mesh_coords, shard_extents)}

    shards = ttnn.decompose_mesh_value(
        global_value,
        topology=topology,
        shard_shapes_by_mesh_coord=shard_shapes,
    )

    offsets = (0, 1, 4, 6, 10)
    for index, mesh_coord in enumerate(mesh_coords):
        assert torch.equal(shards[mesh_coord], global_value[offsets[index] : offsets[index + 1]])
    assert torch.equal(ttnn.compose_mesh_value(shards_by_mesh_coord=shards, topology=topology), global_value)


def test_compose_mesh_value_validates_replicated_shards(expect_error):
    topology = _tensor_topology(shard_dims=(0, None))
    mesh_coords = topology.mesh_coords
    first_partition = torch.arange(4, dtype=torch.float32).reshape(1, 4)
    second_partition = torch.arange(8, dtype=torch.float32).reshape(2, 4) + 4
    shards = {
        mesh_coords[0]: first_partition,
        mesh_coords[1]: first_partition.clone(),
        mesh_coords[2]: second_partition,
        mesh_coords[3]: second_partition.clone(),
    }

    composed = ttnn.compose_mesh_value(shards_by_mesh_coord=shards, topology=topology)

    assert torch.equal(composed, torch.cat((first_partition, second_partition)))

    shards[mesh_coords[1]] = first_partition + 1
    with expect_error(ValueError, "differs from its replica group"):
        ttnn.compose_mesh_value(shards_by_mesh_coord=shards, topology=topology)


def test_mesh_value_helpers_support_partial_replica_coordinate_sets():
    topology = _tensor_topology(shard_dims=(0, None))
    mesh_coords = topology.mesh_coords
    global_value = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    local_shapes = {
        mesh_coords[1]: (1, 4),
        mesh_coords[2]: (2, 4),
    }

    local_shards = ttnn.decompose_mesh_value(
        global_value,
        topology=topology,
        shard_shapes_by_mesh_coord=local_shapes,
    )

    assert set(local_shards) == set(local_shapes)
    assert torch.equal(local_shards[mesh_coords[1]], global_value[:1])
    assert torch.equal(local_shards[mesh_coords[2]], global_value[1:])
    assert torch.equal(
        ttnn.compose_mesh_value(shards_by_mesh_coord=local_shards, topology=topology),
        global_value,
    )


def test_mesh_value_helpers_reject_incomplete_sharded_coverage(expect_error):
    topology = _tensor_topology()
    mesh_coords = topology.mesh_coords
    incomplete_shapes = {
        mesh_coords[0]: (1, 2),
        mesh_coords[1]: (1, 3),
        mesh_coords[2]: (2, 2),
    }

    with expect_error(ValueError, "do not cover every sharded region"):
        ttnn.decompose_mesh_value(
            torch.zeros((3, 5)),
            topology=topology,
            shard_shapes_by_mesh_coord=incomplete_shapes,
        )


def test_mesh_value_helpers_validate_replicated_extents_and_global_shape(expect_error):
    topology = _tensor_topology(shard_dims=(0, None))
    mesh_coords = topology.mesh_coords
    inconsistent_replica_shapes = {
        mesh_coords[0]: (1, 4),
        mesh_coords[1]: (1, 5),
        mesh_coords[2]: (2, 4),
        mesh_coords[3]: (2, 5),
    }

    with expect_error(ValueError, "Replicated tensor dimension 1 has inconsistent shard extents"):
        ttnn.decompose_mesh_value(
            torch.zeros((3, 4)),
            topology=topology,
            shard_shapes_by_mesh_coord=inconsistent_replica_shapes,
        )

    valid_local_shapes = {
        mesh_coords[1]: (1, 4),
        mesh_coords[2]: (2, 4),
    }
    with expect_error(ValueError, r"Global value has shape \(4, 4\), expected \(3, 4\)"):
        ttnn.decompose_mesh_value(
            torch.zeros((4, 4)),
            topology=topology,
            shard_shapes_by_mesh_coord=valid_local_shapes,
        )


def _two_group_collective_inputs():
    return [
        torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
        torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16),
        torch.tensor([[10.0, 20.0]], dtype=torch.bfloat16),
        torch.tensor([[30.0, 40.0]], dtype=torch.bfloat16),
    ]


def test_all_broadcast_golden_composes_every_collective_group():
    golden_function = ttnn.get_golden_function(ttnn.all_broadcast)

    outputs = golden_function(
        _distributed_golden(_two_group_collective_inputs()),
        cluster_axis=1,
    )

    assert len(outputs) == 2
    assert torch.equal(outputs[0].global_value, torch.tensor([[1.0, 2.0], [10.0, 20.0]], dtype=torch.bfloat16))
    assert torch.equal(outputs[1].global_value, torch.tensor([[3.0, 4.0], [30.0, 40.0]], dtype=torch.bfloat16))


def test_all_gather_golden_composes_every_collective_group():
    golden_function = ttnn.get_golden_function(ttnn.all_gather)

    output = golden_function(
        _distributed_golden(_two_group_collective_inputs()),
        dim=1,
        cluster_axis=1,
    )

    expected = torch.tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]], dtype=torch.bfloat16)
    assert torch.equal(output.global_value, expected)


def test_all_reduce_golden_composes_every_collective_group():
    golden_function = ttnn.get_golden_function(ttnn.all_reduce)

    output = golden_function(
        _distributed_golden(_two_group_collective_inputs()),
        cluster_axis=1,
    )

    assert torch.equal(output.global_value, torch.tensor([[4.0, 6.0], [40.0, 60.0]], dtype=torch.bfloat16))


def test_reduce_scatter_golden_composes_every_rank_chunk():
    golden_function = ttnn.get_golden_function(ttnn.reduce_scatter)

    output = golden_function(
        _distributed_golden(_two_group_collective_inputs()),
        dim=1,
        cluster_axis=1,
    )

    assert torch.equal(output.global_value, torch.tensor([[4.0, 6.0], [40.0, 60.0]], dtype=torch.bfloat16))


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
    root_coord = ttnn.MeshCoordinate(1, 0)

    output_l, output_s, output_m = golden_function(
        _distributed_golden(input_tensors_l, shard_dims=(None, None)),
        _distributed_golden(input_tensors_s, shard_dims=(None, None)),
        _distributed_golden(input_tensors_m, shard_dims=(None, None)),
        root_coord=root_coord,
    )

    assert torch.equal(output_l.shards[root_coord], torch.full_like(input_tensors_l[0], 2.5))
    assert torch.equal(output_s.shards[root_coord], torch.full_like(input_tensors_s[0], 4.0))
    assert torch.equal(output_m.shards[root_coord], torch.zeros_like(input_tensors_m[0]))
    assert output_l.compare_coords == frozenset({root_coord})
    assert output_s.compare_coords == frozenset({root_coord})
    assert output_m.compare_coords == frozenset({root_coord})


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
    receiver_coord = ttnn.MeshCoordinate(1, 0)

    output = golden_function(
        _distributed_golden(input_tensors, shard_dims=(None, None)),
        sender_coord=(0, 1),
        receiver_coord=receiver_coord,
    )

    assert torch.equal(output.shards[receiver_coord], input_tensors[1])
    assert output.compare_coords == frozenset({receiver_coord})


@pytest.mark.parametrize("cluster_axis, expert_parallel_size", [(0, 2), (1, 4)])
def test_moe_routing_remap_golden_partitions_each_mesh_member(cluster_axis, expert_parallel_size):
    routing_weights = torch.zeros((1, 32), dtype=torch.bfloat16)
    routing_weights[0, [2, 4, 10, 13, 14, 18, 22, 24]] = torch.arange(1, 9, dtype=torch.bfloat16)
    mesh_shape = (2, 4)
    non_zero_weight_size = 8
    golden_function = ttnn.get_golden_function(ttnn.moe_routing_remap)

    output = golden_function(
        _distributed_golden(
            [routing_weights] * 8,
            mesh_shape=mesh_shape,
            shard_dims=(None, None),
            global_value=routing_weights,
        ),
        non_zero_weight_size,
        expert_parallel_size,
        cluster_axis,
    )
    expected = _expected_moe_routing_outputs(
        routing_weights,
        non_zero_weight_size,
        expert_parallel_size,
        cluster_axis,
        mesh_shape,
    )

    assert torch.equal(output.global_value, expected)
    first_next_member = 4 if cluster_axis == 0 else 1
    assert not torch.equal(output.global_value[0], output.global_value[first_next_member])
