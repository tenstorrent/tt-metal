# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import itertools

import pytest
import torch

import ttnn


def _mesh_kwargs(*, mesh_shape=(2, 2), shard_dims=(0, 1), mesh_coords=None):
    if mesh_coords is None:
        mesh_coords = tuple(itertools.product(*(range(dimension) for dimension in mesh_shape)))
    return {
        "_ttnn_golden_mesh_shape": mesh_shape,
        "_ttnn_golden_mesh_shard_dims": shard_dims,
        "_ttnn_golden_mesh_coords": mesh_coords,
    }


def _two_group_collective_inputs():
    return [
        torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
        torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16),
        torch.tensor([[10.0, 20.0]], dtype=torch.bfloat16),
        torch.tensor([[30.0, 40.0]], dtype=torch.bfloat16),
    ]


# Verifies collective groups are formed from the physical mesh coordinates the topology
# assigns to each logical distribution position, not from logical positions directly.
def test_collective_groups_follow_logical_distribution_coordinates():
    # Nontrivial permutation mapping logical distribution positions to physical coordinates.
    mesh_coords = ((0, 1), (1, 0), (0, 0), (1, 1))
    # Values are per logical distribution position, as produced by golden preprocessing.
    input_tensors = [
        torch.tensor([1.0], dtype=torch.bfloat16),
        torch.tensor([2.0], dtype=torch.bfloat16),
        torch.tensor([4.0], dtype=torch.bfloat16),
        torch.tensor([8.0], dtype=torch.bfloat16),
    ]

    output = ttnn.get_golden_function(ttnn.all_reduce)(
        input_tensors,
        cluster_axis=0,
        **_mesh_kwargs(mesh_shape=(2, 2), shard_dims=(None, None), mesh_coords=mesh_coords),
    )

    # Physical axis-0 groups are {(0, 0), (1, 0)} and {(0, 1), (1, 1)}, i.e. logical
    # positions {2, 1} summing to 6 and {0, 3} summing to 9. The composed logical tensor
    # is anchored at logical position 0, which belongs to the group summing to 9.
    assert torch.equal(output, torch.tensor([9.0], dtype=torch.bfloat16))


# Checks the all_broadcast golden produces the expected per-group broadcast for every collective group.
def test_all_broadcast_golden_composes_every_collective_group():
    golden_function = ttnn.get_golden_function(ttnn.all_broadcast)

    outputs = golden_function(
        _two_group_collective_inputs(),
        cluster_axis=1,
        **_mesh_kwargs(),
    )

    assert len(outputs) == 2
    assert torch.equal(outputs[0], torch.tensor([[1.0, 2.0], [10.0, 20.0]], dtype=torch.bfloat16))
    assert torch.equal(outputs[1], torch.tensor([[3.0, 4.0], [30.0, 40.0]], dtype=torch.bfloat16))


# Checks the all_gather golden concatenates shards within each collective group independently.
def test_all_gather_golden_composes_every_collective_group():
    golden_function = ttnn.get_golden_function(ttnn.all_gather)

    output = golden_function(
        _two_group_collective_inputs(),
        dim=1,
        cluster_axis=1,
        **_mesh_kwargs(),
    )

    expected = torch.tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]], dtype=torch.bfloat16)
    assert torch.equal(output, expected)


# Ensures all_gather output follows logical shard order even when topology coordinates are permuted.
def test_all_gather_golden_preserves_logical_order_with_permuted_topology_coordinates():
    mesh_coords = ((0, 1), (0, 0), (1, 1), (1, 0))
    # Production preprocessing obtains shards from get_device_tensors() in physical
    # storage order, so arrange the values by physical coordinate and apply the same
    # logical reordering the preprocessor performs.
    physical_storage_shards = [
        torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),  # physical (0, 0) = logical 1
        torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16),  # physical (0, 1) = logical 0
        torch.tensor([[10.0, 20.0]], dtype=torch.bfloat16),  # physical (1, 0) = logical 3
        torch.tensor([[30.0, 40.0]], dtype=torch.bfloat16),  # physical (1, 1) = logical 2
    ]
    input_tensors = ttnn.decorators.reorder_shards_to_logical_order(physical_storage_shards, mesh_coords)

    output = ttnn.get_golden_function(ttnn.all_gather)(
        input_tensors,
        dim=1,
        cluster_axis=1,
        **_mesh_kwargs(mesh_coords=mesh_coords),
    )

    expected = torch.tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]], dtype=torch.bfloat16)
    assert torch.equal(output, expected)


# Verifies collective preprocessing reorders get_device_tensors() shards from physical
# storage order into logical distribution order using the topology mapping.
def test_collective_preprocessing_reorders_physical_shards_to_logical_order(monkeypatch):
    from ttnn.operations.ccl import _preprocess_collective_golden_inputs

    mesh_coords = ((0, 1), (0, 0), (1, 1), (1, 0))

    class _FakeTopology:
        def distribution_shape(self):
            return (2, 2)

        def placements(self):
            return (ttnn.PlacementReplicate(), ttnn.PlacementReplicate())

        def mesh_coords(self):
            return mesh_coords

    class _FakeTensor:
        def tensor_topology(self):
            return _FakeTopology()

    physical_storage_shards = [
        torch.tensor([1.0], dtype=torch.bfloat16),  # physical (0, 0) = logical 1
        torch.tensor([2.0], dtype=torch.bfloat16),  # physical (0, 1) = logical 0
        torch.tensor([4.0], dtype=torch.bfloat16),  # physical (1, 0) = logical 3
        torch.tensor([8.0], dtype=torch.bfloat16),  # physical (1, 1) = logical 2
    ]
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda tensor: physical_storage_shards)
    monkeypatch.setattr(ttnn, "to_torch", lambda tensor: tensor)

    function_args, function_kwargs = _preprocess_collective_golden_inputs((_FakeTensor(),), {})

    logical_shards = function_args[0]
    assert torch.equal(logical_shards[0], physical_storage_shards[1])
    assert torch.equal(logical_shards[1], physical_storage_shards[0])
    assert torch.equal(logical_shards[2], physical_storage_shards[3])
    assert torch.equal(logical_shards[3], physical_storage_shards[2])
    assert function_kwargs["_ttnn_golden_mesh_coords"] == mesh_coords


# Ensures the golden raises when the provided mesh coordinates don't cover the full mesh volume.
def test_collective_golden_rejects_incomplete_topology_coordinates(expect_error):
    with expect_error(ValueError, "mesh coordinates for mesh volume"):
        ttnn.get_golden_function(ttnn.all_reduce)(
            _two_group_collective_inputs(),
            cluster_axis=1,
            **_mesh_kwargs(mesh_coords=((0, 0), (0, 1), (1, 0))),
        )


# Checks the all_reduce golden sums shards within each collective group independently.
def test_all_reduce_golden_composes_every_collective_group():
    golden_function = ttnn.get_golden_function(ttnn.all_reduce)

    output = golden_function(
        _two_group_collective_inputs(),
        cluster_axis=1,
        **_mesh_kwargs(),
    )

    assert torch.equal(output, torch.tensor([[4.0, 6.0], [40.0, 60.0]], dtype=torch.bfloat16))


# Checks the reduce_scatter golden reduces each group and returns the correct per-rank chunk.
def test_reduce_scatter_golden_composes_every_rank_chunk():
    golden_function = ttnn.get_golden_function(ttnn.reduce_scatter)

    output = golden_function(
        _two_group_collective_inputs(),
        dim=1,
        cluster_axis=1,
        **_mesh_kwargs(),
    )

    assert torch.equal(output, torch.tensor([[4.0, 6.0], [40.0, 60.0]], dtype=torch.bfloat16))


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


# Verifies reduce_to_root reduces the per-device l/s/m states onto the root
# and tags the outputs with the root's mesh coordinate.
def test_reduce_to_root_golden_reduces_four_device_states():
    input_tensors_l = [torch.full((1, 1, 1, 32), value, dtype=torch.float32) for value in (1.0, 2.0, 3.0, 4.0)]
    input_tensors_s = [torch.ones((1, 1, 1, 32), dtype=torch.float32) for _ in range(4)]
    input_tensors_m = [torch.zeros((1, 1, 1, 32), dtype=torch.float32) for _ in range(4)]
    golden_function = ttnn.get_golden_function(ttnn.reduce_to_root)
    root_coord = ttnn.MeshCoordinate(1, 0)

    output_l, output_s, output_m = golden_function(
        input_tensors_l,
        input_tensors_s,
        input_tensors_m,
        root_coord=root_coord,
        **_mesh_kwargs(shard_dims=(None, None)),
    )

    assert torch.equal(output_l, torch.full_like(input_tensors_l[0], 2.5))
    assert torch.equal(output_s, torch.full_like(input_tensors_s[0], 4.0))
    assert torch.equal(output_m, torch.zeros_like(input_tensors_m[0]))
    assert output_l._ttnn_mesh_coord == (1, 0)
    assert output_s._ttnn_mesh_coord == (1, 0)
    assert output_m._ttnn_mesh_coord == (1, 0)


def _expected_moe_routing_outputs(
    routing_weights, non_zero_weight_size, expert_parallel_size, cluster_axis, mesh_shape, mesh_coords=None
):
    non_zero_indices = torch.nonzero(routing_weights.flatten(), as_tuple=False).flatten()
    local_non_zero_size = non_zero_weight_size // expert_parallel_size
    outputs = [torch.zeros_like(routing_weights) for _ in range(mesh_shape[0] * mesh_shape[1])]

    if mesh_coords is None:
        mesh_coords = tuple(itertools.product(*(range(dimension) for dimension in mesh_shape)))
    for device_index, coordinate in enumerate(mesh_coords):
        # The expert partition follows the device's physical coordinate along the cluster axis.
        member_index = coordinate[cluster_axis]
        local_start = member_index * local_non_zero_size
        local_indices = non_zero_indices[local_start : local_start + local_non_zero_size]
        outputs[device_index].flatten()[local_indices] = routing_weights.flatten()[local_indices]
    return torch.cat(outputs, dim=0)


# Verifies point_to_point delivers the sender's shard to the receiver coordinate
# and tags the output with the receiver's mesh coordinate.
def test_point_to_point_golden_selects_nonzero_receiver_shard():
    input_tensors = [torch.full((1, 4), index, dtype=torch.bfloat16) for index in range(4)]
    golden_function = ttnn.get_golden_function(ttnn.point_to_point)
    mesh_coords = ((1, 1), (0, 0), (1, 0), (0, 1))

    output = golden_function(
        input_tensors,
        sender_coord=(1, 0),
        receiver_coord=(0, 1),
        **_mesh_kwargs(shard_dims=(None, None), mesh_coords=mesh_coords),
    )

    assert torch.equal(output, input_tensors[2])
    assert output._ttnn_mesh_coord == (0, 1)


# Checks moe_routing_remap partitions non-zero routing weights across mesh members along the cluster axis.
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
        **_mesh_kwargs(mesh_shape=mesh_shape, shard_dims=(None, None)),
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


# Ensures moe_routing_remap output follows logical device order even with permuted topology coordinates.
def test_moe_routing_remap_golden_preserves_logical_order_with_permuted_topology_coordinates():
    routing_weights = torch.zeros((1, 32), dtype=torch.bfloat16)
    routing_weights[0, [2, 4, 10, 13, 14, 18, 22, 24]] = torch.arange(1, 9, dtype=torch.bfloat16)
    mesh_shape = (2, 4)
    mesh_coords = tuple((row, column) for row in range(2) for column in reversed(range(4)))

    output = ttnn.get_golden_function(ttnn.moe_routing_remap)(
        routing_weights,
        non_zero_weight_size=8,
        expert_parallel_size=4,
        cluster_axis=1,
        **_mesh_kwargs(mesh_shape=mesh_shape, shard_dims=(None, None), mesh_coords=mesh_coords),
    )
    expected = _expected_moe_routing_outputs(
        routing_weights,
        non_zero_weight_size=8,
        expert_parallel_size=4,
        cluster_axis=1,
        mesh_shape=mesh_shape,
        mesh_coords=mesh_coords,
    )

    assert torch.equal(output, expected)
