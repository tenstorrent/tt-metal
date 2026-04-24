# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from models.demos.deepseek_v4_flash.expert_plan import plan_batch1_decode_expert_placements
from models.demos.deepseek_v4_flash.ttnn_expert_group import (
    TtPlannedRoutedExpertGroup,
    _expand_route_weights,
    _validate_plan_for_group,
    route_weights_by_expert,
    unique_route_expert_ids,
)
from models.demos.deepseek_v4_flash.ttnn_moe_block import (
    build_moe_decode_plan,
    default_moe_primary_submesh_coord,
    validate_moe_primary_submesh_disjoint,
)


class _FakeExpert:
    hidden_size = 32
    intermediate_size = 32


def _fake_group() -> TtPlannedRoutedExpertGroup:
    plan = plan_batch1_decode_expert_placements((2, 4), (0, 2), replicas_per_expert=1)
    return TtPlannedRoutedExpertGroup(
        mesh_device=object(),
        plan=plan,
        experts={expert_id: _FakeExpert() for expert_id in plan.activated_expert_ids},
        submeshes={expert_id: object() for expert_id in plan.activated_expert_ids},
    )


def test_planned_routed_expert_group_validates_primary_submesh_plan() -> None:
    plan = plan_batch1_decode_expert_placements((2, 4), (0, 2), replicas_per_expert=1)
    _validate_plan_for_group(plan, (2, 4))

    with pytest.raises(ValueError, match="does not match"):
        _validate_plan_for_group(plan, (1, 8))

    duplicate_primary_plan = plan_batch1_decode_expert_placements((2, 4), (0, 1, 2), replicas_per_expert=1)
    with pytest.raises(ValueError, match="primary submesh coordinate"):
        _validate_plan_for_group(duplicate_primary_plan, (2, 4))


def test_planned_routed_expert_group_rejects_bad_host_combine_api_inputs() -> None:
    group = _fake_group()
    hidden_states = torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16)
    route_weights = {
        0: torch.zeros(1, 32, 1, dtype=torch.bfloat16),
        2: torch.zeros(1, 32, 1, dtype=torch.bfloat16),
    }

    with pytest.raises(ValueError, match="hidden_states must have shape"):
        group.run_torch_host_combine(hidden_states.reshape(1, 32, 32), route_weights)

    with pytest.raises(ValueError, match="hidden dim"):
        group.run_torch_host_combine(torch.zeros(1, 1, 32, 16, dtype=torch.bfloat16), route_weights)

    with pytest.raises(ValueError, match="keys must be"):
        group.run_torch_host_combine(hidden_states, {0: route_weights[0]})

    bad_route_weights = dict(route_weights)
    bad_route_weights[2] = torch.zeros(1, 1, 32, 1, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="route weights for expert 2"):
        group.run_torch_host_combine(hidden_states, bad_route_weights)


def test_expand_route_weights_matches_ttnn_routed_expert_shape() -> None:
    route_weights = torch.arange(32, dtype=torch.float32).reshape(1, 32, 1)

    expanded = _expand_route_weights(route_weights, intermediate_size=4)

    assert expanded.shape == (1, 1, 32, 4)
    torch.testing.assert_close(expanded[0, 0, :, 0], route_weights[0, :, 0])
    torch.testing.assert_close(expanded[0, 0, :, 3], route_weights[0, :, 0])


def test_route_weights_by_expert_builds_planned_group_inputs() -> None:
    route_weights = torch.tensor([[[0.2, 0.8], [0.4, 0.6], [0.5, 0.25]]], dtype=torch.float32)
    route_indices = torch.tensor([[[2, 0], [0, 2], [2, 2]]], dtype=torch.int64)

    assert unique_route_expert_ids(route_indices) == (2, 0)

    weights_by_expert = route_weights_by_expert(route_weights, route_indices, expert_ids=(2, 0))

    assert set(weights_by_expert) == {0, 2}
    torch.testing.assert_close(weights_by_expert[0], torch.tensor([[[0.8], [0.4], [0.0]]]))
    torch.testing.assert_close(weights_by_expert[2], torch.tensor([[[0.2], [0.6], [0.75]]]))

    with pytest.raises(ValueError, match="same shape"):
        route_weights_by_expert(route_weights[..., :1], route_indices)
    with pytest.raises(ValueError, match="non-negative"):
        route_weights_by_expert(route_weights, torch.full_like(route_indices, -1))


def test_moe_decode_plan_builds_route_mapping_and_validates_primary_submesh() -> None:
    route_weights = torch.tensor([[[0.2, 0.8], [0.4, 0.6], [0.5, 0.25]]], dtype=torch.float32)
    route_indices = torch.tensor([[[2, 0], [0, 2], [2, 2]]], dtype=torch.int64)

    decode_plan = build_moe_decode_plan(route_weights, route_indices, mesh_shape=(2, 4), replicas_per_expert=1)

    assert decode_plan.expert_plan.activated_expert_ids == (2, 0)
    assert decode_plan.expert_plan.devices_for_expert(2) == ((0, 0),)
    assert decode_plan.expert_plan.devices_for_expert(0) == ((1, 0),)
    torch.testing.assert_close(decode_plan.route_weights_by_expert[0], torch.tensor([[[0.8], [0.4], [0.0]]]))
    torch.testing.assert_close(decode_plan.route_weights_by_expert[2], torch.tensor([[[0.2], [0.6], [0.75]]]))

    assert default_moe_primary_submesh_coord((2, 4)) == (0, 3)
    validate_moe_primary_submesh_disjoint((0, 3), decode_plan.expert_plan)
    with pytest.raises(ValueError, match="overlaps a routed expert"):
        validate_moe_primary_submesh_disjoint((0, 0), decode_plan.expert_plan)
