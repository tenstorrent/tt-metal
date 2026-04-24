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
