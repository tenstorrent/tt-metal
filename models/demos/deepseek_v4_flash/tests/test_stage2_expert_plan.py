# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from models.demos.deepseek_v4_flash.expert_plan import plan_batch1_decode_expert_placements


def test_plan_2x4_replicates_topk_experts_across_tp_ranks() -> None:
    plan = plan_batch1_decode_expert_placements((2, 4), (7, 2, 5, 3))

    assert plan.mesh_shape == (2, 4)
    assert plan.tp_axis == 1
    assert plan.ep_axis == 0
    assert plan.replicas_per_expert == 4
    assert plan.activated_expert_ids == (7, 2, 5, 3)

    assert [placement.ep_rank for placement in plan.placements] == [0, 1, 0, 1]
    assert plan.devices_for_expert(7) == ((0, 0), (0, 1), (0, 2), (0, 3))
    assert plan.devices_for_expert(2) == ((1, 0), (1, 1), (1, 2), (1, 3))
    assert [replica.linear_device_id for replica in plan.placements[1].replicas] == [4, 5, 6, 7]

    assert plan.experts_for_device((0, 0)) == (7, 5)
    assert plan.experts_for_device((1, 3)) == (2, 3)


def test_plan_1x8_can_limit_replica_count_for_fast_stepping_stones() -> None:
    plan = plan_batch1_decode_expert_placements((1, 8), (4, 1), replicas_per_expert=2)

    assert plan.mesh_shape == (1, 8)
    assert plan.tp_axis == 1
    assert plan.ep_axis == 0
    assert plan.replicas_per_expert == 2
    assert [placement.ep_rank for placement in plan.placements] == [0, 0]
    assert plan.devices_for_expert(4) == ((0, 0), (0, 1))
    assert plan.devices_for_expert(1) == ((0, 0), (0, 1))
    assert plan.experts_for_device((0, 0)) == (4, 1)
    assert plan.experts_for_device((0, 7)) == ()


@pytest.mark.parametrize(
    ("activated_expert_ids", "error_type"),
    [
        ((), ValueError),
        ((1, 1), ValueError),
        ((-1,), ValueError),
        ((1.5,), TypeError),
    ],
)
def test_plan_rejects_invalid_activated_expert_ids(activated_expert_ids, error_type) -> None:
    with pytest.raises(error_type):
        plan_batch1_decode_expert_placements((2, 4), activated_expert_ids)


@pytest.mark.parametrize("replicas_per_expert", [0, 5])
def test_plan_rejects_invalid_replica_counts(replicas_per_expert: int) -> None:
    with pytest.raises(ValueError):
        plan_batch1_decode_expert_placements((2, 4), (0, 1), replicas_per_expert=replicas_per_expert)


def test_plan_rejects_lookup_for_inactive_expert() -> None:
    plan = plan_batch1_decode_expert_placements((2, 4), (0, 1))

    with pytest.raises(KeyError):
        plan.devices_for_expert(2)
