# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from models.demos.deepseek_v4_flash.mesh_config import mesh_for_shape


@dataclass(frozen=True)
class ExpertReplica:
    expert_id: int
    activation_index: int
    replica_index: int
    mesh_coord: tuple[int, int]
    linear_device_id: int
    ep_rank: int
    tp_rank: int


@dataclass(frozen=True)
class PlannedExpert:
    expert_id: int
    activation_index: int
    ep_rank: int
    replicas: tuple[ExpertReplica, ...]

    @property
    def primary_replica(self) -> ExpertReplica:
        return self.replicas[0]

    @property
    def replica_coords(self) -> tuple[tuple[int, int], ...]:
        return tuple(replica.mesh_coord for replica in self.replicas)


@dataclass(frozen=True)
class ExpertPlacementPlan:
    mesh_shape: tuple[int, int]
    activated_expert_ids: tuple[int, ...]
    tp_axis: int
    ep_axis: int
    replicas_per_expert: int
    placements: tuple[PlannedExpert, ...]

    @property
    def replicas(self) -> tuple[ExpertReplica, ...]:
        return tuple(replica for placement in self.placements for replica in placement.replicas)

    def devices_for_expert(self, expert_id: int) -> tuple[tuple[int, int], ...]:
        for placement in self.placements:
            if placement.expert_id == expert_id:
                return placement.replica_coords
        raise KeyError(f"Expert {expert_id} is not active in this plan")

    def experts_for_device(self, mesh_coord: tuple[int, int]) -> tuple[int, ...]:
        return tuple(replica.expert_id for replica in self.replicas if replica.mesh_coord == mesh_coord)


def plan_batch1_decode_expert_placements(
    mesh_shape: tuple[int, int],
    activated_expert_ids: Sequence[int],
    *,
    replicas_per_expert: int | None = None,
) -> ExpertPlacementPlan:
    """Plan batch-1 decode expert placement for the current T3K stepping stone.

    The policy is intentionally small and deterministic: preserve top-k order,
    assign activated experts round-robin over decode EP ranks, and replicate each
    expert over TP ranks in that EP group. It captures where tiny expert execution
    can happen now without introducing weight caches or all-to-all plumbing.
    """

    mesh_config = mesh_for_shape(mesh_shape)
    activated = _validate_activated_experts(activated_expert_ids)
    tp_width = mesh_config.mesh_shape[mesh_config.tp_axis]
    ep_width = mesh_config.decode.ep
    if replicas_per_expert is None:
        replicas_per_expert = tp_width
    if replicas_per_expert < 1 or replicas_per_expert > tp_width:
        raise ValueError(f"replicas_per_expert must be in [1, {tp_width}], got {replicas_per_expert}")

    placements = []
    for activation_index, expert_id in enumerate(activated):
        ep_rank = activation_index % ep_width
        replicas = []
        for replica_index in range(replicas_per_expert):
            tp_rank = replica_index
            mesh_coord = _mesh_coord(mesh_config.ep_axis, mesh_config.tp_axis, ep_rank, tp_rank)
            replicas.append(
                ExpertReplica(
                    expert_id=expert_id,
                    activation_index=activation_index,
                    replica_index=replica_index,
                    mesh_coord=mesh_coord,
                    linear_device_id=_linear_device_id(mesh_config.mesh_shape, mesh_coord),
                    ep_rank=ep_rank,
                    tp_rank=tp_rank,
                )
            )
        placements.append(
            PlannedExpert(
                expert_id=expert_id,
                activation_index=activation_index,
                ep_rank=ep_rank,
                replicas=tuple(replicas),
            )
        )

    return ExpertPlacementPlan(
        mesh_shape=mesh_config.mesh_shape,
        activated_expert_ids=activated,
        tp_axis=mesh_config.tp_axis,
        ep_axis=mesh_config.ep_axis,
        replicas_per_expert=replicas_per_expert,
        placements=tuple(placements),
    )


def _validate_activated_experts(activated_expert_ids: Sequence[int]) -> tuple[int, ...]:
    if len(activated_expert_ids) == 0:
        raise ValueError("activated_expert_ids must be non-empty")
    seen = set()
    activated = []
    for expert_id in activated_expert_ids:
        if not isinstance(expert_id, int):
            raise TypeError(f"expert IDs must be integers, got {expert_id!r}")
        if expert_id < 0:
            raise ValueError(f"expert IDs must be >= 0, got {expert_id}")
        if expert_id in seen:
            raise ValueError(f"activated_expert_ids must not contain duplicates, got {expert_id}")
        seen.add(expert_id)
        activated.append(expert_id)
    return tuple(activated)


def _mesh_coord(ep_axis: int, tp_axis: int, ep_rank: int, tp_rank: int) -> tuple[int, int]:
    coord = [0, 0]
    coord[ep_axis] = ep_rank
    coord[tp_axis] = tp_rank
    return tuple(coord)


def _linear_device_id(mesh_shape: tuple[int, int], mesh_coord: tuple[int, int]) -> int:
    return mesh_coord[0] * mesh_shape[1] + mesh_coord[1]
