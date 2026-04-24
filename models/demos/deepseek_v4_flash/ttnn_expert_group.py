# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v4_flash.expert_plan import ExpertPlacementPlan, PlannedExpert
from models.demos.deepseek_v4_flash.manifest import load_tt_manifest
from models.demos.deepseek_v4_flash.ttnn_routed_expert import TtRoutedExpertMLP


class TtPlannedRoutedExpertGroup(LightweightModule):
    """Batch-1 decode routed expert group for the current T3K stepping stone.

    The group owns one ``TtRoutedExpertMLP`` per activated expert and places it
    on the expert's primary 1x1 submesh. Inputs and scalar per-expert route
    weights are materialized on each expert submesh per call; only expert
    weights persist for the module lifetime. The final sum is intentionally
    host-side until the router/all-to-all path exists.
    """

    def __init__(
        self,
        *,
        mesh_device,
        plan: ExpertPlacementPlan,
        experts: Mapping[int, TtRoutedExpertMLP],
        submeshes: Mapping[int, object],
    ):
        self.mesh_device = mesh_device
        self.plan = plan
        self.expert_ids = tuple(placement.expert_id for placement in plan.placements)
        expected_keys = set(self.expert_ids)
        if set(experts) != expected_keys:
            raise ValueError(f"experts keys must be {sorted(expected_keys)}, got {sorted(experts)}")
        if set(submeshes) != expected_keys:
            raise ValueError(f"submeshes keys must be {sorted(expected_keys)}, got {sorted(submeshes)}")
        self.experts = dict(experts)
        self.submeshes = dict(submeshes)
        self.hidden_size = _single_expert_dim(self.experts, "hidden_size")
        self.intermediate_size = _single_expert_dim(self.experts, "intermediate_size")

    @classmethod
    def from_preprocessed(
        cls,
        preprocessed_path: str | Path,
        mesh_device,
        plan: ExpertPlacementPlan,
        *,
        layer: int = 0,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ) -> "TtPlannedRoutedExpertGroup":
        mesh_shape = tuple(mesh_device.shape)
        _validate_plan_for_group(plan, mesh_shape)

        preprocessed_path = Path(preprocessed_path)
        swiglu_limit = float(load_tt_manifest(preprocessed_path)["config"].get("swiglu_limit", 0.0))
        experts: dict[int, TtRoutedExpertMLP] = {}
        submeshes = {}
        for placement in plan.placements:
            coord = placement.primary_replica.mesh_coord
            submesh = mesh_device.create_submesh(
                ttnn.MeshShape(1, 1),
                offset=ttnn.MeshCoordinate(*coord),
            )
            submeshes[placement.expert_id] = submesh
            experts[placement.expert_id] = TtRoutedExpertMLP.from_preprocessed(
                preprocessed_path,
                device=submesh,
                layer=layer,
                expert=placement.expert_id,
                dtype=dtype,
                memory_config=memory_config,
                swiglu_limit=swiglu_limit,
            )
        return cls(mesh_device=mesh_device, plan=plan, experts=experts, submeshes=submeshes)

    def forward(self, hidden_states: torch.Tensor, route_weights_by_expert: Mapping[int, torch.Tensor]) -> torch.Tensor:
        return self.run_torch_host_combine(hidden_states, route_weights_by_expert)

    def run_torch_host_combine(
        self,
        hidden_states: torch.Tensor,
        route_weights_by_expert: Mapping[int, torch.Tensor],
    ) -> torch.Tensor:
        """Run planned experts serially and return the host-side combined output."""

        _validate_torch_decode_inputs(hidden_states, route_weights_by_expert, self.expert_ids, self.hidden_size)
        combined = torch.zeros_like(hidden_states, dtype=torch.float32)
        for expert_id in self.expert_ids:
            module = self.experts[expert_id]
            submesh = self.submeshes[expert_id]
            tt_input = ttnn.from_torch(
                hidden_states.to(torch.bfloat16),
                device=submesh,
                dtype=module.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=module.memory_config,
            )
            tt_route_weights = ttnn.from_torch(
                _expand_route_weights(route_weights_by_expert[expert_id], module.intermediate_size),
                device=submesh,
                dtype=module.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=module.memory_config,
            )
            combined += ttnn.to_torch(module(tt_input, route_weight=tt_route_weights)).float()
        return combined.to(hidden_states.dtype)


def _validate_plan_for_group(plan: ExpertPlacementPlan, mesh_shape: tuple[int, int]) -> None:
    if plan.mesh_shape != mesh_shape:
        raise ValueError(f"Plan mesh_shape {plan.mesh_shape} does not match mesh device shape {mesh_shape}")
    if not plan.placements:
        raise ValueError("Plan must contain at least one planned expert")
    if tuple(placement.expert_id for placement in plan.placements) != plan.activated_expert_ids:
        raise ValueError("Plan placements must match activated_expert_ids order")

    primary_coords: set[tuple[int, int]] = set()
    for placement in plan.placements:
        _validate_primary_replica(placement, mesh_shape)
        coord = placement.primary_replica.mesh_coord
        if coord in primary_coords:
            raise ValueError(f"Multiple planned experts use primary submesh coordinate {coord}")
        primary_coords.add(coord)


def _validate_primary_replica(placement: PlannedExpert, mesh_shape: tuple[int, int]) -> None:
    if len(placement.replicas) == 0:
        raise ValueError(f"Planned expert {placement.expert_id} has no replicas")
    row, col = placement.primary_replica.mesh_coord
    if row < 0 or row >= mesh_shape[0] or col < 0 or col >= mesh_shape[1]:
        raise ValueError(
            f"Planned expert {placement.expert_id} primary replica {placement.primary_replica.mesh_coord} "
            f"is outside mesh shape {mesh_shape}"
        )


def _validate_torch_decode_inputs(
    hidden_states: torch.Tensor,
    route_weights_by_expert: Mapping[int, torch.Tensor],
    expert_ids: tuple[int, ...],
    hidden_size: int,
) -> None:
    if hidden_states.ndim != 4 or hidden_states.shape[0] != 1 or hidden_states.shape[1] != 1:
        raise ValueError(f"hidden_states must have shape [1, 1, tokens, hidden], got {tuple(hidden_states.shape)}")
    if hidden_states.shape[-1] != hidden_size:
        raise ValueError(f"hidden_states hidden dim must be {hidden_size}, got {hidden_states.shape[-1]}")

    expected_keys = set(expert_ids)
    actual_keys = set(route_weights_by_expert)
    if actual_keys != expected_keys:
        raise ValueError(f"route_weights_by_expert keys must be {sorted(expected_keys)}, got {sorted(actual_keys)}")

    expected_shape = (1, hidden_states.shape[2], 1)
    for expert_id, route_weights in route_weights_by_expert.items():
        if tuple(route_weights.shape) != expected_shape:
            raise ValueError(
                f"route weights for expert {expert_id} must have shape {expected_shape}, "
                f"got {tuple(route_weights.shape)}"
            )


def _expand_route_weights(route_weights: torch.Tensor, intermediate_size: int) -> torch.Tensor:
    return route_weights.reshape(1, 1, route_weights.shape[1], 1).expand(
        1, 1, route_weights.shape[1], intermediate_size
    )


def _single_expert_dim(experts: Mapping[int, TtRoutedExpertMLP], field: str) -> int:
    if not experts:
        raise ValueError("Planned expert group requires at least one expert")
    values = {int(getattr(expert, field)) for expert in experts.values()}
    if len(values) != 1:
        raise ValueError(f"Planned experts must share {field}, got {sorted(values)}")
    return values.pop()
