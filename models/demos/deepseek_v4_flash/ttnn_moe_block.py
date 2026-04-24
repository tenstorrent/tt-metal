# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v4_flash.expert_plan import ExpertPlacementPlan, plan_batch1_decode_expert_placements
from models.demos.deepseek_v4_flash.ttnn_expert_group import (
    TtPlannedRoutedExpertGroup,
    route_weights_by_expert,
    unique_route_expert_ids,
)
from models.demos.deepseek_v4_flash.ttnn_router import TtRouter
from models.demos.deepseek_v4_flash.ttnn_shared_expert import TtSharedExpertMLP


@dataclass(frozen=True)
class MoEDecodePlan:
    expert_plan: ExpertPlacementPlan
    route_weights_by_expert: dict[int, torch.Tensor]


class TtMoEFeedForwardBlock(LightweightModule):
    """Batch-1 decode MoE FFN stepping stone.

    The block runs the TTNN router gate projection, builds a routed expert plan
    from the selected unique expert IDs, runs the planned routed experts, runs
    the shared expert on the primary 1x1 submesh, and returns the host-side sum.
    The final combine is intentionally host-side until the decode all-to-all and
    device-side reduction path exists.

    Routed expert groups are constructed from the route selected by each call;
    no persistent TTNN tensor cache is added in this stepping stone.
    """

    def __init__(
        self,
        *,
        preprocessed_path: str | Path,
        mesh_device,
        primary_submesh,
        primary_submesh_coord: tuple[int, int],
        router: TtRouter,
        shared_expert: TtSharedExpertMLP,
        layer: int = 0,
        replicas_per_expert: int = 1,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        self.preprocessed_path = Path(preprocessed_path)
        self.mesh_device = mesh_device
        self.mesh_shape = tuple(int(dim) for dim in mesh_device.shape)
        self.primary_submesh = primary_submesh
        self.primary_submesh_coord = tuple(int(dim) for dim in primary_submesh_coord)
        self.router = router
        self.shared_expert = shared_expert
        self.layer = int(layer)
        self.replicas_per_expert = int(replicas_per_expert)
        self.dtype = dtype
        self.memory_config = memory_config
        self.hidden_size = int(router.hidden_size)
        if self.hidden_size != int(shared_expert.hidden_size):
            raise ValueError(
                f"router hidden size {self.hidden_size} does not match shared expert input "
                f"dim {int(shared_expert.hidden_size)}"
            )

    @classmethod
    def from_preprocessed(
        cls,
        preprocessed_path: str | Path,
        *,
        mesh_device,
        layer: int = 0,
        primary_submesh_coord: tuple[int, int] | None = None,
        primary_submesh=None,
        replicas_per_expert: int = 1,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ) -> "TtMoEFeedForwardBlock":
        mesh_shape = tuple(int(dim) for dim in mesh_device.shape)
        if primary_submesh_coord is None:
            primary_submesh_coord = default_moe_primary_submesh_coord(mesh_shape)
        _validate_submesh_coord(primary_submesh_coord, mesh_shape)

        if primary_submesh is None:
            primary_submesh = mesh_device.create_submesh(
                ttnn.MeshShape(1, 1),
                offset=ttnn.MeshCoordinate(*primary_submesh_coord),
            )
        router = TtRouter.from_preprocessed(
            preprocessed_path,
            device=primary_submesh,
            layer=layer,
            dtype=dtype,
            memory_config=memory_config,
        )
        shared_expert = TtSharedExpertMLP.from_preprocessed(
            preprocessed_path,
            device=primary_submesh,
            layer=layer,
            dtype=dtype,
            memory_config=memory_config,
        )
        return cls(
            preprocessed_path=preprocessed_path,
            mesh_device=mesh_device,
            primary_submesh=primary_submesh,
            primary_submesh_coord=primary_submesh_coord,
            router=router,
            shared_expert=shared_expert,
            layer=layer,
            replicas_per_expert=replicas_per_expert,
            dtype=dtype,
            memory_config=memory_config,
        )

    def forward(self, hidden_states, *, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        _validate_decode_hidden_states(hidden_states, hidden_size=self.hidden_size)
        tt_router_input, torch_hidden_states = self._materialize_inputs(hidden_states)

        route_weights, route_indices = self.router(tt_router_input, input_ids=input_ids)
        decode_plan = build_moe_decode_plan(
            route_weights,
            route_indices,
            mesh_shape=self.mesh_shape,
            replicas_per_expert=self.replicas_per_expert,
        )
        validate_moe_primary_submesh_disjoint(self.primary_submesh_coord, decode_plan.expert_plan)

        routed_group = TtPlannedRoutedExpertGroup.from_preprocessed(
            self.preprocessed_path,
            self.mesh_device,
            decode_plan.expert_plan,
            layer=self.layer,
            dtype=self.dtype,
            memory_config=self.memory_config,
        )
        try:
            routed_output = routed_group.run_torch_host_combine(
                torch_hidden_states,
                decode_plan.route_weights_by_expert,
            )
        finally:
            _close_routed_group_submeshes(routed_group)
        shared_output = self._run_shared_expert(torch_hidden_states)
        return (routed_output.float() + shared_output.float()).to(torch_hidden_states.dtype)

    def _materialize_inputs(self, hidden_states) -> tuple[object, torch.Tensor]:
        if isinstance(hidden_states, torch.Tensor):
            torch_hidden_states = hidden_states.contiguous()
            tt_router_input = ttnn.from_torch(
                torch_hidden_states,
                device=self.primary_submesh,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=self.memory_config,
            )
            return tt_router_input, torch_hidden_states
        return hidden_states, ttnn.to_torch(hidden_states).contiguous()

    def _run_shared_expert(self, torch_hidden_states: torch.Tensor) -> torch.Tensor:
        tt_shared_input = ttnn.from_torch(
            torch_hidden_states,
            device=self.primary_submesh,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,
        )
        return ttnn.to_torch(self.shared_expert(tt_shared_input))


def build_moe_decode_plan(
    route_weights: torch.Tensor,
    route_indices: torch.Tensor,
    *,
    mesh_shape: tuple[int, int],
    replicas_per_expert: int = 1,
) -> MoEDecodePlan:
    active_expert_ids = unique_route_expert_ids(route_indices)
    expert_plan = plan_batch1_decode_expert_placements(
        mesh_shape,
        active_expert_ids,
        replicas_per_expert=replicas_per_expert,
    )
    return MoEDecodePlan(
        expert_plan=expert_plan,
        route_weights_by_expert=route_weights_by_expert(route_weights, route_indices, active_expert_ids),
    )


def default_moe_primary_submesh_coord(mesh_shape: tuple[int, int]) -> tuple[int, int]:
    if len(mesh_shape) != 2:
        raise ValueError(f"mesh_shape must be (rows, cols), got {mesh_shape}")
    rows, cols = mesh_shape
    if rows <= 0 or cols <= 0:
        raise ValueError(f"mesh_shape dimensions must be positive, got {mesh_shape}")
    return (0, cols - 1)


def _validate_decode_hidden_states(hidden_states, *, hidden_size: int) -> None:
    shape = tuple(hidden_states.shape)
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 1:
        raise ValueError(f"hidden_states must have shape [1, 1, tokens, hidden], got {shape}")
    if shape[-1] != hidden_size:
        raise ValueError(f"hidden_states hidden dim must be {hidden_size}, got {shape[-1]}")
    if shape[-2] == 0:
        raise ValueError("hidden_states must contain at least one token")


def _validate_submesh_coord(coord: tuple[int, int], mesh_shape: tuple[int, int]) -> None:
    if len(coord) != 2:
        raise ValueError(f"primary_submesh_coord must be (row, col), got {coord}")
    row, col = coord
    if row < 0 or row >= mesh_shape[0] or col < 0 or col >= mesh_shape[1]:
        raise ValueError(f"primary_submesh_coord {coord} is outside mesh shape {mesh_shape}")


def validate_moe_primary_submesh_disjoint(primary_coord: tuple[int, int], plan: ExpertPlacementPlan) -> None:
    planned_primary_coords = {placement.primary_replica.mesh_coord for placement in plan.placements}
    if primary_coord in planned_primary_coords:
        raise ValueError(
            f"primary_submesh_coord {primary_coord} overlaps a routed expert primary submesh; "
            "choose a disjoint router/shared expert submesh"
        )


def _close_routed_group_submeshes(routed_group: TtPlannedRoutedExpertGroup) -> None:
    for submesh in routed_group.submeshes.values():
        ttnn.synchronize_device(submesh)
        ttnn.close_mesh_device(submesh)
