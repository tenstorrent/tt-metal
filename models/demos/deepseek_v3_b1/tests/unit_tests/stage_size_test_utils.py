# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ttnn
from models.demos.deepseek_v3_b1.demo.pipeline_routing import (
    EdgeTransport,
    EndpointKind,
    GlobalStageEdge,
    GlobalStageHostIoPlan,
    LocalRole,
    LocalStageEdge,
    LocalStageSocketPlan,
    StageEdgeSet,
    StageEndpointRef,
    StageHostIoPlan,
    build_global_stage_edge_set,
    build_local_stage_socket_plan,
    build_local_stage_socket_plans,
    build_stage_host_io,
    project_local_host_io,
    project_local_stage_edge,
    rank_owns_stage_context,
)
from models.demos.deepseek_v3_b1.demo.stage_family import (
    StageFamily,
    fabric_config_for_stage_family,
    query_global_stage_mesh_shape,
    stage_family_from_shape,
)


def create_fabric_router_config(max_payload_size: int) -> Any:
    """Create a FabricRouterConfig with the requested max payload size."""

    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


@dataclass(frozen=True)
class PhysicalTopologyConfig:
    """Static test inputs needed to open the mesh and configure fabric."""

    name: str
    stage_family: StageFamily
    mesh_device_param: tuple[int, int]
    fabric_config: ttnn.FabricConfig
    initialize_loopback: bool = True
    fabric_router_max_payload_size: int | None = None

    def make_device_params(self) -> dict[str, Any]:
        device_params: dict[str, Any] = {"fabric_config": self.fabric_config}
        if self.fabric_router_max_payload_size is not None:
            device_params["fabric_router_config"] = create_fabric_router_config(self.fabric_router_max_payload_size)
        return device_params


def get_generic_stage_size_loopback_topology_config() -> PhysicalTopologyConfig:
    """Return the loopback smoke topology config derived from the selected MGD."""

    mesh_shape = query_global_stage_mesh_shape()
    stage_family = stage_family_from_shape(mesh_shape)
    return PhysicalTopologyConfig(
        name=f"generic-{stage_family.value}-loopback",
        stage_family=stage_family,
        mesh_device_param=(int(mesh_shape[0]), int(mesh_shape[1])),
        fabric_config=fabric_config_for_stage_family(stage_family),
        initialize_loopback=True,
        fabric_router_max_payload_size=15232,
    )


__all__ = [
    "EdgeTransport",
    "EndpointKind",
    "GlobalStageEdge",
    "GlobalStageHostIoPlan",
    "LocalRole",
    "LocalStageEdge",
    "LocalStageSocketPlan",
    "PhysicalTopologyConfig",
    "StageEdgeSet",
    "StageEndpointRef",
    "StageFamily",
    "StageHostIoPlan",
    "build_global_stage_edge_set",
    "build_local_stage_socket_plan",
    "build_local_stage_socket_plans",
    "build_stage_host_io",
    "create_fabric_router_config",
    "fabric_config_for_stage_family",
    "get_generic_stage_size_loopback_topology_config",
    "project_local_host_io",
    "project_local_stage_edge",
    "rank_owns_stage_context",
    "stage_family_from_shape",
]
