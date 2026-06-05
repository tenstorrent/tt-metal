# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

import ttnn


class StageFamily(str, Enum):
    STAGE_4X2 = "4x2"
    STAGE_4X4 = "4x4"
    STAGE_8X4 = "8x4"


class EdgeTransport(str, Enum):
    LOCAL = "local"
    SAME_MESH_CROSS_RANK = "same_mesh_cross_rank"
    CROSS_MESH = "cross_mesh"


class LocalRole(str, Enum):
    SENDER = "sender"
    RECEIVER = "receiver"
    BOTH = "both"


class EndpointKind(str, Enum):
    STAGE_ENTRY = "stage_entry"
    STAGE_EXIT = "stage_exit"
    LOOPBACK_ENTRY = "loopback_entry"
    HOST_EGRESS = "host_egress"


def create_fabric_router_config(max_payload_size: int) -> Any:
    """Create a FabricRouterConfig with the requested max payload size."""
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


@dataclass(frozen=True)
class PhysicalTopologyConfig:
    """Static test inputs needed to open the mesh and configure fabric.

    This is intentionally not the source of truth for stage ownership. Runtime ownership and endpoint
    placement come from `resolve_blitz_decode_pipeline_allocation()`. This config only captures the
    physical test scenario we want the fixture layer to request.
    """

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


GENERIC_STAGE_SIZE_LOOPBACK_TOPOLOGY_CONFIGS = {
    StageFamily.STAGE_4X2: PhysicalTopologyConfig(
        name="generic-4x2-loopback",
        stage_family=StageFamily.STAGE_4X2,
        mesh_device_param=(4, 2),
        fabric_config=ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
        initialize_loopback=True,
        fabric_router_max_payload_size=15232,
    ),
    StageFamily.STAGE_4X4: PhysicalTopologyConfig(
        name="generic-4x4-loopback",
        stage_family=StageFamily.STAGE_4X4,
        mesh_device_param=(4, 4),
        fabric_config=ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
        initialize_loopback=True,
        fabric_router_max_payload_size=15232,
    ),
    StageFamily.STAGE_8X4: PhysicalTopologyConfig(
        name="generic-8x4-loopback",
        stage_family=StageFamily.STAGE_8X4,
        mesh_device_param=(8, 4),
        fabric_config=ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
        initialize_loopback=True,
        fabric_router_max_payload_size=15232,
    ),
}


def get_generic_stage_size_loopback_topology_config(stage_family: str | StageFamily) -> PhysicalTopologyConfig:
    """Return the loopback smoke topology config for the requested stage family."""

    resolved_stage_family = stage_family if isinstance(stage_family, StageFamily) else StageFamily(stage_family)
    return GENERIC_STAGE_SIZE_LOOPBACK_TOPOLOGY_CONFIGS[resolved_stage_family]


def get_generic_stage_size_loopback_topology_config_from_env(
    env_var_name: str = "TT_STAGE_FAMILY",
) -> PhysicalTopologyConfig:
    """Return the loopback smoke topology config selected by an environment variable."""

    stage_family_value = os.environ.get(env_var_name, StageFamily.STAGE_4X2.value).strip()
    try:
        return get_generic_stage_size_loopback_topology_config(stage_family_value)
    except ValueError as exc:
        allowed_values = ", ".join(stage_family.value for stage_family in StageFamily)
        raise ValueError(f"Invalid {env_var_name}={stage_family_value!r}. Expected one of: {allowed_values}.") from exc


@dataclass(frozen=True)
class StageEndpointRef:
    """One resolved stage-attached endpoint reference."""

    kind: EndpointKind
    logical_stage_index: int
    mesh_id: int
    placement: Any

    @property
    def owner_rank(self) -> int:
        return int(self.placement.host_binding.rank)


@dataclass(frozen=True)
class LocalStageEdge:
    """One pipeline edge projected into the local rank's stage context."""

    transport: EdgeTransport
    local_role: LocalRole
    src: StageEndpointRef
    dst: StageEndpointRef


@dataclass(frozen=True)
class GlobalStageEdge:
    """One global pipeline edge before any rank-local filtering is applied."""

    transport: EdgeTransport
    src: StageEndpointRef
    dst: StageEndpointRef


@dataclass(frozen=True)
class StageEdgeSet:
    """All global edges associated with one logical stage context."""

    incoming_edge: GlobalStageEdge | None
    intra_stage_edge: GlobalStageEdge | None
    outgoing_edge: GlobalStageEdge | None


@dataclass(frozen=True)
class GlobalStageHostIoPlan:
    """Stage-attached host I/O endpoints before rank-local ownership filtering."""

    h2d_target: StageEndpointRef | None = None
    d2h_source: StageEndpointRef | None = None


@dataclass(frozen=True)
class StageHostIoPlan:
    """Host I/O responsibilities for the local rank within one logical stage context."""

    h2d_target: StageEndpointRef | None = None
    d2h_source: StageEndpointRef | None = None

    @property
    def needs_h2d(self) -> bool:
        return self.h2d_target is not None

    @property
    def needs_d2h(self) -> bool:
        return self.d2h_source is not None


@dataclass(frozen=True)
class LocalStageSocketPlan:
    """Per-stage, per-rank plan derived from the resolved pipeline allocation."""

    logical_stage_index: int
    my_rank: int
    is_split_stage: bool
    incoming_edge: LocalStageEdge | None
    intra_stage_edge: LocalStageEdge | None
    outgoing_edge: LocalStageEdge | None
    host_io: StageHostIoPlan


def _make_stage_endpoint_ref(
    kind: EndpointKind, logical_stage_index: int, mesh_id: int, placement: Any
) -> StageEndpointRef:
    return StageEndpointRef(kind=kind, logical_stage_index=logical_stage_index, mesh_id=mesh_id, placement=placement)


def _classify_edge_transport(src: StageEndpointRef, dst: StageEndpointRef) -> EdgeTransport:
    # Same-stage edges either stay local to one rank or become same-mesh cross-rank relays for split stages.
    if src.logical_stage_index == dst.logical_stage_index:
        if src.owner_rank == dst.owner_rank:
            return EdgeTransport.LOCAL
        return EdgeTransport.SAME_MESH_CROSS_RANK

    # Different logical stages are always treated as pipeline hops across meshes, even if two stages
    # happen to be owned by the same process.
    return EdgeTransport.CROSS_MESH


def _make_global_stage_edge(src: StageEndpointRef, dst: StageEndpointRef) -> GlobalStageEdge:
    return GlobalStageEdge(transport=_classify_edge_transport(src, dst), src=src, dst=dst)


def _local_role_for_edge(src: StageEndpointRef, dst: StageEndpointRef, my_rank: int) -> LocalRole | None:
    if src.owner_rank == my_rank and dst.owner_rank == my_rank:
        return LocalRole.BOTH
    if src.owner_rank == my_rank:
        return LocalRole.SENDER
    if dst.owner_rank == my_rank:
        return LocalRole.RECEIVER
    return None


def _find_local_host_binding(stage: Any, my_rank: int) -> Any | None:
    return next((host_binding for host_binding in stage.host_bindings if int(host_binding.rank) == my_rank), None)


def _build_stage_incoming_edge(allocation: Any, allocation_stage_index: int) -> GlobalStageEdge | None:
    stage = allocation.stages[allocation_stage_index]

    if allocation_stage_index == 0:
        if allocation.loopback_entry_endpoint is None:
            return None
        last_stage = allocation.stages[-1]
        if last_stage.exit_endpoint is None:
            return None
        return _make_global_stage_edge(
            _make_stage_endpoint_ref(
                EndpointKind.STAGE_EXIT,
                last_stage.logical_stage_index,
                int(last_stage.mesh_id),
                last_stage.exit_endpoint,
            ),
            _make_stage_endpoint_ref(
                EndpointKind.LOOPBACK_ENTRY,
                int(allocation.loopback_entry_stage_index),
                int(allocation.stages[int(allocation.loopback_entry_stage_index)].mesh_id),
                allocation.loopback_entry_endpoint,
            ),
        )

    prev_stage = allocation.stages[allocation_stage_index - 1]
    if prev_stage.exit_endpoint is None:
        return None
    return _make_global_stage_edge(
        _make_stage_endpoint_ref(
            EndpointKind.STAGE_EXIT,
            prev_stage.logical_stage_index,
            int(prev_stage.mesh_id),
            prev_stage.exit_endpoint,
        ),
        _make_stage_endpoint_ref(
            EndpointKind.STAGE_ENTRY,
            stage.logical_stage_index,
            int(stage.mesh_id),
            stage.entry_endpoint,
        ),
    )


def _build_stage_intra_stage_edge(allocation: Any, allocation_stage_index: int) -> GlobalStageEdge | None:
    stage = allocation.stages[allocation_stage_index]
    if stage.exit_endpoint is None:
        return None
    return _make_global_stage_edge(
        _make_stage_endpoint_ref(
            EndpointKind.STAGE_ENTRY,
            stage.logical_stage_index,
            int(stage.mesh_id),
            stage.entry_endpoint,
        ),
        _make_stage_endpoint_ref(
            EndpointKind.STAGE_EXIT,
            stage.logical_stage_index,
            int(stage.mesh_id),
            stage.exit_endpoint,
        ),
    )


def _build_stage_outgoing_edge(allocation: Any, allocation_stage_index: int) -> GlobalStageEdge | None:
    stage = allocation.stages[allocation_stage_index]
    if stage.exit_endpoint is None:
        return None

    if allocation_stage_index == len(allocation.stages) - 1:
        if allocation.loopback_entry_endpoint is None:
            return None
        return _make_global_stage_edge(
            _make_stage_endpoint_ref(
                EndpointKind.STAGE_EXIT,
                stage.logical_stage_index,
                int(stage.mesh_id),
                stage.exit_endpoint,
            ),
            _make_stage_endpoint_ref(
                EndpointKind.LOOPBACK_ENTRY,
                int(allocation.loopback_entry_stage_index),
                int(allocation.stages[int(allocation.loopback_entry_stage_index)].mesh_id),
                allocation.loopback_entry_endpoint,
            ),
        )

    next_stage = allocation.stages[allocation_stage_index + 1]
    return _make_global_stage_edge(
        _make_stage_endpoint_ref(
            EndpointKind.STAGE_EXIT,
            stage.logical_stage_index,
            int(stage.mesh_id),
            stage.exit_endpoint,
        ),
        _make_stage_endpoint_ref(
            EndpointKind.STAGE_ENTRY,
            next_stage.logical_stage_index,
            int(next_stage.mesh_id),
            next_stage.entry_endpoint,
        ),
    )


def build_global_stage_edge_set(allocation: Any, allocation_stage_index: int) -> StageEdgeSet:
    """Build the stage-scoped global edges without applying any rank-local ownership filtering."""

    return StageEdgeSet(
        incoming_edge=_build_stage_incoming_edge(allocation, allocation_stage_index),
        intra_stage_edge=_build_stage_intra_stage_edge(allocation, allocation_stage_index),
        outgoing_edge=_build_stage_outgoing_edge(allocation, allocation_stage_index),
    )


def build_stage_host_io(allocation: Any, allocation_stage_index: int) -> GlobalStageHostIoPlan:
    """Build the stage-scoped host I/O endpoints without applying any rank-local ownership filtering."""

    stage = allocation.stages[allocation_stage_index]

    h2d_target = None
    if stage.logical_stage_index == 0:
        h2d_target = _make_stage_endpoint_ref(
            EndpointKind.STAGE_ENTRY,
            stage.logical_stage_index,
            int(stage.mesh_id),
            stage.entry_endpoint,
        )

    d2h_source = None
    if int(allocation.host_egress_stage_index) == stage.logical_stage_index:
        d2h_source = _make_stage_endpoint_ref(
            EndpointKind.HOST_EGRESS,
            int(allocation.host_egress_stage_index),
            int(stage.mesh_id),
            allocation.host_egress_endpoint,
        )

    return GlobalStageHostIoPlan(h2d_target=h2d_target, d2h_source=d2h_source)


def rank_owns_stage_context(
    allocation: Any,
    allocation_stage_index: int,
    my_rank: int,
    edge_set: StageEdgeSet,
    host_io: GlobalStageHostIoPlan,
) -> bool:
    """Return True only when `my_rank` owns this stage context or a stage-attached special endpoint."""

    stage = allocation.stages[allocation_stage_index]
    owns_stage_binding = _find_local_host_binding(stage, my_rank) is not None
    owns_loopback_entry = (
        edge_set.incoming_edge is not None
        and edge_set.incoming_edge.dst.kind == EndpointKind.LOOPBACK_ENTRY
        and edge_set.incoming_edge.dst.owner_rank == my_rank
    )
    owns_h2d = host_io.h2d_target is not None and host_io.h2d_target.owner_rank == my_rank
    owns_d2h = host_io.d2h_source is not None and host_io.d2h_source.owner_rank == my_rank
    return owns_stage_binding or owns_loopback_entry or owns_h2d or owns_d2h


def project_local_stage_edge(edge: GlobalStageEdge | None, my_rank: int, edge_kind: str) -> LocalStageEdge | None:
    """Project one global edge into a local rank-scoped view, or return None if this rank is not responsible."""

    if edge is None:
        return None

    local_role = _local_role_for_edge(edge.src, edge.dst, my_rank)
    if local_role is None:
        return None

    if edge_kind == "incoming" and local_role not in (LocalRole.RECEIVER, LocalRole.BOTH):
        return None
    if edge_kind == "outgoing" and local_role not in (LocalRole.SENDER, LocalRole.BOTH):
        return None

    return LocalStageEdge(
        transport=edge.transport,
        local_role=local_role,
        src=edge.src,
        dst=edge.dst,
    )


def project_local_host_io(host_io: GlobalStageHostIoPlan, my_rank: int) -> StageHostIoPlan:
    """Project the global host I/O description into a local rank-scoped responsibility set."""

    needs_h2d = host_io.h2d_target is not None and host_io.h2d_target.owner_rank == my_rank
    needs_d2h = host_io.d2h_source is not None and host_io.d2h_source.owner_rank == my_rank
    return StageHostIoPlan(
        h2d_target=host_io.h2d_target if needs_h2d else None,
        d2h_source=host_io.d2h_source if needs_d2h else None,
    )


def build_local_stage_socket_plan(
    allocation: Any, allocation_stage_index: int, my_rank: int
) -> LocalStageSocketPlan | None:
    """Build the local socket/endpoint responsibilities for one rank and one logical stage.

    The returned plan is stage-centered:
      - `incoming_edge` is the inbound pipeline edge associated with this stage context
      - `intra_stage_edge` models the relay from stage entry -> stage exit
      - `outgoing_edge` is the outbound pipeline edge from this stage context
      - `host_io` captures H2D/D2H responsibilities

    When the stage is split across ranks, the intra-stage relay becomes a same-mesh cross-rank edge.
    """

    stage = allocation.stages[allocation_stage_index]
    edge_set = build_global_stage_edge_set(allocation, allocation_stage_index)
    host_io = build_stage_host_io(allocation, allocation_stage_index)

    if not rank_owns_stage_context(allocation, allocation_stage_index, my_rank, edge_set, host_io):
        return None

    return LocalStageSocketPlan(
        logical_stage_index=stage.logical_stage_index,
        my_rank=my_rank,
        is_split_stage=len(stage.host_bindings) > 1,
        incoming_edge=project_local_stage_edge(edge_set.incoming_edge, my_rank, "incoming"),
        intra_stage_edge=project_local_stage_edge(edge_set.intra_stage_edge, my_rank, "intra"),
        outgoing_edge=project_local_stage_edge(edge_set.outgoing_edge, my_rank, "outgoing"),
        host_io=project_local_host_io(host_io, my_rank),
    )
