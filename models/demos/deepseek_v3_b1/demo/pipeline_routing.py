# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Pipeline routing: derive each rank's socket/host-I/O responsibilities from a resolved allocation.

The pipeline is a ring of stages; every MPI rank runs exactly one stage context. The plan is
derived **independently and deterministically on every rank** from the same shared ``allocation``
object (no handshake) — so all ranks agree on transports, socket scoping, and endpoint ownership
without communicating. That determinism is achieved with a two-phase design:

1. Global (rank-agnostic) phase — describe the wiring as it objectively is, regardless of who is
   looking. Stage N's exit connects to stage N+1's entry; that hop is LOCAL / SAME_MESH_CROSS_RANK
   / CROSS_MESH; stage 0 has an H2D target; the egress stage has a D2H source. Every rank computes
   byte-identical global structures. See ``GlobalStageEdge`` / ``StageEdgeSet`` and the
   ``_build_*`` helpers. This phase is module-internal scaffolding.

2. Local (rank-scoped) phase — project the global wiring onto "what does *my* rank do?". For each
   edge: am I the sender, receiver, both, or uninvolved (``_project_local_stage_edge`` stamps a
   ``LocalRole``)? For host I/O: keep only the endpoints I own (``_project_local_host_io``). The
   result is a ``LocalStageSocketPlan`` — the only thing the stage runtime (``pipeline_block/op.py``)
   ever consumes.

Public surface: ``build_local_stage_socket_plans`` (the entry point production calls),
``LocalStageSocketPlan`` and its local types, ``local_input_edge`` / ``local_output_edge``
accessors, and the separate ``StageRouting`` / ``build_stage_routing`` legacy metadata path.
Everything prefixed with ``_`` (and the ``Global*`` types) is the internal phase-1 machinery.

Edges per stage context:
- incoming: previous stage's exit -> this stage's entry (for stage 0: last stage's exit -> loopback entry)
- intra:    this stage's entry -> this stage's exit (only meaningful when a stage is split across ranks)
- outgoing: this stage's exit -> next stage's entry (for the last stage: exit -> loopback entry)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

# --------------------------------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------------------------------


class EdgeTransport(str, Enum):
    """How an edge's two endpoints are connected (drives socket scoping in the stage runtime)."""

    LOCAL = "local"  # same stage, same rank -> no socket needed
    SAME_MESH_CROSS_RANK = "same_mesh_cross_rank"  # same stage, different ranks (a split stage)
    CROSS_MESH = "cross_mesh"  # different stages -> inter-mesh fabric/D2D hop


class LocalRole(str, Enum):
    """This rank's role on a projected edge."""

    SENDER = "sender"
    RECEIVER = "receiver"
    BOTH = "both"  # this rank owns both endpoints (a fully-local edge)


class EndpointKind(str, Enum):
    """What an endpoint represents within a stage context (used to spot the loopback-entry owner)."""

    STAGE_ENTRY = "stage_entry"
    STAGE_EXIT = "stage_exit"
    LOOPBACK_ENTRY = "loopback_entry"
    HOST_EGRESS = "host_egress"


# --------------------------------------------------------------------------------------------------
# Shared endpoint reference
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StageEndpointRef:
    """One resolved stage-attached endpoint (a coordinate plus the host binding that owns it)."""

    kind: EndpointKind
    logical_stage_index: int
    mesh_id: int
    placement: Any

    @property
    def owner_rank(self) -> int:
        return int(self.placement.host_binding.rank)


# --------------------------------------------------------------------------------------------------
# Local (rank-scoped) plan types — the public output consumed by pipeline_block/op.py
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalStageEdge:
    """One pipeline edge projected into the local rank's stage context (adds ``local_role``)."""

    transport: EdgeTransport
    local_role: LocalRole
    src: StageEndpointRef
    dst: StageEndpointRef


@dataclass(frozen=True)
class StageHostIoPlan:
    """Stage-attached host I/O endpoints.

    Before projection (``_build_stage_host_io``) both fields are populated regardless of owner;
    after projection (``_project_local_host_io``) only the endpoints this rank owns remain, so
    ``owns_h2d`` / ``owns_d2h`` answer "does this rank own host I/O here?".
    """

    h2d_target: StageEndpointRef | None = None
    d2h_source: StageEndpointRef | None = None

    @property
    def owns_h2d(self) -> bool:
        return self.h2d_target is not None

    @property
    def owns_d2h(self) -> bool:
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


# --------------------------------------------------------------------------------------------------
# Global (rank-agnostic) intermediate representation — internal phase-1 scaffolding
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GlobalStageEdge:
    """One global pipeline edge before any rank-local filtering is applied (no ``local_role``)."""

    transport: EdgeTransport
    src: StageEndpointRef
    dst: StageEndpointRef


@dataclass(frozen=True)
class StageEdgeSet:
    """All global edges associated with one logical stage context."""

    incoming_edge: GlobalStageEdge | None
    intra_stage_edge: GlobalStageEdge | None
    outgoing_edge: GlobalStageEdge | None


# --------------------------------------------------------------------------------------------------
# StageRouting — separate legacy "stages_metadata" concern (entry/exit owners + coords per stage)
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StageRouting:
    """Endpoint-owner routing for one logical stage."""

    mesh_id: int
    entry_owner_rank: int
    exit_owner_rank: int
    entry_coord: Any
    exit_coord: Any | None

    @property
    def rank(self) -> int:
        """Legacy rank view for non-split stages."""

        if self.entry_owner_rank != self.exit_owner_rank:
            raise ValueError("Split-stage routing has different entry and exit owner ranks")
        return self.entry_owner_rank


def build_stage_routing(allocation: Any) -> dict[int, StageRouting]:
    """Build endpoint-owner routing indexed by logical stage index."""

    routing: dict[int, StageRouting] = {}
    for stage in allocation.stages:
        exit_endpoint = stage.exit_endpoint if stage.exit_endpoint is not None else stage.entry_endpoint
        routing[int(stage.logical_stage_index)] = StageRouting(
            mesh_id=int(stage.mesh_id),
            entry_owner_rank=int(stage.entry_endpoint.host_binding.rank),
            exit_owner_rank=int(exit_endpoint.host_binding.rank),
            entry_coord=stage.entry_endpoint.mesh_coord,
            exit_coord=exit_endpoint.mesh_coord,
        )
    return routing


# --------------------------------------------------------------------------------------------------
# Internal helpers (construction + classification)
# --------------------------------------------------------------------------------------------------


def _make_stage_endpoint_ref(
    kind: EndpointKind, logical_stage_index: int, mesh_id: int, placement: Any
) -> StageEndpointRef:
    return StageEndpointRef(kind=kind, logical_stage_index=logical_stage_index, mesh_id=mesh_id, placement=placement)


def _classify_edge_transport(src: StageEndpointRef, dst: StageEndpointRef) -> EdgeTransport:
    """Same stage + same rank -> LOCAL; same stage + different rank -> SAME_MESH_CROSS_RANK; else CROSS_MESH."""
    if src.logical_stage_index == dst.logical_stage_index:
        if src.owner_rank == dst.owner_rank:
            return EdgeTransport.LOCAL
        return EdgeTransport.SAME_MESH_CROSS_RANK
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


# --------------------------------------------------------------------------------------------------
# Internal phase-1 builders: global edges + host I/O for one stage context
# --------------------------------------------------------------------------------------------------


def _build_stage_incoming_edge(allocation: Any, allocation_stage_index: int) -> GlobalStageEdge | None:
    """Edge feeding this stage: prev stage's exit -> this entry (stage 0: last exit -> loopback entry)."""
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
    """Within-stage edge entry -> exit (only a real hop when the stage is split across ranks)."""
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
    """Edge leaving this stage: this exit -> next entry (last stage: exit -> loopback entry)."""
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


def _build_global_stage_edge_set(allocation: Any, allocation_stage_index: int) -> StageEdgeSet:
    """Bundle a stage's incoming/intra/outgoing global edges (no rank-local filtering)."""

    return StageEdgeSet(
        incoming_edge=_build_stage_incoming_edge(allocation, allocation_stage_index),
        intra_stage_edge=_build_stage_intra_stage_edge(allocation, allocation_stage_index),
        outgoing_edge=_build_stage_outgoing_edge(allocation, allocation_stage_index),
    )


def _build_stage_host_io(allocation: Any, allocation_stage_index: int) -> StageHostIoPlan:
    """Stage-scoped host I/O endpoints, unprojected: H2D on stage 0, D2H on the egress stage."""

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

    return StageHostIoPlan(h2d_target=h2d_target, d2h_source=d2h_source)


# --------------------------------------------------------------------------------------------------
# Internal phase-2 projection: filter the global view down to one rank's responsibilities
# --------------------------------------------------------------------------------------------------


def _rank_owns_stage_context(
    allocation: Any,
    allocation_stage_index: int,
    my_rank: int,
    edge_set: StageEdgeSet,
    host_io: StageHostIoPlan,
) -> bool:
    """True when this rank owns a host binding in the stage, or a stage-attached special endpoint."""

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


def _project_local_stage_edge(edge: GlobalStageEdge | None, my_rank: int, edge_kind: str) -> LocalStageEdge | None:
    """Project one global edge into a local rank-scoped view, dropping edges this rank isn't on.

    An incoming edge is kept only when this rank receives on it; an outgoing edge only when this
    rank sends on it; an intra edge is kept for either role.
    """

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


def _project_local_host_io(host_io: StageHostIoPlan, my_rank: int) -> StageHostIoPlan:
    """Keep only the H2D/D2H endpoints this rank owns (null the rest)."""

    owns_h2d = host_io.h2d_target is not None and host_io.h2d_target.owner_rank == my_rank
    owns_d2h = host_io.d2h_source is not None and host_io.d2h_source.owner_rank == my_rank
    return StageHostIoPlan(
        h2d_target=host_io.h2d_target if owns_h2d else None,
        d2h_source=host_io.d2h_source if owns_d2h else None,
    )


# --------------------------------------------------------------------------------------------------
# Public entry points + accessors
# --------------------------------------------------------------------------------------------------


def build_local_stage_socket_plan(
    allocation: Any, allocation_stage_index: int, my_rank: int
) -> LocalStageSocketPlan | None:
    """Build this rank's socket/endpoint responsibilities for one stage, or None if it owns nothing."""

    stage = allocation.stages[allocation_stage_index]
    edge_set = _build_global_stage_edge_set(allocation, allocation_stage_index)
    host_io = _build_stage_host_io(allocation, allocation_stage_index)

    if not _rank_owns_stage_context(allocation, allocation_stage_index, my_rank, edge_set, host_io):
        return None

    return LocalStageSocketPlan(
        logical_stage_index=stage.logical_stage_index,
        my_rank=my_rank,
        is_split_stage=len(stage.host_bindings) > 1,
        incoming_edge=_project_local_stage_edge(edge_set.incoming_edge, my_rank, "incoming"),
        intra_stage_edge=_project_local_stage_edge(edge_set.intra_stage_edge, my_rank, "intra"),
        outgoing_edge=_project_local_stage_edge(edge_set.outgoing_edge, my_rank, "outgoing"),
        host_io=_project_local_host_io(host_io, my_rank),
    )


def build_local_stage_socket_plans(allocation: Any, my_rank: int) -> list[LocalStageSocketPlan]:
    """Return all local stage plans owned by a rank (one per stage context it participates in)."""

    plans = []
    for stage_idx in range(len(allocation.stages)):
        stage_plan = build_local_stage_socket_plan(allocation, stage_idx, my_rank)
        if stage_plan is not None:
            plans.append(stage_plan)
    return plans


def local_input_edge(stage_plan: LocalStageSocketPlan) -> LocalStageEdge | None:
    """The edge this rank receives its activation on: the incoming edge, or a cross-rank intra edge."""
    if stage_plan.incoming_edge is not None:
        return stage_plan.incoming_edge
    if (
        stage_plan.intra_stage_edge is not None
        and stage_plan.intra_stage_edge.transport != EdgeTransport.LOCAL
        and stage_plan.intra_stage_edge.local_role == LocalRole.RECEIVER
    ):
        return stage_plan.intra_stage_edge
    return None


def local_output_edge(stage_plan: LocalStageSocketPlan) -> LocalStageEdge | None:
    """The edge this rank forwards its activation on: the outgoing edge, or a cross-rank intra edge."""
    if stage_plan.outgoing_edge is not None:
        return stage_plan.outgoing_edge
    if (
        stage_plan.intra_stage_edge is not None
        and stage_plan.intra_stage_edge.transport != EdgeTransport.LOCAL
        and stage_plan.intra_stage_edge.local_role == LocalRole.SENDER
    ):
        return stage_plan.intra_stage_edge
    return None
