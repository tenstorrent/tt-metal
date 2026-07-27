// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/fabric_edge_capability.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>

namespace tt::tt_fabric {

namespace {

// Walk this node's Z-direction neighbours, reporting whether any of them is in another mesh.
// get_chip_neighbors covers both intra- and inter-mesh adjacency, so one query answers both.
bool has_z_edge_crossing_mesh(const ControlPlane& control_plane, FabricNodeId local, bool want_crossing) {
    const auto neighbors = control_plane.get_chip_neighbors(local, RoutingDirection::Z);
    for (const auto& [neighbor_mesh, chips] : neighbors) {
        if (chips.empty()) {
            continue;
        }
        const bool crossing = neighbor_mesh != local.mesh_id;
        if (crossing == want_crossing) {
            return true;
        }
    }
    return false;
}

}  // namespace

const char* to_string(EdgeCapability capability) {
    switch (capability) {
        case EdgeCapability::INTRAMESH_CARDINAL: return "INTRAMESH_CARDINAL";
        case EdgeCapability::INTRAMESH_EXPRESS: return "INTRAMESH_EXPRESS";
        case EdgeCapability::INTERMESH: return "INTERMESH";
    }
    return "UNKNOWN";
}

EdgeCapability classify_fabric_edge(
    const ControlPlane& control_plane, FabricNodeId local, FabricNodeId remote, RoutingDirection direction) {
    // Only consult express enablement for a same-mesh Z, so classifying an ordinary edge never forces
    // the ring model to be derived.
    const bool express_enabled = (remote.mesh_id == local.mesh_id && direction == RoutingDirection::Z) &&
                                 control_plane.express_routing_enabled(local.mesh_id);
    return classify_fabric_edge(local, remote, direction, express_enabled);
}

EdgeCapability classify_fabric_edge(
    FabricNodeId local, FabricNodeId remote, RoutingDirection direction, bool express_routing_enabled) {
    if (remote.mesh_id != local.mesh_id) {
        // Any direction can carry a mesh boundary, including Z.
        return EdgeCapability::INTERMESH;
    }

    if (direction != RoutingDirection::Z) {
        return EdgeCapability::INTRAMESH_CARDINAL;
    }

    TT_FATAL(
        express_routing_enabled,
        "Chip M{}D{} has a same-mesh Z edge to D{}, but express routing is not enabled for mesh {}. A same-mesh Z "
        "adjacency is an express chord, so this means the descriptor's express intent was not materialized and "
        "validated; treating it as cardinal or intermesh would hide that mismatch.",
        *local.mesh_id,
        local.chip_id,
        remote.chip_id,
        *local.mesh_id);

    return EdgeCapability::INTRAMESH_EXPRESS;
}

bool has_intermesh_z_edge(const ControlPlane& control_plane, FabricNodeId local) {
    return has_z_edge_crossing_mesh(control_plane, local, /*want_crossing=*/true);
}

bool has_intramesh_express_edge(const ControlPlane& control_plane, FabricNodeId local) {
    return has_z_edge_crossing_mesh(control_plane, local, /*want_crossing=*/false);
}

bool is_y_axis_direction(RoutingDirection direction) {
    return direction == RoutingDirection::N || direction == RoutingDirection::S || direction == RoutingDirection::Z;
}

bool is_x_axis_direction(RoutingDirection direction) {
    return direction == RoutingDirection::E || direction == RoutingDirection::W;
}

std::optional<EdgeCapability> capability_in_direction(
    const ControlPlane& control_plane, FabricNodeId local, RoutingDirection direction) {
    const auto neighbors = control_plane.get_chip_neighbors(local, direction);
    for (const auto& [neighbor_mesh, chips] : neighbors) {
        if (chips.empty()) {
            continue;
        }
        // discover_channels() rejects more than one neighbor mesh per direction, so the first entry is
        // the only one.
        return classify_fabric_edge(control_plane, local, FabricNodeId(neighbor_mesh, chips.front()), direction);
    }
    return std::nullopt;
}

const char* to_string(ProtectedDomainEffect effect) {
    switch (effect) {
        case ProtectedDomainEffect::NON_RING: return "NON_RING";
        case ProtectedDomainEffect::REMAIN: return "REMAIN";
        case ProtectedDomainEffect::ENTER: return "ENTER";
        case ProtectedDomainEffect::NON_CANONICAL: return "NON_CANONICAL";
    }
    return "UNKNOWN";
}

bool is_injection_effect(ProtectedDomainEffect effect) { return effect == ProtectedDomainEffect::ENTER; }

ProtectedRingQueries make_protected_ring_queries(const ControlPlane& control_plane, FabricNodeId local) {
    ProtectedRingQueries queries;
    queries.is_protected_ring_edge = [&control_plane, local](RoutingDirection egress) {
        return control_plane.is_protected_ring_edge(local, egress);
    };
    queries.are_same_directed_ring_edges = [&control_plane, local](RoutingDirection ingress, RoutingDirection egress) {
        return control_plane.are_same_directed_ring_edges(local, ingress, egress);
    };
    queries.continuation_allowed = [&control_plane, local](RoutingDirection ingress, RoutingDirection egress) {
        return control_plane.continuation_allowed(local, ingress, egress);
    };
    return queries;
}

ProtectedDomainEffect classify_worker_effect(const ProtectedRingQueries& queries, RoutingDirection egress) {
    return queries.is_protected_ring_edge(egress) ? ProtectedDomainEffect::ENTER : ProtectedDomainEffect::NON_RING;
}

ProtectedDomainEffect classify_producer_effect(
    const ProtectedRingQueries& queries,
    RoutingDirection ingress,
    EdgeCapability ingress_capability,
    RoutingDirection egress,
    EdgeCapability egress_capability) {
    TT_FATAL(
        !is_static_dor_forbidden(ingress, ingress_capability, egress, egress_capability),
        "Producer {} -> {} violates dimension order but is still wired. Connection mapping should have unwired it, so "
        "the maps and this derivation disagree.",
        static_cast<int>(ingress),
        static_cast<int>(egress));

    if (!queries.is_protected_ring_edge(egress)) {
        return ProtectedDomainEffect::NON_RING;
    }

    if (ingress_capability == EdgeCapability::INTERMESH) {
        // A landed carrier holds no position on this mesh's rings, so its first protected egress is an
        // acquisition. The landing map rebuild itself does not acquire anything.
        return ProtectedDomainEffect::ENTER;
    }

    if (queries.are_same_directed_ring_edges(ingress, egress)) {
        return ProtectedDomainEffect::REMAIN;
    }

    if (is_y_axis_direction(ingress) != is_y_axis_direction(egress)) {
        // A dimension change. Dimension order leaves Y->X as the only legal case here, and the first
        // X hop acquires the X ring.
        return ProtectedDomainEffect::ENTER;
    }

    if (queries.continuation_allowed(ingress, egress)) {
        return ProtectedDomainEffect::ENTER;
    }

    return ProtectedDomainEffect::NON_CANONICAL;
}

bool is_static_dor_forbidden(
    RoutingDirection ingress,
    EdgeCapability ingress_capability,
    RoutingDirection egress,
    EdgeCapability egress_capability) {
    // Only an ordinary same-mesh X edge puts a packet in its X phase. An INTERMESH port is a landing
    // root even when its local compass letter is E or W, so it is not an X ingress.
    const bool is_intramesh_x_ingress =
        ingress_capability == EdgeCapability::INTRAMESH_CARDINAL && is_x_axis_direction(ingress);

    const bool is_intramesh_y_egress =
        (egress_capability == EdgeCapability::INTRAMESH_CARDINAL && (egress == RoutingDirection::N ||
                                                                    egress == RoutingDirection::S)) ||
        (egress_capability == EdgeCapability::INTRAMESH_EXPRESS && egress == RoutingDirection::Z);

    return is_intramesh_x_ingress && is_intramesh_y_egress;
}

}  // namespace tt::tt_fabric
