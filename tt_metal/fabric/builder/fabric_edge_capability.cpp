// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/fabric_edge_capability.hpp"

#include <enchantum/enchantum.hpp>
#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>

namespace tt::tt_fabric {

ZPortRole z_port_role(const ControlPlane& control_plane, FabricNodeId node) {
    // get_chip_neighbors covers both intra- and inter-mesh adjacency, so one query answers both.
    const auto neighbors = control_plane.get_chip_neighbors(node, RoutingDirection::Z);
    for (const auto& [neighbor_mesh, chips] : neighbors) {
        if (chips.empty()) {
            continue;
        }
        return neighbor_mesh != node.mesh_id ? ZPortRole::INTERMESH_BOUNDARY : ZPortRole::EXPRESS_CHORD;
    }
    return ZPortRole::NONE;
}

ZPortRole z_role_of(const PerDirectionCapabilities& caps) {
    const auto& z = caps[static_cast<size_t>(RoutingDirection::Z)];
    if (!z.has_value()) {
        return ZPortRole::NONE;
    }
    return *z == EdgeCapability::INTERMESH ? ZPortRole::INTERMESH_BOUNDARY : ZPortRole::EXPRESS_CHORD;
}

void validate_facing_role_consistency(RoutingDirection facing, EdgeCapability edge_capability, ZPortRole chip_z_role) {
    if (facing != RoutingDirection::Z) {
        // The chord lives on the chip's Z port; a cardinal-facing router can never carry it.
        TT_FATAL(
            edge_capability != EdgeCapability::INTRAMESH_EXPRESS,
            "Router facing {} carries INTRAMESH_EXPRESS, but an express chord lives on the chip's "
            "Z port",
            enchantum::to_string(facing));
        return;
    }
    switch (edge_capability) {
        case EdgeCapability::INTERMESH:
            TT_FATAL(
                chip_z_role == ZPortRole::INTERMESH_BOUNDARY,
                "A Z-facing intermesh edge means the chip's Z port is the boundary: role must be "
                "INTERMESH_BOUNDARY, got {}",
                enchantum::to_string(chip_z_role));
            break;
        case EdgeCapability::INTRAMESH_EXPRESS:
            TT_FATAL(
                chip_z_role == ZPortRole::EXPRESS_CHORD,
                "A same-mesh Z edge is this chip's express chord: role must be EXPRESS_CHORD, got {}",
                enchantum::to_string(chip_z_role));
            break;
        case EdgeCapability::INTRAMESH_CARDINAL:
            TT_FATAL(
                false,
                "A same-mesh Z edge is an express chord and must carry INTRAMESH_EXPRESS capability; "
                "an ordinary cardinal-capability Z edge cannot exist");
            break;
    }
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
        enchantum::to_string(ingress),
        enchantum::to_string(egress));

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
