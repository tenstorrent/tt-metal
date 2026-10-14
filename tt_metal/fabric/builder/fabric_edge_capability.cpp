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

namespace {

size_t checked_direction_index(RoutingDirection direction) {
    TT_FATAL(
        direction <= RoutingDirection::Z,
        "RoutingDirection {} is not a port and cannot index PerDirectionCapabilities",
        enchantum::to_string(direction));
    return static_cast<size_t>(direction);
}

}  // namespace

std::optional<EdgeCapability>& PerDirectionCapabilities::at(RoutingDirection direction) {
    return by_direction_[checked_direction_index(direction)];
}

const std::optional<EdgeCapability>& PerDirectionCapabilities::at(RoutingDirection direction) const {
    return by_direction_[checked_direction_index(direction)];
}

ZPortRole z_role_of(const PerDirectionCapabilities& caps) {
    const auto& z = caps.at(RoutingDirection::Z);
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
    // express_routing_enabled is a pointer read over ring state the RoutingTableGenerator derives at
    // construction, so it is cheap to consult unconditionally; the flag is inert outside the
    // same-mesh-Z branch (tests drive the four-argument overload directly).
    return classify_fabric_edge(local, remote, direction, control_plane.express_routing_enabled(local.mesh_id));
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
        "Chip M{}D{} has a same-mesh Z edge to D{} with express routing disabled for mesh {}. The express_links "
        "expansion in mesh_graph.cpp is the sole writer of intramesh Z edges and express_routing_enabled is its "
        "validated echo, so this state is unreachable unless that single-writer invariant broke; there is no valid "
        "non-express reading of a same-mesh Z edge to fall back to.",
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

PerDirectionCapabilities chip_capabilities_of(const ControlPlane& control_plane, FabricNodeId local) {
    PerDirectionCapabilities caps;
    for (const auto direction :
         {RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W, RoutingDirection::Z}) {
        caps.at(direction) = capability_in_direction(control_plane, local, direction);
    }
    return caps;
}

}  // namespace tt::tt_fabric
