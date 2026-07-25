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

}  // namespace tt::tt_fabric
