// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>

namespace tt::tt_fabric {

class ControlPlane;

// What transport behaviour an edge carries, independent of which direction letter selects it.
//
// This separation is the point: a direction picks an output, capability picks transport behaviour.
// "Direction is Z" does not mean intermesh, and an intermesh edge can sit on any compass letter.
enum class EdgeCapability : uint8_t {
    INTRAMESH_CARDINAL,  // ordinary same-mesh N/S/E/W edge
    INTRAMESH_EXPRESS,   // same-mesh express chord, command Z
    INTERMESH,           // crosses a mesh boundary, on any direction
};

// What this chip's Z port is used for.
//
// A chip has one Z port, so one enum makes express-chord and intermesh-boundary uses mutually
// exclusive. validate_facing_role_consistency() checks that a Z-facing router agrees with this role.
enum class ZPortRole : uint8_t {
    NONE,                // no Z port on this chip
    INTERMESH_BOUNDARY,  // crosses a mesh boundary; carries no intramesh routing direction
    EXPRESS_CHORD,       // a same-mesh Y-axis express chord; an ordinary routing direction
};

// Per-direction capabilities for one chip; nullopt means the direction is absent. Indexed in
// RoutingDirection order (N,E,S,W,Z), not eth_chan_directions order (E,W,N,S,Z). C and NONE are not ports.
class PerDirectionCapabilities {
public:
    std::optional<EdgeCapability>& at(RoutingDirection direction);
    const std::optional<EdgeCapability>& at(RoutingDirection direction) const;

private:
    std::array<std::optional<EdgeCapability>, 5> by_direction_{};
};

// Derive the Z-port role from one chip's capabilities.
ZPortRole z_role_of(const PerDirectionCapabilities& caps);

// Require the router facing, edge capability, and chip Z role to describe the same physical port.
// Express capability is valid only on Z; Z intermesh and same-mesh edges require their matching roles.
void validate_facing_role_consistency(RoutingDirection facing, EdgeCapability edge_capability, ZPortRole chip_z_role);

// Read this chip's Z-port role from the neighbor graph. Capability classification separately
// validates express intent for a same-mesh Z edge.
ZPortRole z_port_role(const ControlPlane& control_plane, FabricNodeId node);

// True only for the Z-facing intermesh port, which carries no intramesh routing direction and
// therefore uses the boundary template instead of the direction-keyed turn matrix.
inline bool is_z_boundary_router(RoutingDirection facing, EdgeCapability capability) {
    return facing == RoutingDirection::Z && capability == EdgeCapability::INTERMESH;
}

// Classify one edge from `local` toward `remote` leaving through `direction`.
//
// Fails the configuration for a same-mesh Z edge on a mesh where express routing was not
// materialized and validated: that combination means topology intent and the neighbor graph
// disagree, and silently treating it as either cardinal or intermesh would hide the mismatch.
EdgeCapability classify_fabric_edge(
    FabricNodeId local, FabricNodeId remote, RoutingDirection direction, bool express_routing_enabled);

// Same classification, resolving express enablement from the ControlPlane.
EdgeCapability classify_fabric_edge(
    const ControlPlane& control_plane, FabricNodeId local, FabricNodeId remote, RoutingDirection direction);

// Which axis a direction belongs to: N/S/Z are Y, E/W are X.
bool is_y_axis_direction(RoutingDirection direction);
bool is_x_axis_direction(RoutingDirection direction);

// Capability of the edge leaving `local` through `direction`, or nullopt when no neighbor exists
// there, which means the corresponding producer slot is not wired.
std::optional<EdgeCapability> capability_in_direction(
    const ControlPlane& control_plane, FabricNodeId local, RoutingDirection direction);

// Classify every direction of one chip for callers deriving a peer router's archetype.
PerDirectionCapabilities chip_capabilities_of(const ControlPlane& control_plane, FabricNodeId local);

}  // namespace tt::tt_fabric
