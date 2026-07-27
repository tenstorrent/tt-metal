// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>

namespace tt::tt_fabric {

class ControlPlane;

// What transport behaviour an edge carries, independent of which direction letter selects it.
//
// This separation is the point: a direction picks an output, capability picks transport behaviour.
// "Direction is Z" does not mean intermesh, and an intermesh edge can sit on any compass letter.
// See GALAXY_BUILDER_ROUTING_CONFIG_CONTRACT.md section 4.3.
enum class EdgeCapability : uint8_t {
    INTRAMESH_CARDINAL,  // ordinary same-mesh N/S/E/W edge
    INTRAMESH_EXPRESS,   // same-mesh express chord, command Z
    INTERMESH,           // crosses a mesh boundary, on any direction
};

const char* to_string(EdgeCapability capability);

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

// Does this node terminate a Z edge that crosses a mesh boundary?
//
// This is the precise gate for the intermesh Z router shape (its VC1 sender fan and the MESH_TO_Z
// connection). It is deliberately narrower than "any active Z channel", which is also true for a
// same-mesh express chord and would wrongly hand that chord the intermesh template.
bool has_intermesh_z_edge(const ControlPlane& control_plane, FabricNodeId local);

// Does this node terminate a same-mesh express chord?
bool has_intramesh_express_edge(const ControlPlane& control_plane, FabricNodeId local);

// Which axis a direction belongs to: N/S/Z are Y, E/W are X (builder contract section 4.2.1).
bool is_y_axis_direction(RoutingDirection direction);
bool is_x_axis_direction(RoutingDirection direction);

// Would forwarding from `ingress` to `egress` violate the fixed Y-before-X dimension order?
//
// True only for an ordinary same-mesh X ingress turning back into an intramesh Y egress. Dimension
// order is what keeps X resources from ever waiting on Y ones, which the deadlock-freedom argument
// relies on, so such a producer is never wired.
//
// An INTERMESH ingress is deliberately exempt even on an E or W port: a boundary landing is a route
// root rather than a packet already in its X phase, so it may legally begin Y.
bool is_static_dor_forbidden(
    RoutingDirection ingress, EdgeCapability ingress_capability, RoutingDirection egress,
    EdgeCapability egress_capability);

}  // namespace tt::tt_fabric
