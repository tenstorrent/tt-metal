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

}  // namespace tt::tt_fabric
