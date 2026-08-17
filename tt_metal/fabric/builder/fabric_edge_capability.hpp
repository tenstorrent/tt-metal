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
// See GALAXY_BUILDER_ROUTING_CONFIG_CONTRACT.md section 4.3.
enum class EdgeCapability : uint8_t {
    INTRAMESH_CARDINAL,  // ordinary same-mesh N/S/E/W edge
    INTRAMESH_EXPRESS,   // same-mesh express chord, command Z
    INTERMESH,           // crosses a mesh boundary, on any direction
};

// What this chip's Z port is used for.
//
// Historically this port had exactly one role, so "has a Z port" and "has an intermesh boundary"
// were the same statement. They no longer are: an express chord gives the port a genuine intramesh
// routing direction. Naming the role rather than the port keeps that distinction structural: a
// chip has one Z port, so it can hold exactly one role -- the mutual exclusion between
// "intermesh Z" and "express chord" is unrepresentable on a chip. validate_facing_role_consistency
// (below) asserts the matching half of it: a Z-facing router's own capability must agree with the
// chip's role, and both mapping factories run it first.
enum class ZPortRole : uint8_t {
    NONE,                // no Z port on this chip
    INTERMESH_BOUNDARY,  // crosses a mesh boundary; carries no intramesh routing direction
    EXPRESS_CHORD,       // a same-mesh Y-axis express chord; an ordinary routing direction
};

// Per-direction capability set of one chip: each present direction's edge capability, nullopt
// where the direction is absent. Indexed by RoutingDirection (N=0, E=1, S=2, W=3, Z=4) -- NOT
// eth_chan_directions order (E,W,N,S,Z). C and NONE are not ports and must never index this: the
// array has five slots and the enum has seven values, so at() checks.
class PerDirectionCapabilities {
public:
    std::optional<EdgeCapability>& at(RoutingDirection direction);
    const std::optional<EdgeCapability>& at(RoutingDirection direction) const;

private:
    std::array<std::optional<EdgeCapability>, 5> by_direction_{};
};

// The Z port's role read off a per-direction capability set: absent means NONE, an intermesh Z is
// the boundary, anything else same-mesh is the chord. The pure spelling of the fact z_port_role
// queries the neighbor graph for.
ZPortRole z_role_of(const PerDirectionCapabilities& caps);

// The chip-level cross-check: a Z-facing router's own edge capability and the chip's Z-port
// role are two spellings of one fact and must agree -- a Z-facing intermesh edge means role
// INTERMESH_BOUNDARY, a same-mesh Z edge (an express chord) means role EXPRESS_CHORD, and express
// capability never sits on a cardinal facing. Anything else is an impossible chip, which the
// factories' independent parameters would otherwise make representable again.
void validate_facing_role_consistency(RoutingDirection facing, EdgeCapability edge_capability, ZPortRole chip_z_role);

// The role of this chip's Z port, from the neighbor graph. Pure structure: a same-mesh Z edge
// reports EXPRESS_CHORD, and its validity (validated express intent) is enforced separately at
// capability classification, which fails the configuration for a same-mesh Z without it.
ZPortRole z_port_role(const ControlPlane& control_plane, FabricNodeId node);

// Is this router the chip's Z-facing intermesh boundary? The one spelling of the fact, shared by
// the turn-set and shape derivations.
//
// Every port except this one carries an intramesh routing direction: cardinal ports always do
// (that is what makes them cardinal), and the Z port does exactly when it carries an express
// chord (a Y-axis resource, like N/S). A Z port crossing a mesh boundary carries none -- it
// has no place in the turn matrix, and that, not the letter Z, is the whole reason a separate
// boundary template ever existed.
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

// Which axis a direction belongs to: N/S/Z are Y, E/W are X (builder contract section 4.2.1).
bool is_y_axis_direction(RoutingDirection direction);
bool is_x_axis_direction(RoutingDirection direction);

// Capability of the edge leaving `local` through `direction`, or nullopt when no neighbor exists
// there, which means the corresponding producer slot is not wired.
std::optional<EdgeCapability> capability_in_direction(
    const ControlPlane& control_plane, FabricNodeId local, RoutingDirection direction);

// Every direction of one chip, classified in one pass. The live-query spelling of the set that
// discovery classifies once and threads through ChipRoutingFacts, for callers asking what a
// router on ANOTHER chip would look like.
PerDirectionCapabilities chip_capabilities_of(const ControlPlane& control_plane, FabricNodeId local);

}  // namespace tt::tt_fabric
