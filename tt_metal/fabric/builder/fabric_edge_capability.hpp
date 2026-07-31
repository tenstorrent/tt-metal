// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
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

const char* to_string(EdgeCapability capability);

// What the extra (non-grid) ethernet port on this chip is used for.
//
// Historically this port had exactly one role, so "has a Z port" and "has an intermesh boundary"
// were the same statement. They no longer are: an express chord gives the port a genuine intramesh
// routing direction. Naming the role rather than the port keeps that distinction structural: a
// chip has one extra port, so it can hold exactly one role -- the mutual exclusion between
// "intermesh Z" and "express chord" is unrepresentable rather than asserted.
enum class ZPortRole : uint8_t {
    NONE,                // no extra port on this chip
    INTERMESH_BOUNDARY,  // crosses a mesh boundary; carries no intramesh routing direction
    EXPRESS_CHORD,       // a same-mesh Y-axis express chord; an ordinary routing direction
};

const char* to_string(ZPortRole role);

// The role of this chip's extra port, from the neighbor graph. Pure structure: a same-mesh Z edge
// reports EXPRESS_CHORD, and its validity (validated express intent) is enforced separately at
// capability classification, which fails the configuration for a same-mesh Z without it.
ZPortRole z_port_role(const ControlPlane& control_plane, FabricNodeId node);

// Does this router's own port carry an intramesh routing direction, i.e. participate in ordinary
// dimension-ordered forwarding?
//
// Cardinal ports always do; that is what makes them cardinal. The extra port does exactly when it
// carries an express chord (a Y-axis resource, like N/S). An extra port crossing a mesh boundary
// does not -- it has no place in the turn matrix, and that, not the letter Z, is the whole reason
// a separate boundary template ever existed.
bool carries_routing_direction(RoutingDirection facing, EdgeCapability capability);

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

// Capability of the edge leaving `local` through `direction`, or nullopt when no neighbor exists
// there, which means the corresponding producer slot is not wired.
std::optional<EdgeCapability> capability_in_direction(
    const ControlPlane& control_plane, FabricNodeId local, RoutingDirection direction);

// What one wired producer does to protected-ring occupancy (builder contract section 4.4).
//
// This is the fact that selects a sender's flow-control guard, and it is not derivable from the
// command letter: the same express output is transit when fed by the ring and an acquisition when fed
// by a leaf attachment.
enum class ProtectedDomainEffect : uint8_t {
    NON_RING,       // the egress is not a cyclic resource, so no bubble applies
    REMAIN,         // same directed ring: transit, needs only the weaker guard
    ENTER,          // acquires a protected ring, needs the stronger guard
    NON_CANONICAL,  // wired, but no canonical route uses this turn
};

const char* to_string(ProtectedDomainEffect effect);

// Only an acquisition is an injection channel.
bool is_injection_effect(ProtectedDomainEffect effect);

// The protected-ring facts the derivation below needs, already bound to one local node.
//
// Bound rather than queried directly so the ladder stays free of ControlPlane, which lets regression
// drive it from a real derived ring model without a device.
struct ProtectedRingQueries {
    std::function<bool(RoutingDirection egress)> is_protected_ring_edge;
    std::function<bool(RoutingDirection ingress, RoutingDirection egress)> are_same_directed_ring_edges;
    std::function<bool(RoutingDirection ingress, RoutingDirection egress)> continuation_allowed;
};

ProtectedRingQueries make_protected_ring_queries(const ControlPlane& control_plane, FabricNodeId local);

// Total effect of the producer path U->V->W, where V is the node the queries are bound to.
//
// Fails the configuration on a dimension-order-forbidden turn: such a producer should have been
// unwired during connection mapping, so reaching here means the maps and this derivation disagree.
ProtectedDomainEffect classify_producer_effect(
    const ProtectedRingQueries& queries,
    RoutingDirection ingress,
    EdgeCapability ingress_capability,
    RoutingDirection egress,
    EdgeCapability egress_capability);

// Worker source injection has no ingress direction, so it is a first acquisition whenever its egress
// is protected. It deliberately does not consult the turn predicates.
ProtectedDomainEffect classify_worker_effect(const ProtectedRingQueries& queries, RoutingDirection egress);

}  // namespace tt::tt_fabric
