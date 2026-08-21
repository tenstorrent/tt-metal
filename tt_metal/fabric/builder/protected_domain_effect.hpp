// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "tt_metal/fabric/builder/fabric_edge_capability.hpp"

namespace tt::tt_fabric {

class ControlPlane;

// Is this egress one of the Y resources dimension order protects?
//
// The Y phase of a route runs on same-mesh N/S edges and on the express chord, so those are the
// egresses an X-phase packet may not turn back into. An INTERMESH egress is the packet leaving the
// mesh entirely rather than re-entering a Y ring, so it is not one of them whatever compass letter
// discovery put the seam on -- which is what keeps an exit chip's E/W routers wired to the
// boundary (builder contract section 4.4).
bool is_protected_y_egress(RoutingDirection egress, EdgeCapability egress_capability);

// Would forwarding from `ingress` to `egress` violate the fixed Y-before-X dimension order?
//
// True only for an ordinary same-mesh X ingress turning back into a protected Y egress. Dimension
// order is what keeps X resources from ever waiting on Y ones, which the deadlock-freedom argument
// relies on, so such a producer is never wired.
//
// An INTERMESH ingress is deliberately exempt even on an E or W port: a boundary landing is a route
// root rather than a packet already in its X phase, so it may legally begin Y.
bool is_static_dor_forbidden(
    RoutingDirection ingress,
    EdgeCapability ingress_capability,
    RoutingDirection egress,
    EdgeCapability egress_capability);

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

// The per-chip facts a router build needs, bound once at chip scope: the edge capabilities
// classified at discovery, and the ring predicates bound at FabricBuilder construction (a live
// handle onto ControlPlane's node-scoped predicates, not data). Threaded through create -> build
// alongside the per-router RouterLocation. Membership rule: only per-node, bound-once facts --
// nothing mesh-wide, nothing derivable from what's already here.
struct ChipRoutingFacts {
    PerDirectionCapabilities per_direction_capabilities;
    ProtectedRingQueries protected_ring_queries;
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
