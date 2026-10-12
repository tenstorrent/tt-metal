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
// Same-mesh N/S and express-chord edges are protected Y resources. An intermesh egress leaves the
// mesh, so it is excluded regardless of direction; this keeps E/W exit routers wired to the boundary.
bool is_protected_y_egress(RoutingDirection egress, EdgeCapability egress_capability);

// Would forwarding from `ingress` to `egress` violate the fixed Y-before-X dimension order?
//
// Forbids a same-mesh X ingress from turning back into protected Y. An intermesh ingress is exempt,
// even on E/W, because a boundary landing is a new route root and may begin Y.
bool is_static_dor_forbidden(
    RoutingDirection ingress,
    EdgeCapability ingress_capability,
    RoutingDirection egress,
    EdgeCapability egress_capability);

// What one wired producer does to protected-ring occupancy.
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

// Protected-ring queries bound to one local node. Keeping the derivation independent of ControlPlane
// lets machine-free tests supply the same facts from a derived topology.
struct ProtectedRingQueries {
    std::function<bool(RoutingDirection egress)> is_protected_ring_edge;
    std::function<bool(RoutingDirection ingress, RoutingDirection egress)> are_same_directed_ring_edges;
    std::function<bool(RoutingDirection ingress, RoutingDirection egress)> continuation_allowed;
};

// Per-chip facts bound once and shared by every router on that chip: discovered edge capabilities
// and live handles to ControlPlane's node-scoped ring predicates.
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
