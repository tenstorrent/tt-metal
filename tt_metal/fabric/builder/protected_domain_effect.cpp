// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/protected_domain_effect.hpp"

#include <enchantum/enchantum.hpp>
#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>

namespace tt::tt_fabric {

bool is_protected_y_egress(RoutingDirection egress, EdgeCapability egress_capability) {
    return (egress_capability == EdgeCapability::INTRAMESH_CARDINAL &&
            (egress == RoutingDirection::N || egress == RoutingDirection::S)) ||
           (egress_capability == EdgeCapability::INTRAMESH_EXPRESS && egress == RoutingDirection::Z);
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

    return is_intramesh_x_ingress && is_protected_y_egress(egress, egress_capability);
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

}  // namespace tt::tt_fabric
