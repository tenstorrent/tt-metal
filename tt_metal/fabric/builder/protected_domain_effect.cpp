// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/protected_domain_effect.hpp"

#include <enchantum/enchantum.hpp>
#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>

namespace tt::tt_fabric {

bool is_protected_y_egress(RoutingDirection egress, EdgeCapability egress_capability, ExpressAxis express_axis) {
    return (egress_capability == EdgeCapability::INTRAMESH_CARDINAL &&
            (egress == RoutingDirection::N || egress == RoutingDirection::S)) ||
           (egress_capability == EdgeCapability::INTRAMESH_EXPRESS && egress == RoutingDirection::Z &&
            is_y_axis_direction(egress, express_axis));
}

bool is_static_dor_forbidden(
    RoutingDirection ingress,
    EdgeCapability ingress_capability,
    RoutingDirection egress,
    EdgeCapability egress_capability,
    ExpressAxis express_axis) {
    // Only a same-mesh X edge puts a packet in its X phase: E/W, or the chord when it runs along X. An
    // INTERMESH port is a landing root even when its local compass letter is E or W, so it is not an X
    // ingress.
    const bool is_intramesh_x_ingress =
        ingress_capability != EdgeCapability::INTERMESH && is_x_axis_direction(ingress, express_axis);

    return is_intramesh_x_ingress && is_protected_y_egress(egress, egress_capability, express_axis);
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
    queries.express_axis = express_axis_of(control_plane, local.mesh_id);
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
    const ExpressAxis express_axis = queries.express_axis;
    TT_FATAL(
        express_axis != ExpressAxis::NONE,
        "Producer effects are only derived on express meshes, but these ring queries carry no express axis");
    TT_FATAL(
        !is_static_dor_forbidden(ingress, ingress_capability, egress, egress_capability, express_axis),
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

    if (is_y_axis_direction(ingress, express_axis) != is_y_axis_direction(egress, express_axis)) {
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
