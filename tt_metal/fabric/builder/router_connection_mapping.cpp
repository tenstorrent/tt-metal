// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/router_connection_mapping.hpp"
#include "tt_metal/fabric/builder/fabric_builder_config.hpp"

#include <algorithm>
#include <array>

namespace tt::tt_fabric {

namespace {

constexpr std::array<RoutingDirection, 4> k_cardinal_directions = {
    RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W};

}  // namespace

std::vector<ConnectionTarget> RouterConnectionMapping::get_downstream_targets(uint32_t vc) const {
    if (vc >= targets_by_vc_.size()) {
        return {};
    }
    return targets_by_vc_[vc];
}

bool RouterConnectionMapping::has_targets(uint32_t vc) const {
    return vc < targets_by_vc_.size() && !targets_by_vc_[vc].empty();
}

size_t RouterConnectionMapping::get_total_target_count() const {
    size_t total = 0;
    for (const auto& targets : targets_by_vc_) {
        total += targets.size();
    }
    return total;
}

void RouterConnectionMapping::add_target(uint32_t vc, const ConnectionTarget& target) {
    targets_by_vc_[vc].push_back(target);
}

RouterConnectionMapping RouterConnectionMapping::for_router(
    Topology topology,
    RoutingDirection facing,
    EdgeCapability edge_capability,
    ZPortRole z_role,
    bool express_routing_enabled,
    bool enable_vc1,
    bool enable_mesh_pass_through) {
    RouterConnectionMapping mapping;

    validate_facing_role_consistency(facing, edge_capability, z_role);

    // A port with no routing direction gets the boundary template: the full non-self set on VC1,
    // typed from-boundary. Its VC0 senders are fed by the mesh routers' boundary targets on their
    // own maps, so there is no VC0 arm here -- traffic arriving on its VC0 receiver crosses over
    // onto these same VC1 downstream senders instead. Nothing in the turn matrix applies to it,
    // which is why the set is the full set: not a special case, a consequence. The requirements
    // match the shape derivation (router_vc_shape): the boundary is 2D-only and exists only with
    // VC1, since its entire shape is the from-boundary VC1 fanout.
    if (!carries_routing_direction(facing, edge_capability)) {
        TT_FATAL(is_2D_topology(topology), "A Z-facing intermesh boundary router requires a 2D topology");
        TT_FATAL(
            enable_vc1,
            "A Z-facing intermesh boundary router cannot be constructed without VC1: its entire "
            "shape is the from-boundary VC1 fanout");
        for (size_t i = 0; i < k_cardinal_directions.size(); ++i) {
            mapping.add_target(1, ConnectionTarget(1, k_cardinal_directions[i]));
        }
        return mapping;
    }

    // A Z-facing router reaching this point is the express chord: the cross-check above has
    // already rejected every other capability, and its role has been verified EXPRESS_CHORD.
    if (facing == RoutingDirection::Z) {
        TT_FATAL(
            express_routing_enabled && is_2D_topology(topology),
            "An express (Z) chord requires 2D Mesh/Torus routing with express routing enabled");
    }

    // 1D: opposite only. There is no 1D boundary target: intermesh connections are rejected
    // upstream for 1D ("1D routing does not support intermesh connections"), so a 1D router with
    // role INTERMESH_BOUNDARY cannot occur in a valid configuration -- and get_router_connection_pairs
    // emits no Z pairs in 1D, so such a target would be unestablishable anyway.
    if (topology == Topology::Linear || topology == Topology::Ring) {
        const auto opposite = get_opposite_direction(facing);
        mapping.add_target(0, ConnectionTarget(0, opposite));
        return mapping;
    }

    // 2D: the turn set from the wiring primitive -- opposite first, then the remaining cardinals
    // in enum order, then the extra port. Every member is what the primitive wires, so this set
    // and the guard derivation cannot disagree. A Z-facing router has no opposite: its set is all
    // four cardinals (an express chord is a Y resource, like N/S).
    // The vc argument below is irrelevant to turn-set construction: the only vc-sensitive arm of
    // the primitive (a boundary producer) is unreachable here -- the boundary path early-returns
    // before this point, and a Z-facing producer reaching this point is an express chord, whose
    // feed rides every carrier VC.
    std::vector<RoutingDirection> outbound;
    const auto opposite = facing == RoutingDirection::Z ? facing : get_opposite_direction(facing);
    if (facing != RoutingDirection::Z &&
        wires_into(facing, edge_capability, opposite, z_role, express_routing_enabled, 0)) {
        outbound.push_back(opposite);
    }
    for (const auto candidate : k_cardinal_directions) {
        if (candidate == facing || candidate == opposite) {
            continue;
        }
        if (wires_into(facing, edge_capability, candidate, z_role, express_routing_enabled, 0)) {
            outbound.push_back(candidate);
        }
    }
    if (facing != RoutingDirection::Z &&
        wires_into(facing, edge_capability, RoutingDirection::Z, z_role, express_routing_enabled, 0)) {
        outbound.push_back(RoutingDirection::Z);
    }

    // Downstream capacity. The boundary template's target has its own accounting; the check is on
    // the ordinary intramesh members, matching each mode's historical convention.
    if (express_routing_enabled) {
        const size_t vc0_limit = builder_config::get_vc0_downstream_edm_count(/*is_2D_routing=*/true, /*express=*/true);
        TT_FATAL(
            outbound.size() <= vc0_limit,
            "Express VC0 outbound direction count ({}) exceeds the downstream EDM count ({})",
            outbound.size(),
            vc0_limit);
    } else {
        const size_t cardinals = static_cast<size_t>(std::count_if(
            outbound.begin(), outbound.end(), [](RoutingDirection d) { return d != RoutingDirection::Z; }));
        TT_FATAL(
            cardinals <= builder_config::num_downstream_edms_2d_vc0,
            "Outbound cardinal direction count ({}) exceeds the downstream EDM count ({})",
            cardinals,
            builder_config::num_downstream_edms_2d_vc0);
    }

    // Cardinal and express outputs are realized on every carrier VC, not just VC0: traffic that
    // has crossed a mesh boundary stays on VC1 through every later mesh, and a landed carrier can
    // still decode a Z action. target_vc always equals the source VC here: there is no VC1->VC0
    // landing crossover. The targets carry no channel index -- connection establishment computes
    // the slot from the direction bijection.
    for (size_t i = 0; i < outbound.size(); ++i) {
        const auto target = outbound[i];
        if (target == RoutingDirection::Z && z_role == ZPortRole::INTERMESH_BOUNDARY) {
            // The boundary template's target: VC0 always, VC1 only in pass-through mode.
            mapping.add_target(0, ConnectionTarget(0, target));
            if (enable_vc1 && enable_mesh_pass_through) {
                mapping.add_target(1, ConnectionTarget(1, target));
            }
            continue;
        }
        mapping.add_target(0, ConnectionTarget(0, target));
        if (enable_vc1) {
            mapping.add_target(1, ConnectionTarget(1, target));
        }
    }

    return mapping;
}

}  // namespace tt::tt_fabric
