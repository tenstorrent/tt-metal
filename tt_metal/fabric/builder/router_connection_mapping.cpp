// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/router_connection_mapping.hpp"
#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"

#include <algorithm>
#include <array>

namespace tt::tt_fabric {

std::vector<ConnectionTarget> RouterConnectionMapping::get_downstream_targets(
    uint32_t vc, uint32_t receiver_channel) const {
    ReceiverChannelKey key{vc, receiver_channel};
    auto it = receiver_to_targets_.find(key);

    if (it != receiver_to_targets_.end()) {
        return it->second;
    }

    return {};  // No targets for this sender channel
}

bool RouterConnectionMapping::has_targets(uint32_t vc, uint32_t receiver) const {
    ReceiverChannelKey key{vc, receiver};
    return receiver_to_targets_.contains(key);
}

std::vector<ReceiverChannelKey> RouterConnectionMapping::get_all_receiver_keys() const {
    std::vector<ReceiverChannelKey> keys;
    keys.reserve(receiver_to_targets_.size());

    for (const auto& [key, _] : receiver_to_targets_) {
        keys.push_back(key);
    }

    return keys;
}

void RouterConnectionMapping::add_target(uint32_t vc, uint32_t receiver_channel, const ConnectionTarget& target) {
    ReceiverChannelKey key{vc, receiver_channel};
    receiver_to_targets_[key].push_back(target);
}

RoutingDirection RouterConnectionMapping::get_opposite_direction(RoutingDirection dir) {
    switch (dir) {
        case RoutingDirection::N: return RoutingDirection::S;
        case RoutingDirection::S: return RoutingDirection::N;
        case RoutingDirection::E: return RoutingDirection::W;
        case RoutingDirection::W: return RoutingDirection::E;
        default:
            TT_FATAL(false, "Invalid routing direction for opposite calculation: {}", static_cast<int>(dir));
            return dir;  // Unreachable
    }
}

namespace {

constexpr std::array<RoutingDirection, 4> k_cardinal_directions = {
    RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W};

constexpr std::array<RoutingDirection, 5> k_all_directions = {
    RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W, RoutingDirection::Z};

// The extra port's role from a per-direction capability set: absent means NONE, an intermesh Z is
// the boundary, anything else same-mesh is the chord.
ZPortRole z_role_of(const RouterConnectionMapping::PerDirectionCapabilities& caps) {
    const auto& z = caps[static_cast<size_t>(RoutingDirection::Z)];
    if (!z.has_value()) {
        return ZPortRole::NONE;
    }
    return *z == EdgeCapability::INTERMESH ? ZPortRole::INTERMESH_BOUNDARY : ZPortRole::EXPRESS_CHORD;
}

}  // namespace

RouterConnectionMapping::PerDirectionCapabilities RouterConnectionMapping::canonical_express_endpoint_capabilities() {
    PerDirectionCapabilities caps;
    for (size_t i = 0; i < caps.size() - 1; ++i) {
        caps[i] = EdgeCapability::INTRAMESH_CARDINAL;
    }
    caps[static_cast<size_t>(RoutingDirection::Z)] = EdgeCapability::INTRAMESH_EXPRESS;
    return caps;
}

bool RouterConnectionMapping::wires_into(
    RoutingDirection producer_direction,
    EdgeCapability producer_capability,
    RoutingDirection egress_direction,
    ZPortRole z_role,
    bool express_routing_enabled,
    uint32_t vc) {
    if (producer_direction == egress_direction) {
        return false;  // a router never wires back over its own link
    }

    if (egress_direction == RoutingDirection::Z) {
        if (!express_routing_enabled) {
            // Legacy: the extra port exists in the set only as the boundary template (MESH_TO_Z).
            return z_role == ZPortRole::INTERMESH_BOUNDARY;
        }
        // Express: Z is a legal target for every producer except an intramesh X one (dimension
        // order forbids X -> Y), and it is realized only when the chip has the port at all.
        const bool legal = producer_direction == RoutingDirection::Z || !is_x_axis_direction(producer_direction) ||
                           producer_capability == EdgeCapability::INTERMESH;
        return legal && z_role != ZPortRole::NONE;
    }

    if (!express_routing_enabled) {
        // Legacy: every non-self cardinal direction wires in, on every carrier VC.
        return true;
    }

    if (producer_direction == RoutingDirection::Z) {
        // A Z-facing producer fans out to every non-self direction -- but a boundary producer's
        // feed is VC-shaped: its VC1 receiver fans out onto VC1 senders, while its VC0 receiver
        // crosses over onto downstream VC1 senders and feeds nothing on VC0. An express chord's
        // feed rides every carrier VC.
        if (producer_capability == EdgeCapability::INTERMESH) {
            return vc == 1;
        }
        return true;
    }
    if (is_x_axis_direction(producer_direction) && producer_capability != EdgeCapability::INTERMESH) {
        // Dimension order: an ordinary X producer may only continue around the X ring. A landing
        // (INTERMESH) X producer is exempt -- it is a route root, not a packet mid-X-phase.
        return egress_direction == get_opposite_direction(producer_direction);
    }
    return true;
}

uint32_t RouterConnectionMapping::express_vc0_producer_arity(
    RoutingDirection direction, const PerDirectionCapabilities& caps) {
    // The extra port's role the wiring rule consults is this chip's own, not a global.
    const ZPortRole z_role = z_role_of(caps);

    uint32_t count = 1;  // sender channel 0 is the local worker
    for (const auto producer : k_all_directions) {
        if (producer == direction) {
            continue;
        }
        const auto& capability = caps[static_cast<size_t>(producer)];
        if (!capability.has_value()) {
            continue;  // direction absent on this chip: no producer to wire
        }
        if (wires_into(producer, *capability, direction, z_role, /*express_routing_enabled=*/true, /*vc=*/0)) {
            ++count;
        }
    }
    return count;
}

uint32_t RouterConnectionMapping::express_mesh_vc0_sender_count() {
    // The canonical endpoint chip attains the structural ceiling: every Y and X producer wires
    // into an E/W facing under any capability assignment, so no per-chip set produces a wider one.
    const auto caps = canonical_express_endpoint_capabilities();
    uint32_t count = 0;
    for (const auto direction : k_all_directions) {
        count = std::max(count, express_vc0_producer_arity(direction, caps));
    }
    return count;
}

uint32_t RouterConnectionMapping::express_mesh_vc1_sender_count() {
    // VC1 forwards the same wired producer set as VC0 on every facing, minus the worker slot, so
    // vc1_arity(f) == vc0_arity(f) - 1 for every facing f. The family max therefore commutes with
    // the subtraction: max_f(vc0_arity(f)) - 1 == max_f(vc0_arity(f) - 1).
    return express_mesh_vc0_sender_count() - 1;
}

uint32_t RouterConnectionMapping::mesh_router_vc0_sender_count(ZPortRole /*z_role*/, bool express_routing_enabled) {
    if (express_routing_enabled) {
        return express_mesh_vc0_sender_count();  // family max over facing of the express rule
    }
    // Legacy rule: the worker plus every non-self cardinal producer. An intermesh Z edge adds
    // nothing on VC0: the boundary's VC0 receiver forwards nowhere, so no from-Z producer exists
    // on VC0 regardless of the extra port's role.
    return 1 + builder_config::num_downstream_edms_2d_vc0;
}

uint32_t RouterConnectionMapping::mesh_router_vc1_sender_count(ZPortRole z_role, bool express_routing_enabled) {
    if (z_role == ZPortRole::INTERMESH_BOUNDARY) {
        // Legacy non-self downstreams plus the from-Z slot for the boundary's VC1 fanout.
        return builder_config::num_downstream_edms_2d_vc1 + 1;
    }
    if (express_routing_enabled) {
        return express_mesh_vc1_sender_count();  // family max over facing of the express rule
    }
    // Legacy rule: every non-self cardinal receiver forwards into this router on VC1.
    return builder_config::num_downstream_edms_2d_vc1;
}

namespace {

// Emit the flat-index prefix sums and enforce the capacity ceilings at the one construction
// site. The ceiling checks are what turn the "express with VC2 reaches it exactly (5+4+1)"
// comment into a guarantee: every family's shape passes through here, with zero margin on
// senders and on the 32 stream registers the flat space maps onto.
void finalize_vc_shape_bases(RouterConnectionMapping::RouterVcShape& shape) {
    uint32_t sender_base = 0;
    uint32_t receiver_base = 0;
    for (size_t vc = 0; vc < shape.sender_counts.size(); ++vc) {
        shape.sender_flat_base[vc] = sender_base;
        sender_base += shape.sender_counts[vc];
        shape.receiver_flat_base[vc] = receiver_base;
        receiver_base += shape.receiver_counts[vc];
    }
    TT_FATAL(
        sender_base <= builder_config::num_max_sender_channels,
        "Router shape needs {} sender channels, over the {}-channel ceiling",
        sender_base,
        builder_config::num_max_sender_channels);
    TT_FATAL(
        receiver_base <= builder_config::num_max_receiver_channels,
        "Router shape needs {} receiver channels, over the {}-channel ceiling",
        receiver_base,
        builder_config::num_max_receiver_channels);
}

}  // namespace

RouterConnectionMapping::RouterVcShape RouterConnectionMapping::router_vc_shape(
    Topology topology,
    RoutingDirection facing,
    EdgeCapability edge_capability,
    ZPortRole z_role,
    bool express_routing_enabled,
    const IntermeshVCConfig* vc_config) {
    const bool z_boundary = (facing == RoutingDirection::Z && edge_capability == EdgeCapability::INTERMESH);
    const bool requires_vc1 = vc_config && vc_config->requires_vc1;
    const bool requires_vc2 = vc_config && vc_config->requires_vc2;

    RouterVcShape shape{};

    // num_vcs is config-only: the identical answer for 1D and 2D, by design. A 1D router can
    // therefore report more VCs than it has channels for, since VC1/VC2 channels are never
    // created on 1D. That oddity is preserved, not fixed -- get_all_sender_mappings() already
    // tolerates counts exceeding created channels. Whether any 1D configuration should ever set
    // requires_vc1 is a separate question, deliberately out of scope here.
    shape.num_vcs = requires_vc2 ? 3 : (requires_vc1 ? 2 : 1);

    // The intermesh boundary family is 2D-only by construction. Today the boundary VC0 arm of the
    // channel mapping precedes the 2D check, so a 1D Z boundary would silently report 5 -- with
    // topology in hand that becomes an explicit configuration error instead.
    TT_FATAL(!z_boundary || is_2D_topology(topology), "A Z-facing intermesh boundary router requires a 2D topology");

    if (!is_2D_topology(topology)) {
        // 1D counts: worker, plus one forwarding peer on Linear/Ring. No VC1 or VC2 channels are
        // ever created, independent of the num_vcs answer above.
        shape.sender_counts = {builder_config::get_num_used_sender_channel_count(topology), 0, 0};
        shape.receiver_counts = {1, 0, 0};
        finalize_vc_shape_bases(shape);
        return shape;
    }

    // A Z-facing boundary cannot exist without VC1: its whole shape is the from-boundary VC1
    // fanout. This is where the construction error lives now (moved from the channel mapping).
    TT_FATAL(!z_boundary || requires_vc1, "A Z-facing intermesh boundary router cannot be constructed without VC1");

    // VC0: worker + wired producers.
    const uint32_t vc0_senders = z_boundary ? builder_config::num_sender_channels_intermesh_z_boundary_vc0
                                            : mesh_router_vc0_sender_count(z_role, express_routing_enabled);

    // VC1: wired producers, when the VC exists.
    const uint32_t vc1_senders = !requires_vc1 ? 0
                                 : z_boundary  ? builder_config::num_sender_channels_intermesh_z_boundary_vc1
                                               : mesh_router_vc1_sender_count(z_role, express_routing_enabled);

    // VC2: one sender by VC2's own definition.
    const uint32_t vc2_senders = requires_vc2 ? 1 : 0;

    // Receivers: one per active carrier VC. The boundary services no VC2 receiver.
    const uint32_t vc0_receivers = 1;
    const uint32_t vc1_receivers = requires_vc1 ? 1 : 0;
    const uint32_t vc2_receivers = (requires_vc2 && !z_boundary) ? 1 : 0;

    shape.sender_counts = {vc0_senders, vc1_senders, vc2_senders};
    shape.receiver_counts = {vc0_receivers, vc1_receivers, vc2_receivers};
    finalize_vc_shape_bases(shape);
    return shape;
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

    // A port with no routing direction gets the boundary template: the full non-self set on VC1,
    // typed from-boundary. Its VC0 senders are fed by the mesh routers' MESH_TO_Z targets on their
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
            mapping.add_target(
                1,  // VC1
                0,  // Receiver channel 0
                ConnectionTarget(1, k_cardinal_directions[i]));
        }
        return mapping;
    }

    // A Z-facing router with a routing direction is an express chord; anything else is a
    // direction/capability contradiction.
    if (facing == RoutingDirection::Z) {
        TT_FATAL(
            edge_capability == EdgeCapability::INTRAMESH_EXPRESS,
            "A same-mesh Z edge is an express chord and must carry INTRAMESH_EXPRESS capability, got {}. "
            "An ordinary cardinal-capability Z edge cannot exist.",
            to_string(edge_capability));
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
        mapping.add_target(
            0,  // VC0
            0,  // receiver channel 1
            ConnectionTarget(0, opposite));
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
            mapping.add_target(0, 0, ConnectionTarget(0, target));
            if (enable_vc1 && enable_mesh_pass_through) {
                mapping.add_target(1, 0, ConnectionTarget(1, target));
            }
            continue;
        }
        mapping.add_target(0, 0, ConnectionTarget(0, target));
        if (enable_vc1) {
            mapping.add_target(1, 0, ConnectionTarget(1, target));
        }
    }

    return mapping;
}

}  // namespace tt::tt_fabric
