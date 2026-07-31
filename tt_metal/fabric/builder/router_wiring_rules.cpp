// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/router_wiring_rules.hpp"

#include "tt_metal/fabric/fabric_builder_context.hpp"

#include <algorithm>
#include <array>

namespace tt::tt_fabric {

RoutingDirection get_opposite_direction(RoutingDirection dir) {
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

constexpr std::array<RoutingDirection, 5> k_all_directions = {
    RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W, RoutingDirection::Z};

// The extra port's role from a per-direction capability set: absent means NONE, an intermesh Z is
// the boundary, anything else same-mesh is the chord.
ZPortRole z_role_of(const PerDirectionCapabilities& caps) {
    const auto& z = caps[static_cast<size_t>(RoutingDirection::Z)];
    if (!z.has_value()) {
        return ZPortRole::NONE;
    }
    return *z == EdgeCapability::INTERMESH ? ZPortRole::INTERMESH_BOUNDARY : ZPortRole::EXPRESS_CHORD;
}

// Per-VC sender arity of one 2D router of the ordinary forwarding family (legacy or express --
// everything except the Z-facing intermesh boundary, which has its own derived accessors in
// builder_config): the local worker (VC0 only) plus the producers wired into it on that VC.
// Called only by router_vc_shape; the answers are read off the shape, never re-derived at a
// consumption site.
uint32_t forwarding_vc0_sender_count(ZPortRole z_role, bool express_routing_enabled);
uint32_t forwarding_vc1_sender_count(ZPortRole z_role, bool express_routing_enabled);

// Emit the flat-index prefix sums and enforce the capacity ceilings at the one construction
// site. The ceiling checks are what turn the "express with VC2 reaches it exactly (5+4+1)"
// comment into a guarantee: every family's shape passes through here, with zero margin on
// senders and on the 32 stream registers the flat space maps onto.
void finalize_vc_shape_bases(RouterVcShape& shape) {
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

bool wires_into(
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
            // Legacy: the extra port exists in the set only as the boundary template's target.
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

PerDirectionCapabilities canonical_express_endpoint_capabilities() {
    PerDirectionCapabilities caps;
    for (size_t i = 0; i < caps.size() - 1; ++i) {
        caps[i] = EdgeCapability::INTRAMESH_CARDINAL;
    }
    caps[static_cast<size_t>(RoutingDirection::Z)] = EdgeCapability::INTRAMESH_EXPRESS;
    return caps;
}

uint32_t express_vc0_producer_arity(RoutingDirection direction, const PerDirectionCapabilities& caps) {
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

uint32_t express_mesh_vc0_sender_count() {
    // The canonical endpoint chip attains the structural ceiling: every Y and X producer wires
    // into an E/W facing under any capability assignment, so no per-chip set produces a wider one.
    const auto caps = canonical_express_endpoint_capabilities();
    uint32_t count = 0;
    for (const auto direction : k_all_directions) {
        count = std::max(count, express_vc0_producer_arity(direction, caps));
    }
    return count;
}

uint32_t express_mesh_vc1_sender_count() {
    // VC1 forwards the same wired producer set as VC0 on every facing, minus the worker slot, so
    // vc1_arity(f) == vc0_arity(f) - 1 for every facing f. The family max therefore commutes with
    // the subtraction: max_f(vc0_arity(f)) - 1 == max_f(vc0_arity(f) - 1).
    return express_mesh_vc0_sender_count() - 1;
}

namespace {

uint32_t forwarding_vc0_sender_count(ZPortRole /*z_role*/, bool express_routing_enabled) {
    if (express_routing_enabled) {
        return express_mesh_vc0_sender_count();  // family max over facing of the express rule
    }
    // Legacy rule: the worker plus every non-self cardinal producer. An intermesh Z edge adds
    // nothing on VC0: the boundary's VC0 receiver forwards nowhere, so no from-Z producer exists
    // on VC0 regardless of the extra port's role.
    return 1 + builder_config::num_downstream_edms_2d_vc0;
}

uint32_t forwarding_vc1_sender_count(ZPortRole z_role, bool express_routing_enabled) {
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

}  // namespace

RouterVcShape router_vc_shape(
    Topology topology,
    RoutingDirection facing,
    EdgeCapability edge_capability,
    ZPortRole z_role,
    bool express_routing_enabled,
    const IntermeshVCConfig* vc_config) {
    validate_facing_role_consistency(facing, edge_capability, z_role);

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
                                            : forwarding_vc0_sender_count(z_role, express_routing_enabled);

    // VC1: wired producers, when the VC exists.
    const uint32_t vc1_senders = !requires_vc1 ? 0
                                 : z_boundary  ? builder_config::num_sender_channels_intermesh_z_boundary_vc1
                                               : forwarding_vc1_sender_count(z_role, express_routing_enabled);

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

}  // namespace tt::tt_fabric
