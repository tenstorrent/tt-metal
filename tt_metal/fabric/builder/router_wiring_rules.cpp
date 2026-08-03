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

constexpr std::array<RoutingDirection, 4> k_cardinal_directions = {
    RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W};

// The extra port's role from a per-direction capability set: absent means NONE, an intermesh Z is
// the boundary, anything else same-mesh is the chord.
ZPortRole z_role_of(const PerDirectionCapabilities& caps) {
    const auto& z = caps[static_cast<size_t>(RoutingDirection::Z)];
    if (!z.has_value()) {
        return ZPortRole::NONE;
    }
    return *z == EdgeCapability::INTERMESH ? ZPortRole::INTERMESH_BOUNDARY : ZPortRole::EXPRESS_CHORD;
}

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
    ZPortRole chip_z_role,
    bool express_routing_enabled,
    uint32_t vc) {
    if (producer_direction == egress_direction) {
        return false;  // a router never wires back over its own link
    }

    if (egress_direction == RoutingDirection::Z) {
        if (!express_routing_enabled) {
            // Non-express: the extra port exists in the set only as the boundary template's target.
            return chip_z_role == ZPortRole::INTERMESH_BOUNDARY;
        }
        // Express: Z is a legal target for every producer except an intramesh X one (dimension
        // order forbids X -> Y), and it is realized only when the chip has the port at all.
        const bool legal = producer_direction == RoutingDirection::Z || !is_x_axis_direction(producer_direction) ||
                           producer_capability == EdgeCapability::INTERMESH;
        return legal && chip_z_role != ZPortRole::NONE;
    }

    // A boundary producer's feed is VC-shaped in either mode: its VC1 receiver fans out onto
    // every non-self VC1 sender, while its VC0 receiver crosses over onto downstream VC1 senders
    // and feeds nothing on VC0. Hoisted above the non-express shortcut because this is a
    // physical fact about the boundary's receivers, not an express-mode rule -- without it a
    // non-express boundary producer would answer true on VC0, contradicting the count derivation.
    if (producer_direction == RoutingDirection::Z && producer_capability == EdgeCapability::INTERMESH) {
        return vc == 1;
    }

    if (!express_routing_enabled) {
        // Non-express: every non-self cardinal direction wires in, on every carrier VC.
        return true;
    }

    if (producer_direction == RoutingDirection::Z) {
        // Boundary producers are answered above; an express chord's feed rides every carrier VC.
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
    const ZPortRole chip_z_role = z_role_of(caps);

    uint32_t count = 1;  // sender channel 0 is the local worker
    for (const auto producer : k_all_directions) {
        if (producer == direction) {
            continue;
        }
        const auto& capability = caps[static_cast<size_t>(producer)];
        if (!capability.has_value()) {
            continue;  // direction absent on this chip: no producer to wire
        }
        if (wires_into(producer, *capability, direction, chip_z_role, /*express_routing_enabled=*/true, /*vc=*/0)) {
            ++count;
        }
    }
    return count;
}

uint32_t express_vc0_sender_count() {
    // The canonical endpoint chip attains the structural ceiling: every Y and X producer wires
    // into an E/W facing under any capability assignment, so no per-chip set produces a wider one.
    const auto caps = canonical_express_endpoint_capabilities();
    uint32_t count = 0;
    for (const auto direction : k_all_directions) {
        count = std::max(count, express_vc0_producer_arity(direction, caps));
    }
    return count;
}

uint32_t express_vc1_sender_count() {
    // vc1_arity(f) == vc0_arity(f) - 1 for every facing f: on the canonical endpoint chip no
    // producer is VC-sensitive -- the only VC-sensitive arm of wires_into is an INTERMESH Z
    // producer, and that chip's Z is INTRAMESH_EXPRESS. The family max therefore commutes with
    // the worker-slot subtraction: max_f(vc0_arity(f)) - 1 == max_f(vc0_arity(f) - 1).
    return express_vc0_sender_count() - 1;
}

RouterVcShape router_vc_shape(
    Topology topology,
    RoutingDirection facing,
    EdgeCapability edge_capability,
    ZPortRole chip_z_role,
    bool express_routing_enabled,
    const IntermeshVCConfig* vc_config) {
    validate_facing_role_consistency(facing, edge_capability, chip_z_role);

    const bool is_z_boundary_router = (facing == RoutingDirection::Z && edge_capability == EdgeCapability::INTERMESH);
    const bool requires_vc1 = vc_config && vc_config->requires_vc1;
    const bool requires_vc2 = vc_config && vc_config->requires_vc2;

    // The boundary family's two construction preconditions, up front. It is 2D-only by
    // construction: today the boundary VC0 arm of the channel mapping precedes the 2D check, so a
    // 1D Z boundary would silently report 5 -- with topology in hand that becomes an explicit
    // configuration error instead.
    TT_FATAL(
        !is_z_boundary_router || is_2D_topology(topology),
        "A Z-facing intermesh boundary router requires a 2D topology");
    // And it cannot exist without VC1: its whole shape is the from-boundary VC1 fanout. This is
    // where the construction error lives now (moved from the channel mapping).
    TT_FATAL(
        !is_z_boundary_router || requires_vc1,
        "A Z-facing intermesh boundary router cannot be constructed without VC1");

    RouterVcShape shape{};

    // num_vcs is config-only: the identical answer for 1D and 2D, by design. A 1D router can
    // therefore report more VCs than it has channels for, since VC1/VC2 channels are never
    // created on 1D. That oddity is preserved, not fixed -- consumers already tolerate counts
    // exceeding created channels. Whether any 1D configuration should ever set requires_vc1 is a
    // separate question, deliberately out of scope here.
    shape.num_vcs = requires_vc2 ? 3 : (requires_vc1 ? 2 : 1);

    // Per-VC counts: zero unless an arm sets them. VC0 always has its receiver.
    uint32_t vc0_senders = 0, vc1_senders = 0, vc2_senders = 0;
    uint32_t vc0_receivers = 1, vc1_receivers = 0, vc2_receivers = 0;

    if (!is_2D_topology(topology)) {
        // 1D counts from the config accessor: Linear/Ring are worker plus one forwarding peer,
        // NeighborExchange is worker-only. NeighborExchange takes this arm too, so a literal 2
        // would move it -- the accessor exists precisely because three topologies land here.
        // No VC1 or VC2 channels are ever created, independent of the num_vcs answer above.
        vc0_senders = builder_config::get_num_used_sender_channel_count(topology);
    } else {
        // VC0: worker + wired producers, by family. The chip's extra-port role adds nothing on
        // VC0: an intermesh boundary's VC0 receiver forwards nowhere, so no from-boundary
        // producer exists on VC0 regardless of the chip's role -- the role addend lands on VC1.
        if (is_z_boundary_router) {
            vc0_senders = boundary_vc0_sender_count();
        } else if (express_routing_enabled) {
            vc0_senders = express_vc0_sender_count();
        } else {
            // Frozen by standing decision: worker + every non-self cardinal.
            vc0_senders = non_express_vc0_sender_count();
        }

        // VC1: wired producers, by family, when the VC exists (zero by default).
        if (requires_vc1) {
            if (is_z_boundary_router) {
                vc1_senders = boundary_vc1_sender_count();
            } else if (chip_z_role == ZPortRole::INTERMESH_BOUNDARY) {
                // A mesh router on a boundary chip gains the from-boundary slot; its width is the
                // frozen one in either mode (the express family's 4 coincides by unrelated arithmetic).
                vc1_senders = non_express_vc1_sender_count() + 1;
            } else if (express_routing_enabled) {
                vc1_senders = express_vc1_sender_count();
            } else {
                vc1_senders = non_express_vc1_sender_count();
            }
        }

        // VC2: one sender by VC2's own definition.
        vc2_senders = requires_vc2 ? 1 : 0;

        // Receivers: one per active carrier VC. The boundary services no VC2 receiver.
        vc1_receivers = requires_vc1 ? 1 : 0;
        vc2_receivers = (requires_vc2 && !is_z_boundary_router) ? 1 : 0;
    }

    shape.sender_counts = {vc0_senders, vc1_senders, vc2_senders};
    shape.receiver_counts = {vc0_receivers, vc1_receivers, vc2_receivers};
    finalize_vc_shape_bases(shape);
    return shape;
}

RouterTurnSet turn_set_for_router(
    Topology topology,
    RoutingDirection facing,
    EdgeCapability edge_capability,
    ZPortRole chip_z_role,
    bool express_routing_enabled,
    const IntermeshVCConfig* vc_config) {
    RouterTurnSet turn_set{};

    validate_facing_role_consistency(facing, edge_capability, chip_z_role);

    // The VC facts are read off the same config the shape derivation consumes.
    const bool enable_vc1 = vc_config && vc_config->requires_vc1;
    const bool enable_mesh_pass_through = vc_config && vc_config->requires_vc1_mesh_pass_through;

    // A port with no routing direction gets the boundary template: the full non-self set on VC1,
    // typed from-boundary. Its VC0 senders are fed by the mesh routers' boundary targets on their
    // own turn sets, so there is no VC0 arm here -- traffic arriving on its VC0 receiver crosses
    // over onto these same VC1 downstream senders instead. Nothing in the turn matrix applies to
    // it, which is why the set is the full set: not a special case, a consequence. The
    // requirements match the shape derivation (router_vc_shape): the boundary is 2D-only and
    // exists only with VC1, since its entire shape is the from-boundary VC1 fanout.
    if (!carries_routing_direction(facing, edge_capability)) {
        TT_FATAL(is_2D_topology(topology), "A Z-facing intermesh boundary router requires a 2D topology");
        TT_FATAL(
            enable_vc1,
            "A Z-facing intermesh boundary router cannot be constructed without VC1: its entire "
            "shape is the from-boundary VC1 fanout");
        for (size_t i = 0; i < k_cardinal_directions.size(); ++i) {
            turn_set[1].push_back(ConnectionTarget(1, k_cardinal_directions[i]));
        }
        return turn_set;
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
        turn_set[0].push_back(ConnectionTarget(0, opposite));
        return turn_set;
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
        wires_into(facing, edge_capability, opposite, chip_z_role, express_routing_enabled, 0)) {
        outbound.push_back(opposite);
    }
    for (const auto candidate : k_cardinal_directions) {
        if (candidate == facing || candidate == opposite) {
            continue;
        }
        if (wires_into(facing, edge_capability, candidate, chip_z_role, express_routing_enabled, 0)) {
            outbound.push_back(candidate);
        }
    }
    if (facing != RoutingDirection::Z &&
        wires_into(facing, edge_capability, RoutingDirection::Z, chip_z_role, express_routing_enabled, 0)) {
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
        if (target == RoutingDirection::Z && chip_z_role == ZPortRole::INTERMESH_BOUNDARY) {
            // The boundary template's target: VC0 always, VC1 only in pass-through mode.
            turn_set[0].push_back(ConnectionTarget(0, target));
            if (enable_vc1 && enable_mesh_pass_through) {
                turn_set[1].push_back(ConnectionTarget(1, target));
            }
            continue;
        }
        turn_set[0].push_back(ConnectionTarget(0, target));
        if (enable_vc1) {
            turn_set[1].push_back(ConnectionTarget(1, target));
        }
    }

    return turn_set;
}

RouterArchetype router_archetype(
    Topology topology,
    RoutingDirection facing,
    EdgeCapability edge_capability,
    ZPortRole chip_z_role,
    bool express_routing_enabled,
    const IntermeshVCConfig* vc_config) {
    return RouterArchetype{
        router_vc_shape(topology, facing, edge_capability, chip_z_role, express_routing_enabled, vc_config),
        turn_set_for_router(topology, facing, edge_capability, chip_z_role, express_routing_enabled, vc_config)};
}

}  // namespace tt::tt_fabric
