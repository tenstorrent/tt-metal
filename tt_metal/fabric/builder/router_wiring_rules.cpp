// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/router_wiring_rules.hpp"

#include "tt_metal/fabric/fabric_builder_context.hpp"

#include <algorithm>
#include <array>
#include <enchantum/enchantum.hpp>

namespace tt::tt_fabric {

RoutingDirection get_opposite_direction(RoutingDirection dir) {
    switch (dir) {
        case RoutingDirection::N: return RoutingDirection::S;
        case RoutingDirection::S: return RoutingDirection::N;
        case RoutingDirection::E: return RoutingDirection::W;
        case RoutingDirection::W: return RoutingDirection::E;
        default:
            TT_FATAL(false, "Invalid routing direction for opposite calculation: {}", enchantum::to_string(dir));
            return dir;  // Unreachable
    }
}

namespace {

constexpr std::array<RoutingDirection, 5> k_all_directions = {
    RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W, RoutingDirection::Z};

constexpr std::array<RoutingDirection, 4> k_cardinal_directions = {
    RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W};

// What the turn matrix keys on, per port: the reduction of the (direction, capability) pair.
// Computed from both, substituting for neither -- direction and capability stay independent axes.
enum class TurnRole : uint8_t {
    UNRESTRICTED,   // Y cardinal (any capability), or an X landing: no dimension-order limit
    X_RING_ONLY,    // X cardinal, same-mesh: may only continue around the X ring
    EXPRESS_CHORD,  // Z carrying a same-mesh chord: a Y resource, like N/S
    BOUNDARY,       // Z crossing a mesh boundary: carries no routing direction
};

// Total over both enums, deliberately: whether a pair describes a possible chip is a chip-level
// fact, enforced once at establishment (classify_fabric_edge, validate_facing_role_consistency),
// and re-checking it per turn would re-derive a decision made upstream, one layer below the node
// context that makes a failure message useful. The fold is load-bearing, not cosmetic:
// (Z, INTRAMESH_CARDINAL) folds into EXPRESS_CHORD and (E/W, INTRAMESH_EXPRESS) into X_RING_ONLY
// because those reproduce the structurally implied answers; an IMPOSSIBLE-role mapping would be a
// behaviour change, not a cleanup.
TurnRole turn_role(RoutingDirection dir, EdgeCapability cap) {
    // Z carries a routing direction unless its edge crosses a mesh boundary.
    if (dir == RoutingDirection::Z) {
        return cap == EdgeCapability::INTERMESH ? TurnRole::BOUNDARY : TurnRole::EXPRESS_CHORD;
    }
    // A landing is a route root, not a packet mid-X-phase, so it is exempt from dimension order.
    if (is_x_axis_direction(dir)) {
        return cap == EdgeCapability::INTERMESH ? TurnRole::UNRESTRICTED : TurnRole::X_RING_ONLY;
    }
    return TurnRole::UNRESTRICTED;  // Y cardinal, any capability
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

// Downstream capacity canary, read off the emitted VC0 set: the boundary template's target has
// its own accounting, so the non-express check counts only the ordinary intramesh members,
// matching each mode's historical convention. Both 2D families run it -- the chord sits exactly
// at the express limit (four cardinals, four downstream EDMs), which is the zero-margin case a
// canary is for.
void check_vc0_downstream_capacity(const RouterTurnSet& turn_set, bool express_routing_enabled) {
    if (express_routing_enabled) {
        const size_t vc0_limit = builder_config::get_vc0_downstream_edm_count(/*is_2D_routing=*/true, /*express=*/true);
        TT_FATAL(
            turn_set[0].size() <= vc0_limit,
            "Express VC0 outbound direction count ({}) exceeds the downstream EDM count ({})",
            turn_set[0].size(),
            vc0_limit);
        return;
    }
    const size_t cardinals =
        static_cast<size_t>(std::count_if(turn_set[0].begin(), turn_set[0].end(), [](const ConnectionTarget& t) {
            return t.target_direction != RoutingDirection::Z;
        }));
    TT_FATAL(
        cardinals <= builder_config::num_downstream_edms_2d_vc0,
        "Outbound cardinal direction count ({}) exceeds the downstream EDM count ({})",
        cardinals,
        builder_config::num_downstream_edms_2d_vc0);
}

}  // namespace

bool wires_into(
    RoutingDirection producer_direction,
    EdgeCapability producer_capability,
    RoutingDirection egress_direction,
    ZPortRole chip_z_role,
    bool express_routing_enabled,
    uint32_t vc) {
    // A U-turn is a property of the turn, not of its ports: no router wires back over its own link
    // whatever they are. It needs no classification, so it answers first.
    if (producer_direction == egress_direction) {
        return false;
    }

    const TurnRole producer_role = turn_role(producer_direction, producer_capability);
    const bool egress_is_z = (egress_direction == RoutingDirection::Z);

    // A boundary producer's feed is VC-shaped in either mode: its VC1 receiver fans out onto every
    // non-self VC1 sender, while its VC0 receiver crosses over onto downstream VC1 senders and
    // feeds nothing on VC0. A physical fact about the boundary's receivers, not an express-mode
    // rule -- and the only VC-sensitive arm in this function: express_vc1_sender_count()'s
    // max-commutes-with-subtraction argument holds exactly while that stays true.
    if (producer_role == TurnRole::BOUNDARY) {
        return vc == 1;
    }

    // Non-express is producer-blind: every non-self cardinal wires in, and Z is a target only as
    // the boundary template's.
    if (!express_routing_enabled) {
        return !egress_is_z || chip_z_role == ZPortRole::INTERMESH_BOUNDARY;
    }

    switch (producer_role) {
        case TurnRole::UNRESTRICTED:   // no dimension-order limit applies to this producer
        case TurnRole::EXPRESS_CHORD:  // a chord is a Y resource; its feed rides every carrier VC
            // Any non-self egress, and Z only when the chip has a Z port.
            return !egress_is_z || chip_z_role != ZPortRole::NONE;
        case TurnRole::X_RING_ONLY:
            // Dimension order, stated once: an ordinary X producer may only continue around the X
            // ring. opposite(E/W) is never Z, so this also unwires X -> Y through the chord, with
            // no second copy of the rule.
            return egress_direction == get_opposite_direction(producer_direction);
        case TurnRole::BOUNDARY: break;  // answered above
    }
    TT_FATAL(false, "unreachable: every TurnRole is handled above");
    return false;
}

PerDirectionCapabilities canonical_express_endpoint_capabilities() {
    PerDirectionCapabilities caps;
    for (const auto direction : k_cardinal_directions) {
        caps.at(direction) = EdgeCapability::INTRAMESH_CARDINAL;
    }
    caps.at(RoutingDirection::Z) = EdgeCapability::INTRAMESH_EXPRESS;
    return caps;
}

uint32_t express_vc0_producer_arity(RoutingDirection direction, const PerDirectionCapabilities& caps) {
    // The Z-port role the wiring rule consults is this chip's own, not a global.
    const ZPortRole chip_z_role = z_role_of(caps);

    uint32_t count = 1;  // sender channel 0 is the local worker
    for (const auto producer : k_all_directions) {
        if (producer == direction) {
            continue;
        }
        const auto& capability = caps.at(producer);
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

    const bool requires_vc1 = vc_config && vc_config->requires_vc1;
    const bool requires_vc2 = vc_config && vc_config->requires_vc2;
    // Evaluated once per router: is this the chip's Z-facing intermesh boundary?
    const bool z_boundary_router = is_z_boundary_router(facing, edge_capability);

    // The boundary family's two construction preconditions, up front: a 1D Z boundary is a
    // configuration error, not a silently five-wide router.
    TT_FATAL(
        !z_boundary_router || is_2D_topology(topology), "A Z-facing intermesh boundary router requires a 2D topology");
    // And it cannot exist without VC1: its whole shape is the from-boundary VC1 fanout.
    TT_FATAL(
        !z_boundary_router || requires_vc1, "A Z-facing intermesh boundary router cannot be constructed without VC1");

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
        // VC0: worker + wired producers, by family. The chip's Z-port role adds nothing on
        // VC0: an intermesh boundary's VC0 receiver forwards nowhere, so no from-boundary
        // producer exists on VC0 regardless of the chip's role -- the role addend lands on VC1.
        if (z_boundary_router) {
            vc0_senders = boundary_vc0_sender_count();
        } else if (express_routing_enabled) {
            vc0_senders = express_vc0_sender_count();
        } else {
            // Frozen by standing decision: worker + every non-self cardinal.
            vc0_senders = non_express_vc0_sender_count();
        }

        // VC1: wired producers, by family, when the VC exists (zero by default).
        if (requires_vc1) {
            if (z_boundary_router) {
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
        vc2_receivers = (requires_vc2 && !z_boundary_router) ? 1 : 0;
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
    // Evaluated once per router: is this the chip's Z-facing intermesh boundary?
    const bool z_boundary_router = is_z_boundary_router(facing, edge_capability);

    // The boundary family's two construction preconditions, up front: a 1D Z boundary is a
    // configuration error, not a silently five-wide router.
    TT_FATAL(
        !z_boundary_router || is_2D_topology(topology), "A Z-facing intermesh boundary router requires a 2D topology");
    // And it cannot exist without VC1: its whole shape is the from-boundary VC1 fanout.
    TT_FATAL(
        !z_boundary_router || enable_vc1,
        "A Z-facing intermesh boundary router cannot be constructed without VC1: its entire "
        "shape is the from-boundary VC1 fanout");

    // One emission rule for every ordinary target: VC0 always, VC1 when the VC exists.
    const auto emit_on_enabled_vcs = [&turn_set, enable_vc1](RoutingDirection dir) {
        turn_set[0].push_back(ConnectionTarget(0, dir));
        if (enable_vc1) {
            turn_set[1].push_back(ConnectionTarget(1, dir));
        }
    };

    // The boundary template: the full non-self set on VC1, typed from-boundary. Its VC0 senders
    // are fed by the mesh routers' boundary targets on their own turn sets, so there is no VC0
    // arm here -- traffic arriving on its VC0 receiver crosses over onto these same VC1
    // downstream senders instead. Nothing in the turn matrix applies to it, which is why the
    // set is the full set: not a special case, a consequence.
    if (z_boundary_router) {
        for (const auto dir : k_cardinal_directions) {
            turn_set[1].push_back(ConnectionTarget(1, dir));
        }
        return turn_set;
    }

    // The express chord (the cross-check has already verified capability and role): a chord is a
    // Y resource, like N/S, with no opposite and no Z target of its own. Which cardinals
    // it feeds still comes from the primitive, so this set and the guard derivation -- which
    // reads the same primitive per producer -- cannot disagree.
    if (facing == RoutingDirection::Z) {
        TT_FATAL(
            express_routing_enabled && is_2D_topology(topology),
            "An express (Z) chord requires 2D Mesh/Torus routing with express routing enabled");
        for (const auto dir : k_cardinal_directions) {
            if (wires_into(facing, edge_capability, dir, chip_z_role, express_routing_enabled, 0)) {
                emit_on_enabled_vcs(dir);
            }
        }
        check_vc0_downstream_capacity(turn_set, express_routing_enabled);
        return turn_set;
    }

    // 1D: opposite only. There is no 1D boundary target: intermesh connections are rejected
    // upstream for 1D ("1D routing does not support intermesh connections"), so a 1D router with
    // role INTERMESH_BOUNDARY cannot occur in a valid configuration -- and get_router_connection_pairs
    // emits no Z pairs in 1D, so such a target would be unestablishable anyway.
    if (topology == Topology::Linear || topology == Topology::Ring) {
        turn_set[0].push_back(ConnectionTarget(0, get_opposite_direction(facing)));
        return turn_set;
    }

    // 2D cardinal-facing: each wired cardinal emitted directly -- opposite first, then the
    // remaining cardinals in enum order. Every member is what the primitive wires, so this set
    // and the guard derivation cannot disagree.
    const auto opposite = get_opposite_direction(facing);
    if (wires_into(facing, edge_capability, opposite, chip_z_role, express_routing_enabled, 0)) {
        emit_on_enabled_vcs(opposite);
    }
    for (const auto candidate : k_cardinal_directions) {
        if (candidate == facing || candidate == opposite) {
            continue;
        }
        if (wires_into(facing, edge_capability, candidate, chip_z_role, express_routing_enabled, 0)) {
            emit_on_enabled_vcs(candidate);
        }
    }

    // The Z target, last and on its own: whether it enters the set at all comes from the
    // primitive; how its targets are VC'd comes from the port's role. An intermesh boundary
    // stays on VC0 (VC1 only in pass-through mode); anything else rides every enabled carrier
    // VC -- a landed carrier can still decode a Z action, so there is no VC1->VC0 crossover.
    if (wires_into(facing, edge_capability, RoutingDirection::Z, chip_z_role, express_routing_enabled, 0)) {
        const bool boundary_target = (chip_z_role == ZPortRole::INTERMESH_BOUNDARY);
        turn_set[0].push_back(ConnectionTarget(0, RoutingDirection::Z));
        if (enable_vc1 && (!boundary_target || enable_mesh_pass_through)) {
            turn_set[1].push_back(ConnectionTarget(1, RoutingDirection::Z));
        }
    }

    // Downstream capacity canary, read off the emitted set (both 2D families run it).
    check_vc0_downstream_capacity(turn_set, express_routing_enabled);

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
