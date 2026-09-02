// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <iostream>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"
#include "tt_metal/fabric/builder/router_wiring_rules.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"

using namespace tt::tt_fabric;

/**
 * Router turn-set Tests
 *
 * The per-VC turn table of one router (turn_set_for_router in builder/router_wiring_rules.*),
 * by family. Behaviour is keyed on port roles, not router "types": a cardinal-facing router gets
 * the non-express or express turn set, the chip's extra port enters the turn set only through the
 * chip's ZPortRole, and a Z-facing router whose edge is INTERMESH gets the from-boundary fanout.
 * Covered here:
 *
 * - 1D: opposite-direction only, independent of the chip's extra-port role.
 * - Non-express 2D: every non-self cardinal, plus the boundary target when the chip's
 *   extra port is an intermesh boundary; VC1 mirrors the cardinals; pass-through adds the
 *   boundary target on VC1.
 * - The boundary template: VC1 fanout to every mesh direction, nothing on VC0; requires 2D+VC1.
 * - The express chord: a Z-facing INTRAMESH_EXPRESS router is wired as an ordinary routing
 *   direction on every carrier VC; a cardinal-capability Z edge is a configuration error.
 *
 * The wiring rules these sets are read off live in wires_into; the primitive itself is pinned
 * directly in test_express_connection_wiring.cpp so the two cannot drift.
 */

namespace {

// The two VC configurations the turn-set derivation consumes, spelled as configs rather than
// booleans -- the same objects the shape derivation takes.
const IntermeshVCConfig k_full_mesh = IntermeshVCConfig::full_mesh();
const IntermeshVCConfig k_full_mesh_pass_through = IntermeshVCConfig::full_mesh_with_pass_through();

// The set of directions a router's receiver forwards to on one VC.
std::set<RoutingDirection> target_directions(const RouterTurnSet& turn_set, uint32_t vc) {
    std::set<RoutingDirection> out;
    for (const auto& t : turn_set[vc]) {
        EXPECT_TRUE(t.target_direction.has_value());
        out.insert(*t.target_direction);
    }
    return out;
}

void expect_all_targets_on_vc(const RouterTurnSet& turn_set, uint32_t vc, uint32_t expected_target_vc) {
    for (const auto& t : turn_set[vc]) {
        EXPECT_EQ(t.target_vc, expected_target_vc);
    }
}

const std::set<RoutingDirection> k_all_cardinals = {
    RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W};

std::set<RoutingDirection> non_self_cardinals(RoutingDirection facing) {
    auto out = k_all_cardinals;
    out.erase(facing);
    return out;
}

// One chip, named by the only fact that distinguishes the families below: what its Z port is for.
// Every cardinal is an ordinary same-mesh edge. Deriving both facing capability and Z role from
// this set prevents inconsistent pairings; validate_facing_role_consistency is tested directly.
PerDirectionCapabilities chip_with_z(std::optional<EdgeCapability> z_capability) {
    PerDirectionCapabilities caps;
    for (const auto direction : k_all_cardinals) {
        caps.at(direction) = EdgeCapability::INTRAMESH_CARDINAL;
    }
    caps.at(RoutingDirection::Z) = z_capability;
    return caps;
}

RoutingDirection opposite_of(RoutingDirection facing) {
    switch (facing) {
        case RoutingDirection::N: return RoutingDirection::S;
        case RoutingDirection::S: return RoutingDirection::N;
        case RoutingDirection::E: return RoutingDirection::W;
        case RoutingDirection::W: return RoutingDirection::E;
        default: return facing;
    }
}

}  // namespace

class RouterTurnSetTest : public ::testing::Test {};

// ============================================================================
// 1D: the opposite direction only, regardless of the chip's extra-port role
// ============================================================================

TEST_F(RouterTurnSetTest, Linear1D_WiresOnlyTheOpposite) {
    // A 1D router's whole turn set is the opposite direction. The extra port plays no role:
    // intermesh connections are rejected upstream for 1D, and get_router_connection_pairs emits
    // no Z pairs, so a boundary or chord target would be unestablishable anyway.
    for (auto topology : {Topology::Linear, Topology::Ring}) {
        for (auto z_capability :
             {std::optional<EdgeCapability>{},
              std::optional{EdgeCapability::INTERMESH},
              std::optional{EdgeCapability::INTRAMESH_EXPRESS}}) {
            for (auto facing : {RoutingDirection::N, RoutingDirection::E}) {
                const auto turn_set = turn_set_for_router(
                    topology, facing, chip_with_z(z_capability), /*express_routing_enabled=*/false, nullptr);

                const auto& targets = turn_set[0];
                ASSERT_EQ(targets.size(), 1) << "topology " << enchantum::to_string(topology) << " z "
                                             << (z_capability ? enchantum::to_string(*z_capability) : "absent")
                                             << " facing " << enchantum::to_string(facing);
                EXPECT_EQ(*targets[0].target_direction, opposite_of(facing));
                EXPECT_EQ(targets[0].target_vc, 0);
                EXPECT_TRUE(turn_set[1].empty());
            }
        }
    }
}

// ============================================================================
// Non-express 2D
// ============================================================================

TEST_F(RouterTurnSetTest, NonExpress2D_WiresEveryNonSelfCardinal) {
    for (auto topology : {Topology::Mesh, Topology::Torus}) {
        for (auto facing : k_all_cardinals) {
            const auto turn_set = turn_set_for_router(
                topology,
                facing,
                chip_with_z(std::nullopt),
                /*express_routing_enabled=*/false,
                nullptr);

            EXPECT_EQ(target_directions(turn_set, 0), non_self_cardinals(facing))
                << "facing " << enchantum::to_string(facing);
            expect_all_targets_on_vc(turn_set, 0, 0);
        }
    }
}

TEST_F(RouterTurnSetTest, NonExpress2D_KeepsWiredButUnusedXToYTurns) {
    // Standing decision: an E/W-facing non-express router is wired into the Y directions even though
    // 2D routing is already dimension-ordered and never uses those turns. Removing them would move
    // downstream counts, stream assignment, and L1 layout on every existing 2D configuration.
    const auto turn_set = turn_set_for_router(
        Topology::Mesh,
        RoutingDirection::E,
        chip_with_z(std::nullopt),
        /*express_routing_enabled=*/false,
        nullptr);

    const auto dirs = target_directions(turn_set, 0);
    EXPECT_TRUE(dirs.contains(RoutingDirection::N));
    EXPECT_TRUE(dirs.contains(RoutingDirection::S));
    EXPECT_TRUE(dirs.contains(RoutingDirection::W));
}

TEST_F(RouterTurnSetTest, NonExpress2D_BoundaryChipAddsBoundaryTargetOnVC0) {
    // The chip's extra port enters the turn set when it is an intermesh boundary: the three
    // non-self cardinals plus the boundary turn, which stays on VC0.
    for (auto facing : k_all_cardinals) {
        const auto turn_set = turn_set_for_router(
            Topology::Mesh,
            facing,
            chip_with_z(EdgeCapability::INTERMESH),
            /*express_routing_enabled=*/false,
            nullptr);

        auto expected = non_self_cardinals(facing);
        expected.insert(RoutingDirection::Z);
        EXPECT_EQ(target_directions(turn_set, 0), expected) << "facing " << enchantum::to_string(facing);
        expect_all_targets_on_vc(turn_set, 0, 0);
    }
}

TEST_F(RouterTurnSetTest, NonExpress2D_VC1MirrorsCardinalsOnly) {
    // VC1 forwards the same cardinal set, but the boundary target stays off VC1: feeding the
    // boundary's VC1 sender while it does not service VC1 would create an undrained channel.
    const auto turn_set = turn_set_for_router(
        Topology::Mesh,
        RoutingDirection::N,
        chip_with_z(EdgeCapability::INTERMESH),
        /*express_routing_enabled=*/false,
        &k_full_mesh);

    EXPECT_EQ(target_directions(turn_set, 1), non_self_cardinals(RoutingDirection::N));
    expect_all_targets_on_vc(turn_set, 1, 1);

    auto expected_vc0 = non_self_cardinals(RoutingDirection::N);
    expected_vc0.insert(RoutingDirection::Z);
    EXPECT_EQ(target_directions(turn_set, 0), expected_vc0);
}

TEST_F(RouterTurnSetTest, PassThrough_AddsBoundaryTargetOnVC1) {
    // EXPERIMENTAL pass-through (A->B->C) forwards VC1 traffic to the local boundary as well.
    for (auto topology : {Topology::Mesh, Topology::Torus}) {
        const auto turn_set = turn_set_for_router(
            topology,
            RoutingDirection::E,
            chip_with_z(EdgeCapability::INTERMESH),
            /*express_routing_enabled=*/false,
            &k_full_mesh_pass_through);

        auto expected_vc1 = non_self_cardinals(RoutingDirection::E);
        expected_vc1.insert(RoutingDirection::Z);
        EXPECT_EQ(target_directions(turn_set, 1), expected_vc1) << "topology " << enchantum::to_string(topology);
        expect_all_targets_on_vc(turn_set, 1, 1);

        // No aliasing: every VC1 target names a distinct direction.
        EXPECT_EQ(turn_set[1].size(), expected_vc1.size());
    }
}

TEST_F(RouterTurnSetTest, PassThrough_NoEffectWithoutBoundaryPort) {
    // Pass-through requested on a chip whose extra port is absent or is a chord: there is no
    // boundary to forward to, so no Z target appears on either VC.
    const auto turn_set = turn_set_for_router(
        Topology::Mesh,
        RoutingDirection::E,
        chip_with_z(std::nullopt),
        /*express_routing_enabled=*/false,
        &k_full_mesh_pass_through);

    EXPECT_EQ(target_directions(turn_set, 0), non_self_cardinals(RoutingDirection::E));
    EXPECT_EQ(target_directions(turn_set, 1), non_self_cardinals(RoutingDirection::E));
}

// ============================================================================
// The boundary template (Z-facing router whose edge crosses a mesh boundary)
// ============================================================================

TEST_F(RouterTurnSetTest, BoundaryTemplate_FansOutToEveryMeshDirectionOnVC1) {
    // The boundary's whole shape is its from-boundary VC1 fanout: nothing forwards off its VC0
    // receiver (traffic arriving there crosses over onto these same VC1 downstream senders).
    const auto turn_set = turn_set_for_router(
        Topology::Mesh,
        RoutingDirection::Z,
        chip_with_z(EdgeCapability::INTERMESH),
        /*express_routing_enabled=*/false,
        &k_full_mesh);

    EXPECT_TRUE(turn_set[0].empty());

    const auto& targets = turn_set[1];
    ASSERT_EQ(targets.size(), 4);
    // Fanout is emitted in cardinal enum order.
    const std::vector<RoutingDirection> expected_order = {
        RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W};
    for (size_t i = 0; i < expected_order.size(); ++i) {
        ASSERT_TRUE(targets[i].target_direction.has_value());
        EXPECT_EQ(*targets[i].target_direction, expected_order[i]);
        EXPECT_EQ(targets[i].target_vc, 1);
    }
}

TEST_F(RouterTurnSetTest, BoundaryTemplate_Requires2DAndVC1) {
    // Its entire shape is the from-boundary VC1 fanout, so without VC1 or on 1D the boundary
    // router cannot be constructed.
    EXPECT_ANY_THROW(turn_set_for_router(
        Topology::Mesh,
        RoutingDirection::Z,
        chip_with_z(EdgeCapability::INTERMESH),
        /*express_routing_enabled=*/false,
        nullptr));

    EXPECT_ANY_THROW(turn_set_for_router(
        Topology::Linear,
        RoutingDirection::Z,
        chip_with_z(EdgeCapability::INTERMESH),
        /*express_routing_enabled=*/false,
        &k_full_mesh));
}

// ============================================================================
// The express chord (Z-facing router whose edge is a same-mesh express chord)
// ============================================================================

TEST_F(RouterTurnSetTest, ExpressChord_IsWiredAsAnOrdinaryRoutingDirection) {
    // A chord is a Y-axis resource like N/S: it fans out to all four cardinals as ordinary
    // same-VC turns, on VC0 and VC1 alike (a landed carrier can still decode a Z action).
    const auto turn_set = turn_set_for_router(
        Topology::Torus,
        RoutingDirection::Z,
        chip_with_z(EdgeCapability::INTRAMESH_EXPRESS),
        /*express_routing_enabled=*/true,
        &k_full_mesh);

    for (uint32_t vc : {0u, 1u}) {
        // No Z in the set: a router never wires back over its own link.
        EXPECT_EQ(target_directions(turn_set, vc), k_all_cardinals) << "VC" << vc;
        expect_all_targets_on_vc(turn_set, vc, vc);
    }
}

TEST_F(RouterTurnSetTest, ExpressChord_RequiresExpressEnabledAnd2D) {
    EXPECT_ANY_THROW(turn_set_for_router(
        Topology::Torus,
        RoutingDirection::Z,
        chip_with_z(EdgeCapability::INTRAMESH_EXPRESS),
        /*express_routing_enabled=*/false,
        nullptr));

    EXPECT_ANY_THROW(turn_set_for_router(
        Topology::Linear,
        RoutingDirection::Z,
        chip_with_z(EdgeCapability::INTRAMESH_EXPRESS),
        /*express_routing_enabled=*/true,
        nullptr));
}

TEST_F(RouterTurnSetTest, CardinalCapabilityOnZFacingIsAConfigurationError) {
    // Direction letter and capability disagree: a same-mesh Z edge is an express chord and must
    // carry INTRAMESH_EXPRESS; an ordinary cardinal-capability Z edge cannot exist.
    EXPECT_ANY_THROW(turn_set_for_router(
        Topology::Mesh,
        RoutingDirection::Z,
        chip_with_z(EdgeCapability::INTRAMESH_CARDINAL),
        /*express_routing_enabled=*/false,
        &k_full_mesh));
    EXPECT_ANY_THROW(router_vc_shape(
        Topology::Mesh,
        RoutingDirection::Z,
        chip_with_z(EdgeCapability::INTRAMESH_CARDINAL),
        /*express_routing_enabled=*/false,
        nullptr));
}

// ============================================================================
// Role/capability cross-check: the two spellings of the chip's extra port must agree
// ============================================================================

TEST_F(RouterTurnSetTest, ExtraPortRoleAndCapabilityCannotDisagree) {
    // A Z-facing intermesh edge means the chip's extra port IS the boundary, and a same-mesh Z
    // edge means it is the chord. Claiming otherwise describes an impossible chip.
    //
    // The derivations no longer take the role beside the capability -- they read it off the
    // capability set -- so these pairings can no longer be handed to them at all, which is a
    // stronger guarantee than rejecting them at runtime. What is left to test is the cross-check
    // itself, which still guards every caller carrying the two facts separately, and that the
    // set-derived spelling produces the agreeing half of each pair.
    EXPECT_ANY_THROW(validate_facing_role_consistency(RoutingDirection::Z, EdgeCapability::INTERMESH, ZPortRole::NONE));
    EXPECT_ANY_THROW(
        validate_facing_role_consistency(RoutingDirection::Z, EdgeCapability::INTERMESH, ZPortRole::EXPRESS_CHORD));
    EXPECT_ANY_THROW(
        validate_facing_role_consistency(RoutingDirection::Z, EdgeCapability::INTRAMESH_EXPRESS, ZPortRole::NONE));
    EXPECT_ANY_THROW(validate_facing_role_consistency(
        RoutingDirection::Z, EdgeCapability::INTRAMESH_EXPRESS, ZPortRole::INTERMESH_BOUNDARY));

    EXPECT_EQ(z_role_of(chip_with_z(EdgeCapability::INTERMESH)), ZPortRole::INTERMESH_BOUNDARY);
    EXPECT_EQ(z_role_of(chip_with_z(EdgeCapability::INTRAMESH_EXPRESS)), ZPortRole::EXPRESS_CHORD);
    EXPECT_EQ(z_role_of(chip_with_z(std::nullopt)), ZPortRole::NONE);
}

TEST_F(RouterTurnSetTest, CardinalFacingRejectsExpressCapability) {
    // An express chord lives on the chip's extra port; a cardinal-facing router cannot carry it,
    // no matter what the chip's Z port is doing.
    auto caps = chip_with_z(EdgeCapability::INTRAMESH_EXPRESS);
    caps.at(RoutingDirection::N) = EdgeCapability::INTRAMESH_EXPRESS;

    EXPECT_ANY_THROW(
        turn_set_for_router(Topology::Torus, RoutingDirection::N, caps, /*express_routing_enabled=*/true, nullptr));
    EXPECT_ANY_THROW(
        router_vc_shape(Topology::Torus, RoutingDirection::N, caps, /*express_routing_enabled=*/true, nullptr));
}

// ============================================================================
// Queries and value semantics
// ============================================================================

TEST_F(RouterTurnSetTest, Queries_OnAbsentVcsReturnEmpty) {
    RouterTurnSet empty{};
    EXPECT_TRUE(empty[0].empty());
    EXPECT_TRUE(empty[1].empty());

    const auto mesh = turn_set_for_router(
        Topology::Mesh,
        RoutingDirection::N,
        chip_with_z(std::nullopt),
        /*express_routing_enabled=*/false,
        nullptr);
    EXPECT_TRUE(mesh[1].empty()) << "VC1 not enabled";
    EXPECT_TRUE(mesh[2].empty());
}

TEST_F(RouterTurnSetTest, ConnectionTarget_Semantics) {
    ConnectionTarget target(1, RoutingDirection::Z);
    EXPECT_EQ(target.target_vc, 1);
    ASSERT_TRUE(target.target_direction.has_value());
    EXPECT_EQ(*target.target_direction, RoutingDirection::Z);
}

// ============================================================================
// The express multi-mesh landing path, over a CARDINAL seam
// ============================================================================
//
// A mesh graph descriptor names an intermesh connection without naming ports, so which ports the
// seam lands on is discovery's answer, not the descriptor's. On the 16x4x2 express fixture it lands
// on the cardinal N/S edge ports, because the chips' Z ports are already spent on express chords.
// Landed packets therefore combine express mode, a cardinal INTERMESH facing, and EXPRESS_CHORD Z role.

namespace {

// The bijection's producer index is in eth order, so producers and egresses have to be spelled that
// way to index a downstream's sender channels. Written out here rather than reached for through the
// control plane, which would need a cluster for a five-way switch.
eth_chan_directions to_eth(RoutingDirection direction) {
    switch (direction) {
        case RoutingDirection::N: return eth_chan_directions::NORTH;
        case RoutingDirection::E: return eth_chan_directions::EAST;
        case RoutingDirection::S: return eth_chan_directions::SOUTH;
        case RoutingDirection::W: return eth_chan_directions::WEST;
        case RoutingDirection::Z: return eth_chan_directions::Z;
        default: ADD_FAILURE() << "not a port direction"; return eth_chan_directions::EAST;
    }
}

// One chip of the destination mesh: cardinals are ordinary same-mesh edges except for an optional
// seam direction, and the extra port is the express chord.
PerDirectionCapabilities express_chip_with_seam(std::optional<RoutingDirection> seam_facing) {
    auto caps = chip_with_z(EdgeCapability::INTRAMESH_EXPRESS);
    if (seam_facing.has_value()) {
        caps.at(*seam_facing) = EdgeCapability::INTERMESH;
    }
    return caps;
}

RouterVcShape express_shape_of(RoutingDirection facing, std::optional<RoutingDirection> seam_facing) {
    return router_vc_shape(
        Topology::Torus,
        facing,
        express_chip_with_seam(seam_facing),
        /*express_routing_enabled=*/true,
        &k_full_mesh);
}

RouterTurnSet express_turns_of(RoutingDirection facing, std::optional<RoutingDirection> seam_facing) {
    return turn_set_for_router(
        Topology::Torus,
        facing,
        express_chip_with_seam(seam_facing),
        /*express_routing_enabled=*/true,
        &k_full_mesh);
}

}  // namespace

TEST_F(RouterTurnSetTest, ExpressCardinalSeam_IsNotTheBoundaryTemplate) {
    // The from-boundary template is keyed on a Z-facing INTERMESH edge, so a cardinal seam never
    // reaches it: the router keeps a real routing direction and is wired by the ordinary express
    // rule. Pinned because the crossover onto VC1 and the landing intercept both key on the plain
    // "my eth peer is in another mesh" fact, which a cardinal seam does satisfy -- the two halves
    // disagree about what a seam is, and only this half notices the difference.
    const auto seam = express_turns_of(RoutingDirection::S, RoutingDirection::S);

    // The boundary template's signature is an empty VC0 and a four-wide VC1 fanout.
    EXPECT_FALSE(seam[0].empty()) << "a cardinal seam still forwards on VC0";

    // It is wired exactly like the same router would be without the seam: capability does not enter
    // the express turn rule for a Y facing, only the direction's role does.
    EXPECT_EQ(target_directions(seam, 1), target_directions(express_turns_of(RoutingDirection::S, std::nullopt), 1));
}

TEST_F(RouterTurnSetTest, ExpressCardinalSeam_XRouterKeepsTheSeamAsATarget) {
    // Dimension order stops an X producer turning back into the mesh's Y rings, and on an express
    // chip that leaves an E/W router with just one downstream: its opposite. The seam is the
    // exception the rule is written around (contract 4.4 keys both sides on capability): leaving
    // the mesh is not a turn back into a protected Y ring, so an INTERMESH egress stays wired
    // wherever discovery put it.
    //
    // The kernel's intermesh egress is what depends on this. It picks the boundary direction per
    // destination mesh, not from the decoded action, so an exit-bound packet can arrive on any
    // receiver -- including the E/W one whose last intramesh leg was an X hop. If the seam is not
    // in that router's turn set there is no downstream slot to hand the packet to, and it is
    // dropped at the exit chip with no error anywhere: the maps say deliver locally, and locally
    // is where it stays.
    for (const auto seam_facing : {RoutingDirection::N, RoutingDirection::S}) {
        for (const auto x_facing : {RoutingDirection::E, RoutingDirection::W}) {
            SCOPED_TRACE(
                "seam " + std::string(enchantum::to_string(seam_facing)) + ", facing " +
                std::string(enchantum::to_string(x_facing)));

            const auto turns = express_turns_of(x_facing, seam_facing);
            const auto expected = std::set<RoutingDirection>{opposite_of(x_facing), seam_facing};

            EXPECT_EQ(target_directions(turns, 0), expected);
            EXPECT_EQ(target_directions(turns, 1), expected);

            // The ordinary Y cardinal on the same chip is still unwired, so the exemption is the
            // seam's capability rather than a hole in dimension order.
            EXPECT_EQ(
                target_directions(express_turns_of(x_facing, std::nullopt), 0),
                std::set<RoutingDirection>{opposite_of(x_facing)});
        }
    }
}

TEST_F(RouterTurnSetTest, ExpressCardinalSeam_EveryWiredVc1TurnIsPlaceableDownstream) {
    // The invariant a landed packet depends on: for every VC1 turn the derivation wires, the
    // DOWNSTREAM router must own a VC1 sender channel at the index the direction<->slot bijection
    // assigns that producer. A turn that is wired but whose slot index falls outside the
    // downstream's VC1 sender count cannot be established, and traffic needing it is dropped
    // silently rather than rejected at build time -- which is indistinguishable at the endpoint
    // from a routing bug. Checked over every producer/egress pair, on a chip with and without the
    // seam, rather than for one hand-picked turn.
    for (const auto seam_facing : {std::optional<RoutingDirection>{}, std::optional{RoutingDirection::S}}) {
        for (const auto producer :
             {RoutingDirection::N,
              RoutingDirection::E,
              RoutingDirection::S,
              RoutingDirection::W,
              RoutingDirection::Z}) {
            SCOPED_TRACE(
                "seam " + std::string(seam_facing ? enchantum::to_string(*seam_facing) : "none") + ", producer " +
                std::string(enchantum::to_string(producer)));

            const auto turns = express_turns_of(producer, seam_facing);
            for (const auto& target : turns[1]) {
                ASSERT_TRUE(target.target_direction.has_value());
                const auto egress = *target.target_direction;
                const auto egress_shape = express_shape_of(egress, seam_facing);

                const uint32_t slot = builder::get_downstream_sender_channel_for_vc(
                    /*is_2d_routing=*/true, target.target_vc, to_eth(producer), to_eth(egress));

                EXPECT_LT(slot, egress_shape.sender_counts[target.target_vc])
                    << "VC" << target.target_vc << " turn " << enchantum::to_string(producer) << " -> "
                    << enchantum::to_string(egress) << " is wired but lands on slot " << slot
                    << ", outside the downstream's " << egress_shape.sender_counts[target.target_vc]
                    << " sender channel(s)";

                // The two host-side spellings of the same placement must agree, or establishment
                // writes one slot while the credit bookkeeping reads another.
                const builder::RouterProducerSlots slots(to_eth(egress), egress_shape.sender_counts);
                const auto by_slots = slots.channel_for(target.target_vc, to_eth(producer));
                ASSERT_TRUE(by_slots.has_value()) << "RouterProducerSlots has no channel for a wired producer";
                EXPECT_EQ(*by_slots, slot) << "producer-slot mapping disagrees with the bijection";
            }
        }
    }
}
