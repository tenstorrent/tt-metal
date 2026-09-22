// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"
#include "tt_metal/fabric/builder/router_wiring_rules.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"

using namespace tt::tt_fabric;

// Family-level snapshots for turn_set_for_router. Primitive wiring policy is covered separately in
// test_express_connection_wiring.cpp.

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

}  // namespace

class RouterTurnSetTest : public ::testing::Test {};

// ============================================================================
// 1D: the opposite direction only, regardless of the chip's extra-port role
// ============================================================================

TEST_F(RouterTurnSetTest, Linear1D_WiresOnlyTheOpposite) {
    struct Case {
        Topology topology;
        RoutingDirection facing;
        std::optional<EdgeCapability> z_capability;
    };
    for (const auto& [topology, facing, z_capability] :
         {Case{Topology::Linear, RoutingDirection::N, std::nullopt},
          Case{Topology::Ring, RoutingDirection::E, EdgeCapability::INTRAMESH_EXPRESS}}) {
        const auto turn_set = turn_set_for_router(topology, facing, chip_with_z(z_capability), false, nullptr);
        ASSERT_EQ(turn_set[0].size(), 1);
        EXPECT_EQ(*turn_set[0][0].target_direction, get_opposite_direction(facing));
        EXPECT_EQ(turn_set[0][0].target_vc, 0);
        EXPECT_TRUE(turn_set[1].empty());
    }
}

// ============================================================================
// Non-express 2D
// ============================================================================

TEST_F(RouterTurnSetTest, NonExpress2D_WiresEveryNonSelfCardinal) {
    for (auto facing : k_all_cardinals) {
        const auto turn_set = turn_set_for_router(Topology::Mesh, facing, chip_with_z(std::nullopt), false, nullptr);

        EXPECT_EQ(target_directions(turn_set, 0), non_self_cardinals(facing))
            << "facing " << enchantum::to_string(facing);
        expect_all_targets_on_vc(turn_set, 0, 0);
        EXPECT_TRUE(turn_set[1].empty());
        EXPECT_TRUE(turn_set[2].empty());
    }
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
    const auto turn_set = turn_set_for_router(
        Topology::Mesh, RoutingDirection::E, chip_with_z(EdgeCapability::INTERMESH), false, &k_full_mesh_pass_through);

    auto expected_vc1 = non_self_cardinals(RoutingDirection::E);
    expected_vc1.insert(RoutingDirection::Z);
    EXPECT_EQ(target_directions(turn_set, 1), expected_vc1);
    expect_all_targets_on_vc(turn_set, 1, 1);
    EXPECT_EQ(turn_set[1].size(), expected_vc1.size());
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

// Express routing with an intermesh seam on a cardinal port.

namespace {

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
    // Only a Z-facing intermesh router uses the boundary template.
    const auto seam = express_turns_of(RoutingDirection::S, RoutingDirection::S);
    EXPECT_FALSE(seam[0].empty()) << "a cardinal seam still forwards on VC0";
    EXPECT_EQ(target_directions(seam, 1), target_directions(express_turns_of(RoutingDirection::S, std::nullopt), 1));
}

TEST_F(RouterTurnSetTest, ExpressCardinalSeam_XRouterKeepsTheSeamAsATarget) {
    // An intermesh egress is exempt from the intramesh X-to-Y restriction.
    for (const auto seam_facing : {RoutingDirection::N, RoutingDirection::S}) {
        for (const auto x_facing : {RoutingDirection::E, RoutingDirection::W}) {
            SCOPED_TRACE(
                "seam " + std::string(enchantum::to_string(seam_facing)) + ", facing " +
                std::string(enchantum::to_string(x_facing)));

            const auto turns = express_turns_of(x_facing, seam_facing);
            const auto expected = std::set<RoutingDirection>{get_opposite_direction(x_facing), seam_facing};

            EXPECT_EQ(target_directions(turns, 0), expected);
            EXPECT_EQ(target_directions(turns, 1), expected);

            // The ordinary Y cardinal on the same chip is still unwired, so the exemption is the
            // seam's capability rather than a hole in dimension order.
            EXPECT_EQ(
                target_directions(express_turns_of(x_facing, std::nullopt), 0),
                std::set<RoutingDirection>{get_opposite_direction(x_facing)});
        }
    }
}

TEST_F(RouterTurnSetTest, ExpressCardinalSeam_EveryWiredVc1TurnIsPlaceableDownstream) {
    // Every wired VC1 turn must map to a sender slot owned by the downstream shape.
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
                    true,
                    target.target_vc,
                    builder::routing_direction_to_eth_direction(producer),
                    builder::routing_direction_to_eth_direction(egress));

                EXPECT_LT(slot, egress_shape.sender_counts[target.target_vc])
                    << "VC" << target.target_vc << " turn " << enchantum::to_string(producer) << " -> "
                    << enchantum::to_string(egress) << " is wired but lands on slot " << slot
                    << ", outside the downstream's " << egress_shape.sender_counts[target.target_vc]
                    << " sender channel(s)";

                const builder::RouterProducerSlots slots(
                    builder::routing_direction_to_eth_direction(egress), egress_shape.sender_counts);
                const auto by_slots =
                    slots.channel_for(target.target_vc, builder::routing_direction_to_eth_direction(producer));
                ASSERT_TRUE(by_slots.has_value()) << "RouterProducerSlots has no channel for a wired producer";
                EXPECT_EQ(*by_slots, slot) << "producer-slot mapping disagrees with the bijection";
            }
        }
    }
}
