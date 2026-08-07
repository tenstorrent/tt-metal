// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <enchantum/enchantum.hpp>
#include <set>
#include <vector>

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
        for (auto role : {ZPortRole::NONE, ZPortRole::INTERMESH_BOUNDARY, ZPortRole::EXPRESS_CHORD}) {
            for (auto facing : {RoutingDirection::N, RoutingDirection::E}) {
                const auto turn_set = turn_set_for_router(
                    topology,
                    facing,
                    EdgeCapability::INTRAMESH_CARDINAL,
                    role,
                    /*express_routing_enabled=*/false,
                    nullptr);

                const auto& targets = turn_set[0];
                ASSERT_EQ(targets.size(), 1)
                    << "topology " << enchantum::to_string(topology) << " role " << enchantum::to_string(role)
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
                EdgeCapability::INTRAMESH_CARDINAL,
                ZPortRole::NONE,
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
        EdgeCapability::INTRAMESH_CARDINAL,
        ZPortRole::NONE,
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
            EdgeCapability::INTRAMESH_CARDINAL,
            ZPortRole::INTERMESH_BOUNDARY,
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
        EdgeCapability::INTRAMESH_CARDINAL,
        ZPortRole::INTERMESH_BOUNDARY,
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
            EdgeCapability::INTRAMESH_CARDINAL,
            ZPortRole::INTERMESH_BOUNDARY,
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
        EdgeCapability::INTRAMESH_CARDINAL,
        ZPortRole::NONE,
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
        EdgeCapability::INTERMESH,
        ZPortRole::INTERMESH_BOUNDARY,
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
        EdgeCapability::INTERMESH,
        ZPortRole::INTERMESH_BOUNDARY,
        /*express_routing_enabled=*/false,
        nullptr));

    EXPECT_ANY_THROW(turn_set_for_router(
        Topology::Linear,
        RoutingDirection::Z,
        EdgeCapability::INTERMESH,
        ZPortRole::INTERMESH_BOUNDARY,
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
        EdgeCapability::INTRAMESH_EXPRESS,
        ZPortRole::EXPRESS_CHORD,
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
        EdgeCapability::INTRAMESH_EXPRESS,
        ZPortRole::EXPRESS_CHORD,
        /*express_routing_enabled=*/false,
        nullptr));

    EXPECT_ANY_THROW(turn_set_for_router(
        Topology::Linear,
        RoutingDirection::Z,
        EdgeCapability::INTRAMESH_EXPRESS,
        ZPortRole::EXPRESS_CHORD,
        /*express_routing_enabled=*/true,
        nullptr));
}

TEST_F(RouterTurnSetTest, CardinalCapabilityOnZFacingIsAConfigurationError) {
    // Direction letter and capability disagree: a same-mesh Z edge is an express chord and must
    // carry INTRAMESH_EXPRESS; an ordinary cardinal-capability Z edge cannot exist. Both
    // derivations reject it -- the shape used to silently produce mesh-like counts for it.
    EXPECT_ANY_THROW(turn_set_for_router(
        Topology::Mesh,
        RoutingDirection::Z,
        EdgeCapability::INTRAMESH_CARDINAL,
        ZPortRole::NONE,
        /*express_routing_enabled=*/false,
        &k_full_mesh));
    EXPECT_ANY_THROW(router_vc_shape(
        Topology::Mesh,
        RoutingDirection::Z,
        EdgeCapability::INTRAMESH_CARDINAL,
        ZPortRole::NONE,
        /*express_routing_enabled=*/false,
        nullptr));
}

// ============================================================================
// Role/capability cross-check: the two spellings of the chip's extra port must agree
// ============================================================================

TEST_F(RouterTurnSetTest, BoundaryFacingRequiresBoundaryRole) {
    // A Z-facing intermesh edge means the chip's extra port IS the boundary. Role NONE claims the
    // port is absent and EXPRESS_CHORD claims it is a same-mesh chord -- both impossible chips,
    // and both derivations must refuse to build them.
    for (auto role : {ZPortRole::NONE, ZPortRole::EXPRESS_CHORD}) {
        SCOPED_TRACE(enchantum::to_string(role));
        EXPECT_ANY_THROW(turn_set_for_router(
            Topology::Mesh,
            RoutingDirection::Z,
            EdgeCapability::INTERMESH,
            role,
            /*express_routing_enabled=*/false,
            &k_full_mesh));
        EXPECT_ANY_THROW(router_vc_shape(
            Topology::Mesh,
            RoutingDirection::Z,
            EdgeCapability::INTERMESH,
            role,
            /*express_routing_enabled=*/false,
            nullptr));
    }
}

TEST_F(RouterTurnSetTest, ChordFacingRequiresChordRole) {
    // The mirror image: a same-mesh Z edge is this chip's express chord, so the role cannot claim
    // the port is absent or is the boundary.
    for (auto role : {ZPortRole::NONE, ZPortRole::INTERMESH_BOUNDARY}) {
        SCOPED_TRACE(enchantum::to_string(role));
        EXPECT_ANY_THROW(turn_set_for_router(
            Topology::Torus,
            RoutingDirection::Z,
            EdgeCapability::INTRAMESH_EXPRESS,
            role,
            /*express_routing_enabled=*/true,
            nullptr));
        EXPECT_ANY_THROW(router_vc_shape(
            Topology::Torus,
            RoutingDirection::Z,
            EdgeCapability::INTRAMESH_EXPRESS,
            role,
            /*express_routing_enabled=*/true,
            nullptr));
    }
}

TEST_F(RouterTurnSetTest, CardinalFacingRejectsExpressCapability) {
    // An express chord lives on the chip's extra port; a cardinal-facing router cannot carry it,
    // no matter what the chip's role says.
    EXPECT_ANY_THROW(turn_set_for_router(
        Topology::Torus,
        RoutingDirection::N,
        EdgeCapability::INTRAMESH_EXPRESS,
        ZPortRole::EXPRESS_CHORD,
        /*express_routing_enabled=*/true,
        nullptr));
    EXPECT_ANY_THROW(router_vc_shape(
        Topology::Torus,
        RoutingDirection::N,
        EdgeCapability::INTRAMESH_EXPRESS,
        ZPortRole::EXPRESS_CHORD,
        /*express_routing_enabled=*/true,
        nullptr));
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
        EdgeCapability::INTRAMESH_CARDINAL,
        ZPortRole::NONE,
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
