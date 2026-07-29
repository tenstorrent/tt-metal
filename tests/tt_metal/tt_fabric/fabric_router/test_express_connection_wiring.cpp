// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Regression for express-routing connection wiring, per
// GALAXY_BUILDER_ROUTING_CONFIG_CONTRACT.md sections 3.5 and 4.4.
//
// The two properties that matter: cardinal and express outputs exist on every carrier VC (a landed
// VC1 carrier can decode a Z action and there is no VC1->VC0 crossover), and an ordinary X ingress is
// never wired back into intramesh Y (dimension order).

#include <gtest/gtest.h>

#include <algorithm>
#include <set>

#include "tt_metal/fabric/builder/router_connection_mapping.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"

namespace tt::tt_fabric {
namespace {

constexpr bool k_express = true;
constexpr bool k_no_express = false;
constexpr bool k_has_intermesh_z = true;
constexpr bool k_no_intermesh_z = false;
constexpr bool k_vc1 = true;
constexpr bool k_no_vc1 = false;
constexpr bool k_no_pass_through = false;

std::set<RoutingDirection> target_directions(const RouterConnectionMapping& mapping, uint32_t vc) {
    std::set<RoutingDirection> dirs;
    for (const auto& target : mapping.get_downstream_targets(vc, 0)) {
        if (target.type == ConnectionType::INTRA_MESH && target.target_direction.has_value()) {
            dirs.insert(*target.target_direction);
        }
    }
    return dirs;
}

RouterConnectionMapping express_mapping(
    RoutingDirection direction,
    EdgeCapability capability = EdgeCapability::INTRAMESH_CARDINAL,
    bool enable_vc1 = k_vc1,
    bool has_express_chord = true) {
    return RouterConnectionMapping::for_mesh_router(
        Topology::Torus,
        direction,
        k_no_intermesh_z,
        enable_vc1,
        k_no_pass_through,
        k_express,
        capability,
        has_express_chord);
}

// --- Legal transition set (builder contract section 4.4 wiring policy) ---

TEST(ExpressConnectionWiringTest, YIngressReachesCardinalTurnsAndExpress) {
    // A packet still in its Y phase may continue Y, turn onto either X direction, or take the chord.
    const auto mapping = express_mapping(RoutingDirection::N);
    EXPECT_EQ(
        target_directions(mapping, 0),
        std::set<RoutingDirection>({RoutingDirection::S, RoutingDirection::E, RoutingDirection::W, RoutingDirection::Z}));
}

TEST(ExpressConnectionWiringTest, ExpressIngressReachesAllFourCardinals) {
    // Arrived over the chord: continue Y cardinally or turn onto X. Z is absent because that would
    // return the packet over the link it arrived on.
    const auto mapping = express_mapping(RoutingDirection::Z);
    EXPECT_EQ(
        target_directions(mapping, 0),
        std::set<RoutingDirection>({RoutingDirection::N, RoutingDirection::S, RoutingDirection::E, RoutingDirection::W}));
}

TEST(ExpressConnectionWiringTest, IntrameshXIngressIsUnwiredFromY) {
    // Dimension order: an ordinary X ingress may only continue around the X ring. Wiring it back into
    // N/S/Z would let an X resource wait on a Y one, which the deadlock argument rules out.
    for (const auto ingress : {RoutingDirection::E, RoutingDirection::W}) {
        const auto mapping = express_mapping(ingress, EdgeCapability::INTRAMESH_CARDINAL);
        const auto dirs = target_directions(mapping, 0);
        EXPECT_EQ(dirs.size(), 1u);
        EXPECT_TRUE(dirs.contains(ingress == RoutingDirection::E ? RoutingDirection::W : RoutingDirection::E));
        EXPECT_FALSE(dirs.contains(RoutingDirection::N));
        EXPECT_FALSE(dirs.contains(RoutingDirection::S));
        EXPECT_FALSE(dirs.contains(RoutingDirection::Z));
    }
}

TEST(ExpressConnectionWiringTest, IntermeshLandingOnXPortMayBeginY) {
    // A boundary landing is a route root, not a packet mid-X-phase, so it keeps Y available even
    // though its local port is E or W.
    const auto mapping = express_mapping(RoutingDirection::E, EdgeCapability::INTERMESH);
    const auto dirs = target_directions(mapping, 0);
    EXPECT_TRUE(dirs.contains(RoutingDirection::N));
    EXPECT_TRUE(dirs.contains(RoutingDirection::S));
    EXPECT_TRUE(dirs.contains(RoutingDirection::Z));
}

TEST(ExpressConnectionWiringTest, NoRouterIsWiredBackOverItsOwnLink) {
    // A U-turn would add the one dependency arc the deadlock-freedom argument assumes absent.
    for (const auto ingress :
         {RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W, RoutingDirection::Z}) {
        for (const auto capability : {EdgeCapability::INTRAMESH_CARDINAL, EdgeCapability::INTERMESH}) {
            const auto outbound = RouterConnectionMapping::express_outbound_directions(ingress, capability);
            EXPECT_EQ(std::count(outbound.begin(), outbound.end(), ingress), 0)
                << "ingress " << static_cast<int>(ingress) << " is wired back to itself";
        }
    }
}

// --- VC symmetry ---

TEST(ExpressConnectionWiringTest, CardinalAndExpressExistOnBothCarrierVCs) {
    // Traffic that crossed a boundary stays on VC1 through every later mesh, so a decoded Z action
    // needs a VC1 express sender. A cardinal-only VC1 route family is not sufficient.
    const auto mapping = express_mapping(RoutingDirection::N, EdgeCapability::INTRAMESH_CARDINAL, k_vc1);
    EXPECT_EQ(target_directions(mapping, 0), target_directions(mapping, 1));
    EXPECT_TRUE(target_directions(mapping, 1).contains(RoutingDirection::Z));
}

TEST(ExpressConnectionWiringTest, IntrameshTargetsNeverCrossVCs) {
    // There is no VC1->VC0 landing crossover; the initial VC0->VC1 crossover belongs to the intermesh
    // boundary template, not to these shared maps.
    const auto mapping = express_mapping(RoutingDirection::N);
    for (uint32_t vc : {0u, 1u}) {
        for (const auto& target : mapping.get_downstream_targets(vc, 0)) {
            if (target.type == ConnectionType::INTRA_MESH) {
                EXPECT_EQ(target.target_vc, vc);
            }
        }
    }
}

TEST(ExpressConnectionWiringTest, NoVC1TargetsWhenVC1Disabled) {
    const auto mapping = express_mapping(RoutingDirection::N, EdgeCapability::INTRAMESH_CARDINAL, k_no_vc1);
    EXPECT_FALSE(target_directions(mapping, 0).empty());
    EXPECT_TRUE(target_directions(mapping, 1).empty());
}

TEST(ExpressConnectionWiringTest, WorkerChannelIsReservedOnVC0Only) {
    // VC0 sender channel 0 belongs to the local worker, so its forwarding targets start at 1. VC1 has
    // no worker channel and starts at 0.
    const auto mapping = express_mapping(RoutingDirection::N);
    uint32_t lowest_vc0 = ~0u;
    uint32_t lowest_vc1 = ~0u;
    for (const auto& t : mapping.get_downstream_targets(0, 0)) {
        lowest_vc0 = std::min(lowest_vc0, t.target_sender_channel);
    }
    for (const auto& t : mapping.get_downstream_targets(1, 0)) {
        lowest_vc1 = std::min(lowest_vc1, t.target_sender_channel);
    }
    EXPECT_EQ(lowest_vc0, 1u);
    EXPECT_EQ(lowest_vc1, 0u);
}

// --- Non-express wiring must be untouched ---

TEST(ExpressConnectionWiringTest, NonExpressWiringIsUnchanged) {
    // Today's 2D routing is already dimension-ordered, so its wired-but-unused X->Y arcs are
    // harmless. Removing them would change downstream counts, stream assignment, and L1 layout on
    // every existing 2D configuration, so express gates the new behaviour.
    const auto legacy = RouterConnectionMapping::for_mesh_router(
        Topology::Torus, RoutingDirection::E, k_no_intermesh_z, k_vc1, k_no_pass_through, k_no_express);
    EXPECT_EQ(
        target_directions(legacy, 0),
        std::set<RoutingDirection>({RoutingDirection::W, RoutingDirection::N, RoutingDirection::S}));
    EXPECT_FALSE(target_directions(legacy, 0).contains(RoutingDirection::Z));
}

TEST(ExpressConnectionWiringTest, IntermeshZTemplateStillAppliesUnderExpress) {
    // An intermesh Z router is a different edge from an express chord and keeps its own template.
    const auto mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Torus,
        RoutingDirection::N,
        k_has_intermesh_z,
        k_vc1,
        k_no_pass_through,
        k_express,
        EdgeCapability::INTRAMESH_CARDINAL,
        /*has_intramesh_express=*/false);

    bool has_mesh_to_z = false;
    for (const auto& target : mapping.get_downstream_targets(0, 0)) {
        if (target.type == ConnectionType::MESH_TO_Z) {
            has_mesh_to_z = true;
        }
    }
    EXPECT_TRUE(has_mesh_to_z);
}

// --- Z output existence (F3): only a chip that terminates the chord may emit a Z target ---

TEST(ExpressConnectionWiringTest, ChipWithoutExpressChordEmitsNoZTarget) {
    // Express is mesh-level, but this chip has no intramesh chord (a leaf chip, or one whose only Z
    // edge crosses a mesh boundary). A Z target would resolve to nothing -- or worse, to an
    // intermesh Z router -- so it is not emitted at all.
    for (const auto direction : {RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W}) {
        const auto mapping =
            express_mapping(direction, EdgeCapability::INTRAMESH_CARDINAL, k_vc1, /*has_express_chord=*/false);
        for (uint32_t vc : {0u, 1u}) {
            const auto dirs = target_directions(mapping, vc);
            EXPECT_FALSE(dirs.contains(RoutingDirection::Z))
                << "direction " << static_cast<int>(direction) << " VC" << vc << " must not target Z";
        }
    }
}

TEST(ExpressConnectionWiringTest, IntermeshZOnlyChipReachesZRouterOnlyThroughIntermeshTemplate) {
    // The F3 case: express mesh, and this chip's only Z edge crosses a mesh boundary. Same-mesh
    // traffic must not take an express-style Z target onto the boundary link; the intermesh
    // template is the only correct way to reach that router.
    const auto mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Torus,
        RoutingDirection::N,
        k_has_intermesh_z,
        k_vc1,
        k_no_pass_through,
        k_express,
        EdgeCapability::INTRAMESH_CARDINAL,
        /*has_intramesh_express=*/false);

    for (const auto& target : mapping.get_downstream_targets(0, 0)) {
        if (target.type == ConnectionType::INTRA_MESH) {
            EXPECT_NE(target.target_direction, RoutingDirection::Z)
                << "INTRA_MESH Z target would wire same-mesh VC0 traffic onto the intermesh link";
        }
    }

    bool has_mesh_to_z = false;
    for (const auto& target : mapping.get_downstream_targets(0, 0)) {
        if (target.type == ConnectionType::MESH_TO_Z) {
            has_mesh_to_z = true;
            EXPECT_EQ(target.target_direction, RoutingDirection::Z);
        }
    }
    EXPECT_TRUE(has_mesh_to_z);
}

TEST(ExpressConnectionWiringTest, IntermeshZOnlyChipVc1MeshToZDoesNotAliasCardinalOutputs) {
    // With the express Z target filtered, a Y-facing router's VC1 cardinal outputs occupy channels
    // 0-2, so the pass-through VC1 MESH_TO_Z target (fixed at channel 3) no longer aliases one of
    // them. This only holds because the chord-less chip drops the Z output.
    const auto mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Torus,
        RoutingDirection::N,
        k_has_intermesh_z,
        k_vc1,
        /*enable_mesh_pass_through=*/true,
        k_express,
        EdgeCapability::INTRAMESH_CARDINAL,
        /*has_intramesh_express=*/false);

    std::set<uint32_t> used_channels;
    bool has_vc1_mesh_to_z = false;
    for (const auto& target : mapping.get_downstream_targets(1, 0)) {
        EXPECT_TRUE(used_channels.insert(target.target_sender_channel).second)
            << "VC1 sender channel " << target.target_sender_channel << " is shared by two targets";
        if (target.type == ConnectionType::MESH_TO_Z) {
            has_vc1_mesh_to_z = true;
            EXPECT_EQ(target.target_sender_channel, 3u);
        }
    }
    EXPECT_TRUE(has_vc1_mesh_to_z);
    EXPECT_EQ(used_channels.size(), 4u);  // S, E, W cardinals + MESH_TO_Z
}

// --- Wired producer sets (F1): the rule the injection-flag derivation consumes ---
//
// The derivation classifies a producer's protected-ring effect only when the connection map wires
// that producer into the egress. The wired set is pinned here directly so the two cannot drift.

TEST(ExpressConnectionWiringTest, WiredProducerSetsMatchExpectedTransitions) {
    constexpr bool chord = true;

    // Y-facing egress (N/S): the opposite-Y producer and the chord producer are wired; intramesh
    // X producers are dimension-order-unwired.
    for (const auto egress : {RoutingDirection::N, RoutingDirection::S}) {
        const auto opposite = egress == RoutingDirection::N ? RoutingDirection::S : RoutingDirection::N;
        EXPECT_TRUE(RouterConnectionMapping::is_express_producer_wired(
            opposite, EdgeCapability::INTRAMESH_CARDINAL, egress, chord));
        EXPECT_TRUE(RouterConnectionMapping::is_express_producer_wired(
            RoutingDirection::Z, EdgeCapability::INTRAMESH_EXPRESS, egress, chord));
        for (const auto x : {RoutingDirection::E, RoutingDirection::W}) {
            EXPECT_FALSE(RouterConnectionMapping::is_express_producer_wired(
                x, EdgeCapability::INTRAMESH_CARDINAL, egress, chord))
                << "intramesh X must not wire into Y egress " << static_cast<int>(egress);
        }
    }

    // Express-facing egress (Z): both Y cardinals are wired; intramesh X is unwired.
    for (const auto y : {RoutingDirection::N, RoutingDirection::S}) {
        EXPECT_TRUE(RouterConnectionMapping::is_express_producer_wired(
            y, EdgeCapability::INTRAMESH_CARDINAL, RoutingDirection::Z, chord));
    }
    for (const auto x : {RoutingDirection::E, RoutingDirection::W}) {
        EXPECT_FALSE(RouterConnectionMapping::is_express_producer_wired(
            x, EdgeCapability::INTRAMESH_CARDINAL, RoutingDirection::Z, chord));
    }

    // X-facing egress (E/W): the opposite X plus every Y producer is wired, since the Y->X turn is
    // the legal dimension change.
    for (const auto egress : {RoutingDirection::E, RoutingDirection::W}) {
        const auto opposite = egress == RoutingDirection::E ? RoutingDirection::W : RoutingDirection::E;
        EXPECT_TRUE(RouterConnectionMapping::is_express_producer_wired(
            opposite, EdgeCapability::INTRAMESH_CARDINAL, egress, chord));
        for (const auto y : {RoutingDirection::N, RoutingDirection::S}) {
            EXPECT_TRUE(RouterConnectionMapping::is_express_producer_wired(
                y, EdgeCapability::INTRAMESH_CARDINAL, egress, chord));
        }
        EXPECT_TRUE(RouterConnectionMapping::is_express_producer_wired(
            RoutingDirection::Z, EdgeCapability::INTRAMESH_EXPRESS, egress, chord));
    }
}

TEST(ExpressConnectionWiringTest, IntermeshLandingProducerMayWireIntoY) {
    // A boundary landing is a route root, not a packet mid-X-phase: an INTERMESH producer wires
    // into N/S/Z even on an E or W port.
    for (const auto x : {RoutingDirection::E, RoutingDirection::W}) {
        for (const auto egress : {RoutingDirection::N, RoutingDirection::S, RoutingDirection::Z}) {
            EXPECT_TRUE(RouterConnectionMapping::is_express_producer_wired(x, EdgeCapability::INTERMESH, egress, true))
                << "landing producer " << static_cast<int>(x) << " -> " << static_cast<int>(egress);
        }
    }
}

TEST(ExpressConnectionWiringTest, NoChordNothingWiresIntoZEgress) {
    // The chord filter drops Z from every producer's outbound set, so on a chord-less chip no
    // producer is wired into a Z egress.
    for (const auto producer :
         {RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W, RoutingDirection::Z}) {
        const auto capability =
            producer == RoutingDirection::Z ? EdgeCapability::INTRAMESH_EXPRESS : EdgeCapability::INTRAMESH_CARDINAL;
        EXPECT_FALSE(
            RouterConnectionMapping::is_express_producer_wired(producer, capability, RoutingDirection::Z, false))
            << "producer " << static_cast<int>(producer);
    }

    // Y->Y without the chord still wires: leaf attachments and line continuation are real
    // transitions with real flow-control classifications.
    EXPECT_TRUE(RouterConnectionMapping::is_express_producer_wired(
        RoutingDirection::S, EdgeCapability::INTRAMESH_CARDINAL, RoutingDirection::N, false));
    EXPECT_TRUE(RouterConnectionMapping::is_express_producer_wired(
        RoutingDirection::N, EdgeCapability::INTRAMESH_CARDINAL, RoutingDirection::S, false));
}

}  // namespace
}  // namespace tt::tt_fabric
