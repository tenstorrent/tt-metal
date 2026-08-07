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
#include <array>
#include <enchantum/enchantum.hpp>
#include <set>

#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"
#include "tt_metal/fabric/builder/router_wiring_rules.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"

namespace tt::tt_fabric {
namespace {

constexpr bool k_express = true;
constexpr bool k_no_express = false;
constexpr bool k_vc1 = true;
constexpr bool k_no_vc1 = false;

// The VC configuration spelled as a config object -- the same spelling the shape derivation takes.
const IntermeshVCConfig k_full_mesh = IntermeshVCConfig::full_mesh();

// The full input domains of the wiring primitive, swept by the property tests below.
constexpr std::array<RoutingDirection, 5> k_all_directions = {
    RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W, RoutingDirection::Z};
constexpr std::array<EdgeCapability, 3> k_all_capabilities = {
    EdgeCapability::INTRAMESH_CARDINAL, EdgeCapability::INTRAMESH_EXPRESS, EdgeCapability::INTERMESH};
constexpr std::array<ZPortRole, 3> k_all_z_roles = {
    ZPortRole::NONE, ZPortRole::INTERMESH_BOUNDARY, ZPortRole::EXPRESS_CHORD};

std::set<RoutingDirection> target_directions(const RouterTurnSet& turn_set, uint32_t vc) {
    std::set<RoutingDirection> dirs;
    for (const auto& target : turn_set[vc]) {
        if (target.target_direction.has_value()) {
            dirs.insert(*target.target_direction);
        }
    }
    return dirs;
}

RouterTurnSet express_mapping(
    RoutingDirection direction,
    EdgeCapability capability = EdgeCapability::INTRAMESH_CARDINAL,
    bool enable_vc1 = k_vc1,
    bool has_express_chord = true) {
    return turn_set_for_router(
        Topology::Torus,
        direction,
        capability,
        (has_express_chord) ? ZPortRole::EXPRESS_CHORD : ZPortRole::NONE,
        k_express,
        enable_vc1 ? &k_full_mesh : nullptr);
}

// --- Legal transition set (builder contract section 4.4 wiring policy) ---

TEST(ExpressConnectionWiringTest, YIngressReachesCardinalTurnsAndExpress) {
    // A packet still in its Y phase may continue Y, turn onto either X direction, or take the chord.
    const auto mapping = express_mapping(RoutingDirection::N);
    EXPECT_EQ(
        target_directions(mapping, 0),
        std::set<RoutingDirection>(
            {RoutingDirection::S, RoutingDirection::E, RoutingDirection::W, RoutingDirection::Z}));
}

TEST(ExpressConnectionWiringTest, ExpressIngressReachesAllFourCardinals) {
    // Arrived over the chord: continue Y cardinally or turn onto X. Z is absent because that would
    // return the packet over the link it arrived on. The chord router's own edge carries
    // INTRAMESH_EXPRESS capability -- that is what makes it a Z-facing express router at all.
    const auto mapping = express_mapping(RoutingDirection::Z, EdgeCapability::INTRAMESH_EXPRESS);
    EXPECT_EQ(
        target_directions(mapping, 0),
        std::set<RoutingDirection>(
            {RoutingDirection::N, RoutingDirection::S, RoutingDirection::E, RoutingDirection::W}));
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
    // A U-turn would add the one dependency arc the deadlock-freedom argument assumes absent. The
    // primitive answers the self-turn before any classification, and this full-domain sweep is
    // what holds it there: the boundary arm answers vc == 1 unconditionally, so anything narrower
    // would not catch the U-turn check ever moving below the classification.
    for (const auto ingress : k_all_directions) {
        for (const auto capability : k_all_capabilities) {
            for (const auto role : k_all_z_roles) {
                for (const bool express : {false, true}) {
                    for (const uint32_t vc : {0u, 1u}) {
                        EXPECT_FALSE(wires_into(ingress, capability, ingress, role, express, vc))
                            << "ingress " << enchantum::to_string(ingress) << " (" << enchantum::to_string(capability)
                            << "), chip role " << enchantum::to_string(role) << ", express " << express << ", vc "
                            << vc;
                    }
                }
            }
        }
    }
}

TEST(ExpressConnectionWiringTest, BoundaryProducerFeedsNothingOnVC0InEitherMode) {
    // The boundary producer's feed is VC-shaped unconditionally, not just under express: its VC0
    // receiver crosses over and feeds nothing, its VC1 receiver fans out. Pinned at the primitive
    // in both modes so the header contract and the count derivation (no from-boundary slot on
    // VC0) cannot drift apart.
    for (const auto express : {false, true}) {
        for (const auto egress : {RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W}) {
            EXPECT_FALSE(wires_into(
                RoutingDirection::Z,
                EdgeCapability::INTERMESH,
                egress,
                ZPortRole::INTERMESH_BOUNDARY,
                express,
                /*vc=*/0))
                << "express=" << express << " egress " << enchantum::to_string(egress);
            EXPECT_TRUE(wires_into(
                RoutingDirection::Z,
                EdgeCapability::INTERMESH,
                egress,
                ZPortRole::INTERMESH_BOUNDARY,
                express,
                /*vc=*/1))
                << "express=" << express << " egress " << enchantum::to_string(egress);
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
        for (const auto& target : mapping[vc]) {
            // Every target -- cardinal or boundary -- stays on its source VC.
            EXPECT_EQ(target.target_vc, vc);
        }
    }
}

TEST(ExpressConnectionWiringTest, NoVC1TargetsWhenVC1Disabled) {
    const auto mapping = express_mapping(RoutingDirection::N, EdgeCapability::INTRAMESH_CARDINAL, k_no_vc1);
    EXPECT_FALSE(target_directions(mapping, 0).empty());
    EXPECT_TRUE(target_directions(mapping, 1).empty());
}

TEST(ExpressConnectionWiringTest, WorkerChannelIsReservedOnVC0Only) {
    // VC0 sender channel 0 belongs to the local worker, so a wired producer's VC0 slot is always 1
    // or above; VC1 has no worker channel, so the same producer can land at slot 0. The tight case
    // is the X-ring pair: producer W is compact index 0 on the E-facing router, giving exactly the
    // VC0 slot 1 / VC1 slot 0 boundary values.
    uint32_t lowest_vc0 = ~0u;
    uint32_t lowest_vc1 = ~0u;
    for (const auto facing : {RoutingDirection::N, RoutingDirection::E}) {
        const auto mapping = express_mapping(facing);
        const auto producer = builder::routing_direction_to_eth_direction(facing);
        for (const auto& t : mapping[0]) {
            const uint32_t slot = builder::get_downstream_sender_channel_for_vc(
                /*is_2d_routing=*/true, 0, producer, builder::routing_direction_to_eth_direction(*t.target_direction));
            EXPECT_GE(slot, 1u) << "producer " << enchantum::to_string(facing) << " aliases the VC0 worker slot on "
                                << enchantum::to_string(*t.target_direction);
            lowest_vc0 = std::min(lowest_vc0, slot);
        }
        for (const auto& t : mapping[1]) {
            lowest_vc1 = std::min(
                lowest_vc1,
                builder::get_downstream_sender_channel_for_vc(
                    /*is_2d_routing=*/true,
                    1,
                    producer,
                    builder::routing_direction_to_eth_direction(*t.target_direction)));
        }
    }
    EXPECT_EQ(lowest_vc0, 1u);
    EXPECT_EQ(lowest_vc1, 0u);
}

// --- Non-express wiring must be untouched ---

TEST(ExpressConnectionWiringTest, NonExpressWiringIsUnchanged) {
    // Today's 2D routing is already dimension-ordered, so its wired-but-unused X->Y arcs are
    // harmless. Removing them would change downstream counts, stream assignment, and L1 layout on
    // every existing 2D configuration, so express gates the new behaviour.
    const auto legacy = turn_set_for_router(
        Topology::Torus,
        RoutingDirection::E,
        EdgeCapability::INTRAMESH_CARDINAL,
        ZPortRole::NONE,
        k_no_express,
        &k_full_mesh);
    EXPECT_EQ(
        target_directions(legacy, 0),
        std::set<RoutingDirection>({RoutingDirection::W, RoutingDirection::N, RoutingDirection::S}));
    EXPECT_FALSE(target_directions(legacy, 0).contains(RoutingDirection::Z));
}

TEST(ExpressConnectionWiringTest, IntermeshZTemplateStillAppliesUnderExpress) {
    // An intermesh Z edge is a different edge from an express chord and keeps its own template:
    // the Z-direction target on this chip IS the boundary connection. The leak protection lives in
    // role-based emission: on a chord-less chip (role NONE) no Z target is emitted at all, so
    // same-mesh traffic can never leak onto the boundary link through an express-style Z target.
    const auto mapping = turn_set_for_router(
        Topology::Torus,
        RoutingDirection::N,
        EdgeCapability::INTRAMESH_CARDINAL,
        ZPortRole::INTERMESH_BOUNDARY,
        k_express,
        &k_full_mesh);

    EXPECT_TRUE(target_directions(mapping, 0).contains(RoutingDirection::Z));
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
                << "direction " << enchantum::to_string(direction) << " VC" << vc << " must not target Z";
        }
    }
}

TEST(ExpressConnectionWiringTest, IntermeshZOnlyChipVc1BoundaryTargetDoesNotAliasCardinalOutputs) {
    // With the express Z target dropped, a Y-facing router's VC1 targets are exactly the three
    // cardinals plus the pass-through boundary target -- each direction appearing exactly once, so
    // no target aliases another. This only holds because the chord-less chip drops the Z output.
    const auto pass_through = IntermeshVCConfig::full_mesh_with_pass_through();
    const auto mapping = turn_set_for_router(
        Topology::Torus,
        RoutingDirection::N,
        EdgeCapability::INTRAMESH_CARDINAL,
        ZPortRole::INTERMESH_BOUNDARY,
        k_express,
        &pass_through);

    std::set<RoutingDirection> used_directions;
    for (const auto& target : mapping[1]) {
        ASSERT_TRUE(target.target_direction.has_value());
        EXPECT_TRUE(used_directions.insert(*target.target_direction).second)
            << "direction " << enchantum::to_string(*target.target_direction) << " is shared by two VC1 targets";
    }
    EXPECT_TRUE(used_directions.contains(RoutingDirection::Z));
    EXPECT_EQ(used_directions.size(), 4u);  // S, E, W cardinals + the boundary target
}

// --- Wired producer sets (F1): the rule the injection-flag derivation consumes ---
//
// The derivation classifies a producer's protected-ring effect only when the connection map wires
// that producer into the egress. The wired set is pinned here directly so the two cannot drift.

TEST(ExpressConnectionWiringTest, WiredProducerSetsMatchExpectedTransitions) {
    // Y-facing egress (N/S): the opposite-Y producer and the chord producer are wired; intramesh
    // X producers are dimension-order-unwired.
    for (const auto egress : {RoutingDirection::N, RoutingDirection::S}) {
        const auto opposite = egress == RoutingDirection::N ? RoutingDirection::S : RoutingDirection::N;
        EXPECT_TRUE(wires_into(
            opposite,
            EdgeCapability::INTRAMESH_CARDINAL,
            egress,
            ZPortRole::EXPRESS_CHORD,
            /*express_routing_enabled=*/true,
            /*vc=*/0));
        EXPECT_TRUE(wires_into(
            RoutingDirection::Z,
            EdgeCapability::INTRAMESH_EXPRESS,
            egress,
            ZPortRole::EXPRESS_CHORD,
            /*express_routing_enabled=*/true,
            /*vc=*/0));
        for (const auto x : {RoutingDirection::E, RoutingDirection::W}) {
            EXPECT_FALSE(wires_into(
                x,
                EdgeCapability::INTRAMESH_CARDINAL,
                egress,
                ZPortRole::EXPRESS_CHORD,
                /*express_routing_enabled=*/true,
                /*vc=*/0))
                << "intramesh X must not wire into Y egress " << enchantum::to_string(egress);
        }
    }

    // Express-facing egress (Z): both Y cardinals are wired; intramesh X is unwired.
    for (const auto y : {RoutingDirection::N, RoutingDirection::S}) {
        EXPECT_TRUE(wires_into(
            y,
            EdgeCapability::INTRAMESH_CARDINAL,
            RoutingDirection::Z,
            ZPortRole::EXPRESS_CHORD,
            /*express_routing_enabled=*/true,
            /*vc=*/0));
    }
    for (const auto x : {RoutingDirection::E, RoutingDirection::W}) {
        EXPECT_FALSE(wires_into(
            x,
            EdgeCapability::INTRAMESH_CARDINAL,
            RoutingDirection::Z,
            ZPortRole::EXPRESS_CHORD,
            /*express_routing_enabled=*/true,
            /*vc=*/0));
    }

    // X-facing egress (E/W): the opposite X plus every Y producer is wired, since the Y->X turn is
    // the legal dimension change.
    for (const auto egress : {RoutingDirection::E, RoutingDirection::W}) {
        const auto opposite = egress == RoutingDirection::E ? RoutingDirection::W : RoutingDirection::E;
        EXPECT_TRUE(wires_into(
            opposite,
            EdgeCapability::INTRAMESH_CARDINAL,
            egress,
            ZPortRole::EXPRESS_CHORD,
            /*express_routing_enabled=*/true,
            /*vc=*/0));
        for (const auto y : {RoutingDirection::N, RoutingDirection::S}) {
            EXPECT_TRUE(wires_into(
                y,
                EdgeCapability::INTRAMESH_CARDINAL,
                egress,
                ZPortRole::EXPRESS_CHORD,
                /*express_routing_enabled=*/true,
                /*vc=*/0));
        }
        EXPECT_TRUE(wires_into(
            RoutingDirection::Z,
            EdgeCapability::INTRAMESH_EXPRESS,
            egress,
            ZPortRole::EXPRESS_CHORD,
            /*express_routing_enabled=*/true,
            /*vc=*/0));
    }
}

TEST(ExpressConnectionWiringTest, IntermeshLandingProducerMayWireIntoY) {
    // A boundary landing is a route root, not a packet mid-X-phase: an INTERMESH producer wires
    // into N/S/Z even on an E or W port.
    for (const auto x : {RoutingDirection::E, RoutingDirection::W}) {
        for (const auto egress : {RoutingDirection::N, RoutingDirection::S, RoutingDirection::Z}) {
            EXPECT_TRUE(wires_into(
                x,
                EdgeCapability::INTERMESH,
                egress,
                ZPortRole::EXPRESS_CHORD,
                /*express_routing_enabled=*/true,
                /*vc=*/0))
                << "landing producer " << enchantum::to_string(x) << " -> " << enchantum::to_string(egress);
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
        EXPECT_FALSE(wires_into(
            producer, capability, RoutingDirection::Z, ZPortRole::NONE, /*express_routing_enabled=*/true, /*vc=*/0))
            << "producer " << enchantum::to_string(producer);
    }

    // Y->Y without the chord still wires: leaf attachments and line continuation are real
    // transitions with real flow-control classifications.
    EXPECT_TRUE(wires_into(
        RoutingDirection::S,
        EdgeCapability::INTRAMESH_CARDINAL,
        RoutingDirection::N,
        ZPortRole::NONE,
        /*express_routing_enabled=*/true,
        /*vc=*/0));
    EXPECT_TRUE(wires_into(
        RoutingDirection::N,
        EdgeCapability::INTRAMESH_CARDINAL,
        RoutingDirection::S,
        ZPortRole::NONE,
        /*express_routing_enabled=*/true,
        /*vc=*/0));
}

// --- Sender counts are the family max over facing of wired-producer arity, not constants ---

TEST(ExpressConnectionWiringTest, ExpressSenderCountsAreFamilyMaxOverFacing) {
    // E/W-facing routers wire five VC0 producers: the worker plus every Y producer (N/S/Z) and the
    // opposite X, since the Y->X turn is legal. Dimension order leaves N/S/Z-facing routers with
    // three: worker, opposite Y, chord.
    const auto canonical = canonical_express_endpoint_capabilities();
    EXPECT_EQ(express_vc0_producer_arity(RoutingDirection::E, canonical), 5u);
    EXPECT_EQ(express_vc0_producer_arity(RoutingDirection::W, canonical), 5u);
    EXPECT_EQ(express_vc0_producer_arity(RoutingDirection::N, canonical), 3u);
    EXPECT_EQ(express_vc0_producer_arity(RoutingDirection::S, canonical), 3u);
    EXPECT_EQ(express_vc0_producer_arity(RoutingDirection::Z, canonical), 3u);

    // The uniform family counts are the max over facing: one flat index space per family, with
    // per-router wiring filling a subset. VC1 forwards the same producers minus the worker slot.
    EXPECT_EQ(express_vc0_sender_count(), 5u);
    EXPECT_EQ(express_vc1_sender_count(), 4u);
}

TEST(ExpressConnectionWiringTest, ArityRespectsPerChipCapabilities) {
    // The arity is a per-chip fact, not the family constant: on a chip whose E edge is an
    // intermesh landing, that landing producer wires into every Y egress, so a Y-facing router's
    // arity is 4, not the canonical 3. The family max is still attained by E/W facings (5).
    auto landing = canonical_express_endpoint_capabilities();
    landing.at(RoutingDirection::E) = EdgeCapability::INTERMESH;

    EXPECT_EQ(express_vc0_producer_arity(RoutingDirection::N, landing), 4u);
    EXPECT_EQ(express_vc0_producer_arity(RoutingDirection::Z, landing), 4u);
    EXPECT_EQ(express_vc0_producer_arity(RoutingDirection::W, landing), 5u);

    // A leaf chip has no chord: no Z producer exists to wire, so arities drop accordingly.
    auto leaf = canonical_express_endpoint_capabilities();
    leaf.at(RoutingDirection::Z) = std::nullopt;
    EXPECT_EQ(express_vc0_producer_arity(RoutingDirection::N, leaf), 2u);
    EXPECT_EQ(express_vc0_producer_arity(RoutingDirection::E, leaf), 4u);
}

// --- Primitive properties: each a sentence of policy, swept over the domain ---
//
// None of these record answers. Each names a rule the design relies on and checks it everywhere
// the rule can be stated, so a deliberate policy change moves exactly one property.

// The legal archetype space: which (facing, capability, chip role, mode) chips can be built at
// all. Off the Z port, a cardinal facing never carries express capability, so
// 4 facings x {INTRAMESH_CARDINAL, INTERMESH} x 3 roles = 24 triples, in either mode. On the
// Z port there are exactly two consistent triples: the chord (INTRAMESH_EXPRESS,
// EXPRESS_CHORD), buildable only with express enabled, and the boundary (INTERMESH,
// INTERMESH_BOUNDARY), in either mode. 26 triples total. This is which chips exist, not what
// the rule answers.
bool archetype_buildable(RoutingDirection facing, EdgeCapability capability, ZPortRole role, bool express) {
    if (facing == RoutingDirection::Z) {
        if (is_z_boundary_router(facing, capability)) {
            return role == ZPortRole::INTERMESH_BOUNDARY;
        }
        return capability == EdgeCapability::INTRAMESH_EXPRESS && role == ZPortRole::EXPRESS_CHORD && express;
    }
    return capability != EdgeCapability::INTRAMESH_EXPRESS;
}

bool turn_set_has_direction(const RouterTurnSet& turn_set, uint32_t vc, RoutingDirection direction) {
    return std::any_of(turn_set[vc].begin(), turn_set[vc].end(), [&](const ConnectionTarget& target) {
        return target.target_direction == direction;
    });
}

TEST(ExpressConnectionWiringTest, TurnSetMembershipMatchesThePrimitive) {
    // Every member of a router's VC0 turn set is what the primitive wires, and nothing the
    // primitive wires is missing from it. The turn set and the guard derivation read the same
    // relation, so checking containment both ways is what keeps them from diverging.
    for (const auto facing : k_all_directions) {
        for (const auto capability : k_all_capabilities) {
            if (is_z_boundary_router(facing, capability)) {
                continue;  // the boundary template, not turn-matrix-derived
            }
            for (const auto role : k_all_z_roles) {
                for (const bool express : {false, true}) {
                    if (!archetype_buildable(facing, capability, role, express)) {
                        continue;
                    }
                    for (const auto topology : {Topology::Mesh, Topology::Torus}) {
                        const auto turns =
                            turn_set_for_router(topology, facing, capability, role, express, &k_full_mesh);
                        for (const auto egress : k_all_directions) {
                            EXPECT_EQ(
                                turn_set_has_direction(turns, 0, egress),
                                wires_into(facing, capability, egress, role, express, /*vc=*/0))
                                << "facing " << enchantum::to_string(facing) << " (" << enchantum::to_string(capability)
                                << "), chip role " << enchantum::to_string(role) << ", express " << express
                                << ", topology " << enchantum::to_string(topology) << ", egress "
                                << enchantum::to_string(egress);
                        }
                    }
                }
            }
        }
    }
}

TEST(ExpressConnectionWiringTest, OnlyTheBoundaryProducerIsVcSensitive) {
    // express_vc1_sender_count() == express_vc0_sender_count() - 1 because the family max
    // commutes with the worker-slot subtraction, and that holds only while no producer except
    // the boundary answers differently per VC. Pin the premise where it can fail, not in a
    // comment on the count.
    for (const auto producer : k_all_directions) {
        for (const auto capability : k_all_capabilities) {
            if (is_z_boundary_router(producer, capability)) {
                continue;  // the one intended exception
            }
            for (const auto egress : k_all_directions) {
                for (const auto role : k_all_z_roles) {
                    for (const bool express : {false, true}) {
                        EXPECT_EQ(
                            wires_into(producer, capability, egress, role, express, /*vc=*/0),
                            wires_into(producer, capability, egress, role, express, /*vc=*/1))
                            << "producer " << enchantum::to_string(producer) << " (" << enchantum::to_string(capability)
                            << ") -> egress " << enchantum::to_string(egress) << ", chip role "
                            << enchantum::to_string(role) << ", express " << express;
                    }
                }
            }
        }
    }
}

TEST(ExpressConnectionWiringTest, ImpossiblePairsFoldIntoTheirImpliedRoles) {
    // The fold is the behaviour, not a formality: pairs no real chip has take the answers of the
    // role they structurally imply, and an IMPOSSIBLE-returns-false mapping would change these
    // answers. Pin the positive side of each fold -- the negative sides are swept elsewhere.
    // (Z, INTRAMESH_CARDINAL) folds into EXPRESS_CHORD: it wires into cardinals like a chord.
    EXPECT_TRUE(wires_into(
        RoutingDirection::Z,
        EdgeCapability::INTRAMESH_CARDINAL,
        RoutingDirection::N,
        ZPortRole::EXPRESS_CHORD,
        k_express,
        /*vc=*/0));
    // (E/W, INTRAMESH_EXPRESS) folds into X_RING_ONLY: it continues around the X ring. Its
    // unwiring from Y is swept by SameMeshXIngressNeverReentersY.
    for (const auto x : {RoutingDirection::E, RoutingDirection::W}) {
        EXPECT_TRUE(wires_into(
            x, EdgeCapability::INTRAMESH_EXPRESS, get_opposite_direction(x), ZPortRole::NONE, k_express, /*vc=*/0))
            << "producer " << enchantum::to_string(x);
    }
}

TEST(ExpressConnectionWiringTest, Vc1CarriesEveryVc0OutputExceptTheBoundaryTarget) {
    // Cardinal and express outputs ride every carrier VC: a carrier that crossed a boundary
    // stays on VC1 through later meshes and can still decode a Z action. The single exception
    // is the boundary target, which stays off VC1 unless pass-through is on -- feeding the
    // boundary's VC1 sender while it services no VC1 receiver would leave an undrained channel.
    const auto pass_through = IntermeshVCConfig::full_mesh_with_pass_through();
    const struct {
        const IntermeshVCConfig* config;
        bool pass_through;
    } vc_cases[] = {{nullptr, false}, {&k_full_mesh, false}, {&pass_through, true}};
    for (const auto facing : k_all_directions) {
        for (const auto capability : k_all_capabilities) {
            if (is_z_boundary_router(facing, capability)) {
                continue;  // the boundary template has its own VC1 shape
            }
            for (const auto role : k_all_z_roles) {
                for (const bool express : {false, true}) {
                    if (!archetype_buildable(facing, capability, role, express)) {
                        continue;
                    }
                    for (const auto& vc_case : vc_cases) {
                        const auto turns =
                            turn_set_for_router(Topology::Torus, facing, capability, role, express, vc_case.config);
                        std::set<RoutingDirection> expected = target_directions(turns, 0);
                        if (vc_case.config == nullptr) {
                            expected.clear();
                        } else if (role == ZPortRole::INTERMESH_BOUNDARY && !vc_case.pass_through) {
                            expected.erase(RoutingDirection::Z);
                        }
                        EXPECT_EQ(target_directions(turns, 1), expected)
                            << "facing " << enchantum::to_string(facing) << " (" << enchantum::to_string(capability)
                            << "), chip role " << enchantum::to_string(role) << ", express " << express;
                    }
                }
            }
        }
    }
}

TEST(ExpressConnectionWiringTest, SameMeshXIngressNeverReentersY) {
    // Dimension order: under express, a same-mesh X producer is never wired into any intramesh Y
    // egress -- N, S, or the chord. An X resource waiting on a Y one is the dependency arc the
    // deadlock argument forbids. (INTRAMESH_EXPRESS on a cardinal port is an impossible chip; it
    // folds into the same role and rides along in the sweep.)
    for (const auto producer : {RoutingDirection::E, RoutingDirection::W}) {
        for (const auto capability : {EdgeCapability::INTRAMESH_CARDINAL, EdgeCapability::INTRAMESH_EXPRESS}) {
            for (const auto egress : {RoutingDirection::N, RoutingDirection::S, RoutingDirection::Z}) {
                for (const auto role : k_all_z_roles) {
                    for (const uint32_t vc : {0u, 1u}) {
                        EXPECT_FALSE(wires_into(producer, capability, egress, role, k_express, vc))
                            << "producer " << enchantum::to_string(producer) << " (" << enchantum::to_string(capability)
                            << ") -> egress " << enchantum::to_string(egress) << ", chip role "
                            << enchantum::to_string(role) << ", vc " << vc;
                    }
                }
            }
        }
    }
}

TEST(ExpressConnectionWiringTest, LandingXIngressIsExemptFromDimensionOrder) {
    // A boundary landing is a route root, not a packet mid-X-phase: an INTERMESH producer on an X
    // port keeps every non-self egress the chip has, intramesh Y included.
    for (const auto producer : {RoutingDirection::E, RoutingDirection::W}) {
        for (const auto role : k_all_z_roles) {
            for (const uint32_t vc : {0u, 1u}) {
                for (const auto egress :
                     {RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W}) {
                    if (egress == producer) {
                        continue;
                    }
                    EXPECT_TRUE(wires_into(producer, EdgeCapability::INTERMESH, egress, role, k_express, vc))
                        << "landing producer " << enchantum::to_string(producer) << " -> egress "
                        << enchantum::to_string(egress) << ", chip role " << enchantum::to_string(role) << ", vc "
                        << vc;
                }
                EXPECT_EQ(
                    wires_into(producer, EdgeCapability::INTERMESH, RoutingDirection::Z, role, k_express, vc),
                    role != ZPortRole::NONE)
                    << "landing producer " << enchantum::to_string(producer) << " -> Z, chip role "
                    << enchantum::to_string(role) << ", vc " << vc;
            }
        }
    }
}

TEST(ExpressConnectionWiringTest, ZEgressIsWiredOnlyWhenTheChipHasThePort) {
    // A chip with no Z port resolves a Z target to nothing -- or worse, to an intermesh Z
    // router -- so no producer is ever wired into a Z egress on one, in either mode.
    for (const auto producer : k_all_directions) {
        for (const auto capability : k_all_capabilities) {
            for (const bool express : {false, true}) {
                for (const uint32_t vc : {0u, 1u}) {
                    EXPECT_FALSE(wires_into(producer, capability, RoutingDirection::Z, ZPortRole::NONE, express, vc))
                        << "producer " << enchantum::to_string(producer) << " (" << enchantum::to_string(capability)
                        << "), express " << express << ", vc " << vc;
                }
            }
        }
    }
}

TEST(ExpressConnectionWiringTest, NonExpressAdmitsEveryNonSelfCardinal) {
    // The standing decision as an executable statement: non-express wiring is byte-identical on
    // every existing 2D configuration, so non-express is producer-blind -- every non-self
    // cardinal producer wires into every cardinal egress -- and Z is a target only
    // as the boundary template's. The one VC-shaped exception (the boundary producer feeds
    // nothing on VC0) is pinned separately by BoundaryProducerFeedsNothingOnVC0InEitherMode.
    for (const auto producer : k_all_directions) {
        for (const auto capability : k_all_capabilities) {
            if (is_z_boundary_router(producer, capability)) {
                continue;
            }
            for (const auto role : k_all_z_roles) {
                for (const uint32_t vc : {0u, 1u}) {
                    for (const auto egress :
                         {RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W}) {
                        if (egress == producer) {
                            continue;
                        }
                        EXPECT_TRUE(wires_into(producer, capability, egress, role, k_no_express, vc))
                            << "producer " << enchantum::to_string(producer) << " (" << enchantum::to_string(capability)
                            << ") -> egress " << enchantum::to_string(egress) << ", chip role "
                            << enchantum::to_string(role) << ", vc " << vc;
                    }
                    if (producer != RoutingDirection::Z) {  // a Z producer never faces a Z egress
                        EXPECT_EQ(
                            wires_into(producer, capability, RoutingDirection::Z, role, k_no_express, vc),
                            role == ZPortRole::INTERMESH_BOUNDARY)
                            << "producer " << enchantum::to_string(producer) << " (" << enchantum::to_string(capability)
                            << ") -> Z, chip role " << enchantum::to_string(role) << ", vc " << vc;
                    }
                }
            }
        }
    }
}

}  // namespace
}  // namespace tt::tt_fabric
