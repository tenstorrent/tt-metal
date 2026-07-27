// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Regression for the producer effect derivation that selects each sender's flow-control guard, per
// GALAXY_BUILDER_ROUTING_CONFIG_CONTRACT.md section 4.4, checked against the worked node tables in
// its section 6.
//
// The ladder is driven from a real derived ring model rather than hand-written predicate answers, so
// a disagreement between the two would show up here.

#include <gtest/gtest.h>

#include "tt_metal/fabric/builder/fabric_edge_capability.hpp"
#include "tt_metal/fabric/protected_ring_model.hpp"

namespace tt::tt_fabric {
namespace {

constexpr auto k_cardinal = EdgeCapability::INTRAMESH_CARDINAL;
constexpr auto k_express = EdgeCapability::INTRAMESH_EXPRESS;
constexpr auto k_intermesh = EdgeCapability::INTERMESH;

// The validated four-Galaxy fixture: ex4 and ex8 chords over a 32-row column with the cardinal end
// wrap present.
ProtectedRingModel quad_galaxy_model() {
    ExpressYProjection p;
    p.num_rows = 32;
    for (uint32_t row = 0; row + 1 < 32; ++row) {
        p.ordinary_edges.emplace_back(row, row + 1);
    }
    p.ordinary_edges.emplace_back(0, 31);  // cardinal end wrap
    p.express_edges = {
        {2, 5}, {6, 9}, {10, 13}, {14, 17}, {18, 21}, {22, 25}, {26, 29}, {1, 30},  // span 4
        {0, 7}, {8, 15}, {16, 23}, {24, 31}};                                       // span 8
    return ProtectedRingModel::derive(p, /*num_cols=*/4, /*x_ring_closed=*/true);
}

ProtectedRingQueries bind(const ProtectedRingModel& model, uint32_t row) {
    ProtectedRingQueries queries;
    queries.is_protected_ring_edge = [&model, row](RoutingDirection egress) {
        return model.is_protected_ring_edge(row, egress);
    };
    queries.are_same_directed_ring_edges = [&model, row](RoutingDirection ingress, RoutingDirection egress) {
        return model.are_same_directed_ring_edges(row, ingress, egress);
    };
    queries.continuation_allowed = [&model, row](RoutingDirection ingress, RoutingDirection egress) {
        return model.continuation_allowed(row, ingress, egress);
    };
    return queries;
}

// --- Builder contract section 6.1: node Y=2, an ex4 express node ---

TEST(ProtectedDomainEffectsTest, Row2ExpressEgressCarriesBothRoles) {
    const auto model = quad_galaxy_model();
    const auto q = bind(model, 2);

    // This is the case the whole derivation exists for: one express output, two producers, two roles.
    EXPECT_EQ(classify_worker_effect(q, RoutingDirection::Z), ProtectedDomainEffect::ENTER);
    EXPECT_EQ(
        classify_producer_effect(q, RoutingDirection::N, k_cardinal, RoutingDirection::Z, k_express),
        ProtectedDomainEffect::REMAIN);
    EXPECT_EQ(
        classify_producer_effect(q, RoutingDirection::S, k_cardinal, RoutingDirection::Z, k_express),
        ProtectedDomainEffect::ENTER);

    // Only the acquisition becomes an injection channel.
    EXPECT_FALSE(is_injection_effect(ProtectedDomainEffect::REMAIN));
    EXPECT_TRUE(is_injection_effect(ProtectedDomainEffect::ENTER));
}

TEST(ProtectedDomainEffectsTest, Row2ReverseCardinalEgressIsSymmetric) {
    const auto model = quad_galaxy_model();
    const auto q = bind(model, 2);

    // e(2->1) belongs to the reverse orientation: express-face transit remains, leaf attachment enters.
    EXPECT_EQ(
        classify_producer_effect(q, RoutingDirection::Z, k_express, RoutingDirection::N, k_cardinal),
        ProtectedDomainEffect::REMAIN);
    EXPECT_EQ(
        classify_producer_effect(q, RoutingDirection::S, k_cardinal, RoutingDirection::N, k_cardinal),
        ProtectedDomainEffect::ENTER);
}

TEST(ProtectedDomainEffectsTest, Row2TurnOntoXAcquiresTheXRing) {
    const auto model = quad_galaxy_model();
    const auto q = bind(model, 2);

    // Y->X is an X acquisition regardless of which Y producer feeds it, including the express one.
    EXPECT_EQ(classify_worker_effect(q, RoutingDirection::E), ProtectedDomainEffect::ENTER);
    for (const auto ingress : {RoutingDirection::N, RoutingDirection::S, RoutingDirection::Z}) {
        const auto capability = ingress == RoutingDirection::Z ? k_express : k_cardinal;
        EXPECT_EQ(
            classify_producer_effect(q, ingress, capability, RoutingDirection::E, k_cardinal),
            ProtectedDomainEffect::ENTER);
    }
    // X transit stays transit.
    EXPECT_EQ(
        classify_producer_effect(q, RoutingDirection::W, k_cardinal, RoutingDirection::E, k_cardinal),
        ProtectedDomainEffect::REMAIN);
}

// --- Builder contract section 6.2: node Y=3, a leaf ---

TEST(ProtectedDomainEffectsTest, LeafCardinalEgressIsNotRingAcquisition) {
    const auto model = quad_galaxy_model();
    const auto q = bind(model, 3);

    // Cardinal moves to the anchor or the paired leaf are attachments, not ex4 acquisitions.
    EXPECT_EQ(classify_worker_effect(q, RoutingDirection::N), ProtectedDomainEffect::NON_RING);
    EXPECT_EQ(classify_worker_effect(q, RoutingDirection::S), ProtectedDomainEffect::NON_RING);
}

TEST(ProtectedDomainEffectsTest, LeafStillNeedsXRingGuards) {
    const auto model = quad_galaxy_model();
    const auto q = bind(model, 3);

    // A Y leaf is still on the X ring, so flow control can never be decided per chip.
    EXPECT_EQ(classify_worker_effect(q, RoutingDirection::E), ProtectedDomainEffect::ENTER);
    EXPECT_EQ(
        classify_producer_effect(q, RoutingDirection::N, k_cardinal, RoutingDirection::E, k_cardinal),
        ProtectedDomainEffect::ENTER);
    EXPECT_EQ(
        classify_producer_effect(q, RoutingDirection::W, k_cardinal, RoutingDirection::E, k_cardinal),
        ProtectedDomainEffect::REMAIN);
}

// --- Builder contract section 6.3: cross-family turns ---

TEST(ProtectedDomainEffectsTest, ContinueCrossoverEntersButLandOnlyIsNonCanonical) {
    const auto model = quad_galaxy_model();

    // CONTINUE: 0 (ex8) -> 1 (land) -> 2 (first ex4 cyclic edge). The hop 0->1 arrives on row 1's
    // N-facing port and the egress toward 2 is S.
    const auto at_row1 = bind(model, 1);
    EXPECT_EQ(
        classify_producer_effect(at_row1, RoutingDirection::N, k_cardinal, RoutingDirection::S, k_cardinal),
        ProtectedDomainEffect::ENTER);

    // LAND_ONLY: 6 (ex4) -> 7 (land) -> 8 (first ex8 cyclic edge) is terminal in Y, so the turn is
    // outside the canonical route set even though it remains physically wireable.
    const auto at_row7 = bind(model, 7);
    EXPECT_EQ(
        classify_producer_effect(at_row7, RoutingDirection::N, k_cardinal, RoutingDirection::S, k_cardinal),
        ProtectedDomainEffect::NON_CANONICAL);
    EXPECT_FALSE(is_injection_effect(ProtectedDomainEffect::NON_CANONICAL));
}

// --- Intermesh landing and dimension order ---

TEST(ProtectedDomainEffectsTest, IntermeshLandingAcquiresItsFirstProtectedEgress) {
    const auto model = quad_galaxy_model();
    const auto q = bind(model, 2);

    // A landed carrier holds no position on this mesh's rings, so its first protected output is an
    // acquisition -- even where an equivalent intramesh producer would have been transit.
    EXPECT_EQ(
        classify_producer_effect(q, RoutingDirection::N, k_intermesh, RoutingDirection::Z, k_express),
        ProtectedDomainEffect::ENTER);
    // And the landing is exempt from dimension order, so an E-facing boundary port may begin Y.
    EXPECT_EQ(
        classify_producer_effect(q, RoutingDirection::E, k_intermesh, RoutingDirection::Z, k_express),
        ProtectedDomainEffect::ENTER);
}

TEST(ProtectedDomainEffectsTest, IntrameshXIntoYIsRejectedRatherThanClassified) {
    const auto model = quad_galaxy_model();
    const auto q = bind(model, 2);

    // Connection mapping unwires this producer, so reaching the derivation means the maps and this
    // ladder disagree. It fails rather than returning a guess.
    EXPECT_ANY_THROW(
        classify_producer_effect(q, RoutingDirection::E, k_cardinal, RoutingDirection::Z, k_express));
    EXPECT_TRUE(is_static_dor_forbidden(RoutingDirection::E, k_cardinal, RoutingDirection::Z, k_express));
    EXPECT_FALSE(is_static_dor_forbidden(RoutingDirection::E, k_intermesh, RoutingDirection::Z, k_express));
}

// --- What the replaced heuristic got wrong ---

TEST(ProtectedDomainEffectsTest, OneAxisPairYieldsTwoDifferentGuards) {
    // Why no axis-based rule can work. Both producers below are a Y ingress feeding a Y egress on the
    // same node and the same output, so any rule keyed on the axis pair must return one answer for
    // both -- yet one is transit and the other an acquisition. The replaced heuristic collapsed them
    // and gave the leaf-fed acquisition the weaker guard, which is the flow-control violation this
    // derivation removes.
    const auto model = quad_galaxy_model();
    const auto q = bind(model, 2);

    const auto from_ring =
        classify_producer_effect(q, RoutingDirection::N, k_cardinal, RoutingDirection::Z, k_express);
    const auto from_leaf =
        classify_producer_effect(q, RoutingDirection::S, k_cardinal, RoutingDirection::Z, k_express);

    EXPECT_NE(from_ring, from_leaf);
    EXPECT_FALSE(is_injection_effect(from_ring));
    EXPECT_TRUE(is_injection_effect(from_leaf));
}

TEST(ProtectedDomainEffectsTest, EffectNamesAreStable) {
    EXPECT_STREQ(to_string(ProtectedDomainEffect::NON_RING), "NON_RING");
    EXPECT_STREQ(to_string(ProtectedDomainEffect::REMAIN), "REMAIN");
    EXPECT_STREQ(to_string(ProtectedDomainEffect::ENTER), "ENTER");
    EXPECT_STREQ(to_string(ProtectedDomainEffect::NON_CANONICAL), "NON_CANONICAL");
}

}  // namespace
}  // namespace tt::tt_fabric
