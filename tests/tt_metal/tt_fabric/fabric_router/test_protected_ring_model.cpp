// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Regression for the derived protected-ring model behind the ControlPlane express predicates.
//
// The expected rings, leaves, anchors, and turn results below are the reference outputs recorded in
// GALAXY_CONTROL_PLANE_ROUTING_GENERATION_CONTRACT.md section 9 and
// GALAXY_BUILDER_ROUTING_CONFIG_CONTRACT.md section 6. They are oracles, not production inputs: the
// model must derive them from topology alone, never from a fixture-specific node array.

#include <gtest/gtest.h>

#include "tt_metal/fabric/protected_ring_model.hpp"

namespace tt::tt_fabric {
namespace {

constexpr uint32_t k_num_cols = 4;
constexpr bool k_x_ring_closed = true;

// Ordinary Y edges for a `num_rows` column, optionally closing the cardinal end wrap.
std::vector<std::pair<uint32_t, uint32_t>> cardinal_column(uint32_t num_rows, bool end_wrap) {
    std::vector<std::pair<uint32_t, uint32_t>> edges;
    for (uint32_t row = 0; row + 1 < num_rows; ++row) {
        edges.emplace_back(row, row + 1);
    }
    if (end_wrap) {
        edges.emplace_back(0, num_rows - 1);
    }
    return edges;
}

ExpressYProjection projection(
    uint32_t num_rows, bool end_wrap, std::vector<std::pair<uint32_t, uint32_t>> express) {
    ExpressYProjection p;
    p.num_rows = num_rows;
    p.ordinary_edges = cardinal_column(num_rows, end_wrap);
    p.express_edges = std::move(express);
    return p;
}

ProtectedRingModel derive(const ExpressYProjection& p) {
    return ProtectedRingModel::derive(p, k_num_cols, k_x_ring_closed);
}

// Rotate a family's forward order to begin at `first` so expectations can be written the way the
// design documents present them, independently of the canonicalization's starting point.
std::vector<uint32_t> rotated_to(const std::vector<uint32_t>& order, uint32_t first) {
    auto it = std::find(order.begin(), order.end(), first);
    EXPECT_NE(it, order.end());
    std::vector<uint32_t> out(it, order.end());
    out.insert(out.end(), order.begin(), it);
    return out;
}

// --- [8,4]: one express class, cardinal end wrap present (CP contract section 9.1) ---

TEST(ProtectedRingModelTest, Fixture8x4DerivesSingleFamilyAndLeaves) {
    const auto model = derive(projection(8, /*end_wrap=*/true, {{2, 5}}));

    EXPECT_TRUE(model.express_enabled());
    EXPECT_EQ(model.leaves(), std::set<uint32_t>({3, 4}));
    EXPECT_EQ(model.anchors().at(3), 2u);
    EXPECT_EQ(model.anchors().at(4), 5u);

    ASSERT_EQ(model.families().size(), 1u);
    EXPECT_EQ(model.families()[0].forward_order, std::vector<uint32_t>({0, 1, 2, 5, 6, 7}));
    EXPECT_EQ(model.families()[0].span, 4u);
}

TEST(ProtectedRingModelTest, Fixture8x4ExcludesTheComplementaryLeafCycle) {
    const auto model = derive(projection(8, /*end_wrap=*/true, {{2, 5}}));

    // The mathematical cycle 2->3->4->5->2 exists in the graph but is not a routing/BFC domain, so
    // no directed edge along it may be a protected ring resource.
    EXPECT_FALSE(model.is_protected_ring_edge(2, RoutingDirection::S));  // 2->3
    EXPECT_FALSE(model.is_protected_ring_edge(3, RoutingDirection::S));  // 3->4
    EXPECT_FALSE(model.is_protected_ring_edge(4, RoutingDirection::S));  // 4->5

    // Leaves are never Y transit, but still belong to the X ring.
    EXPECT_FALSE(model.has_protected_ring(3, RoutingDimension::Y));
    EXPECT_TRUE(model.has_protected_ring(3, RoutingDimension::X));
    EXPECT_TRUE(model.has_protected_ring(2, RoutingDimension::Y));
}

// --- [16,4] and [24,4]: no end wrap, so one system-spanning family (CP contract 9.2 / 9.3) ---

TEST(ProtectedRingModelTest, Fixture16x4DerivesSpanningFamily) {
    const auto model = derive(projection(16, /*end_wrap=*/false, {{2, 5}, {6, 9}, {10, 13}, {0, 7}, {8, 15}}));

    EXPECT_EQ(model.leaves(), std::set<uint32_t>({3, 4, 11, 12}));
    EXPECT_EQ(model.anchors().at(3), 2u);
    EXPECT_EQ(model.anchors().at(4), 5u);
    EXPECT_EQ(model.anchors().at(11), 10u);
    EXPECT_EQ(model.anchors().at(12), 13u);

    ASSERT_EQ(model.families().size(), 1u);
    EXPECT_EQ(
        rotated_to(model.families()[0].forward_order, 0),
        std::vector<uint32_t>({0, 1, 2, 5, 6, 9, 10, 13, 14, 15, 8, 7}));
}

TEST(ProtectedRingModelTest, Fixture16x4ForbidsOffRingTransitEdges) {
    const auto model = derive(projection(16, /*end_wrap=*/false, {{2, 5}, {6, 9}, {10, 13}, {0, 7}, {8, 15}}));

    // Both directions of 6<->7 and 8<->9 are excluded from transit. Allowing them as off-ring
    // transit would restore the {6,7,8,9} cross-ring cycle the arrangement exists to prevent.
    EXPECT_FALSE(model.is_protected_ring_edge(6, RoutingDirection::S));  // 6->7
    EXPECT_FALSE(model.is_protected_ring_edge(7, RoutingDirection::N));  // 7->6
    EXPECT_FALSE(model.is_protected_ring_edge(8, RoutingDirection::S));  // 8->9
    EXPECT_FALSE(model.is_protected_ring_edge(9, RoutingDirection::N));  // 9->8
}

TEST(ProtectedRingModelTest, Fixture24x4DerivesSpanningFamily) {
    const auto model = derive(projection(
        24, /*end_wrap=*/false, {{2, 5}, {6, 9}, {10, 13}, {14, 17}, {18, 21}, {0, 7}, {8, 15}, {16, 23}}));

    EXPECT_EQ(model.leaves(), std::set<uint32_t>({3, 4, 11, 12, 19, 20}));
    ASSERT_EQ(model.families().size(), 1u);
    EXPECT_EQ(
        rotated_to(model.families()[0].forward_order, 0),
        std::vector<uint32_t>({0, 1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22, 23, 16, 15, 8, 7}));

    // 24 rows minus 6 leaves leaves 18 transit rows.
    EXPECT_EQ(model.families()[0].forward_order.size(), 18u);
}

// --- [32,4]: end wrap present, so two families (ex4 and ex8) (CP contract section 9.4) ---

ExpressYProjection quad_galaxy_projection() {
    return projection(
        32,
        /*end_wrap=*/true,
        {// span-4 chords, including the wrap-crossing 30<->1
         {2, 5},
         {6, 9},
         {10, 13},
         {14, 17},
         {18, 21},
         {22, 25},
         {26, 29},
         {1, 30},
         // span-8 chords
         {0, 7},
         {8, 15},
         {16, 23},
         {24, 31}});
}

TEST(ProtectedRingModelTest, Fixture32x4DerivesTwoFamilies) {
    const auto model = derive(quad_galaxy_projection());

    EXPECT_EQ(model.leaves(), std::set<uint32_t>({3, 4, 11, 12, 19, 20, 27, 28}));
    ASSERT_EQ(model.families().size(), 2u);

    // Families are ordered by increasing derived span: ex4 then ex8.
    const auto& ex4 = model.families()[0];
    const auto& ex8 = model.families()[1];
    EXPECT_EQ(ex4.span, 4u);
    EXPECT_EQ(ex8.span, 8u);

    EXPECT_EQ(
        rotated_to(ex4.forward_order, 1),
        std::vector<uint32_t>({1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22, 25, 26, 29, 30}));
    EXPECT_EQ(rotated_to(ex8.forward_order, 0), std::vector<uint32_t>({0, 7, 8, 15, 16, 23, 24, 31}));

    // 16 ex4 + 8 ex8 + 8 leaves = 32 rows.
    EXPECT_EQ(ex4.forward_order.size() + ex8.forward_order.size() + model.leaves().size(), 32u);
}

// Builder contract section 6.1: the same Z output at row 2 carries both roles depending on producer.
TEST(ProtectedRingModelTest, Row2ExpressOutputIsBothTransitAndAcquisition) {
    const auto model = derive(quad_galaxy_projection());

    // The Z egress is a protected ex4 resource either way.
    EXPECT_TRUE(model.is_protected_ring_edge(2, RoutingDirection::Z));

    // N-face producer carries the ex4 hop 1->2, so continuing onto 2->5 is same-ring transit.
    EXPECT_TRUE(model.are_same_directed_ring_edges(2, RoutingDirection::N, RoutingDirection::Z));

    // S-face producer is leaf 3 over an anchor edge, so the same Z output is an acquisition. This is
    // the case the old cardinal axis-turn heuristic gets wrong: both producers share an axis pair.
    EXPECT_FALSE(model.are_same_directed_ring_edges(2, RoutingDirection::S, RoutingDirection::Z));
    EXPECT_TRUE(model.continuation_allowed(2, RoutingDirection::S, RoutingDirection::Z));
}

TEST(ProtectedRingModelTest, Row2ReverseDomainIsSymmetric) {
    const auto model = derive(quad_galaxy_projection());

    // e(2->1) is the reverse-orientation cardinal output.
    EXPECT_TRUE(model.is_protected_ring_edge(2, RoutingDirection::N));
    // Z-face transit remains in the reverse ring.
    EXPECT_TRUE(model.are_same_directed_ring_edges(2, RoutingDirection::Z, RoutingDirection::N));
    // Leaf attachment enters it.
    EXPECT_TRUE(model.continuation_allowed(2, RoutingDirection::S, RoutingDirection::N));
}

TEST(ProtectedRingModelTest, LeafRowHasNoYRingButKeepsXRing) {
    const auto model = derive(quad_galaxy_projection());

    // Builder contract section 6.2: cardinal N/S out of leaf row 3 is not an ex4 acquisition.
    EXPECT_FALSE(model.is_protected_ring_edge(3, RoutingDirection::N));
    EXPECT_FALSE(model.is_protected_ring_edge(3, RoutingDirection::S));
    EXPECT_FALSE(model.has_protected_ring(3, RoutingDimension::Y));
    // But E/W still need X-ring flow control, so BFC can never be a per-chip decision.
    EXPECT_TRUE(model.has_protected_ring(3, RoutingDimension::X));
    EXPECT_TRUE(model.is_protected_ring_edge(3, RoutingDirection::E));
}

// Builder contract section 6.3. Both turns have a NON-cyclic crossover as the ingress edge, so
// cyclic-ness alone cannot separate them -- the producing row's family span decides.
TEST(ProtectedRingModelTest, CrossFamilyContinueIsAllowedButLandOnlyIsNot) {
    const auto model = derive(quad_galaxy_projection());

    // CONTINUE: 0 (ex8) -> 1 (land) -> 2 (first ex4-forward cyclic edge).
    // The hop 0->1 arrives at row 1's N-facing port; the egress toward 2 is S.
    EXPECT_TRUE(model.continuation_allowed(1, RoutingDirection::N, RoutingDirection::S));

    // LAND_ONLY: 6 (ex4) -> 7 (land) -> 8 (first ex8-forward cyclic edge). Terminal in Y.
    EXPECT_FALSE(model.continuation_allowed(7, RoutingDirection::N, RoutingDirection::S));
}

TEST(ProtectedRingModelTest, OrientationReversalIsNeverAllowed) {
    const auto model = derive(quad_galaxy_projection());

    // Arriving on the ex4 forward hop 1->2 and leaving back toward 1 would join the two orientation
    // views of one ring, which is the dependency arc the VC0 proof assumes absent.
    EXPECT_FALSE(model.are_same_directed_ring_edges(2, RoutingDirection::N, RoutingDirection::N));
    EXPECT_FALSE(model.continuation_allowed(2, RoutingDirection::N, RoutingDirection::N));
}

// --- Fail-closed behaviour ---

TEST(ProtectedRingModelTest, MultipleExpressNeighboursPerRowIsRejected) {
    // Row 2 terminating two chords cannot be identified by a bare Z command.
    EXPECT_ANY_THROW(derive(projection(16, /*end_wrap=*/false, {{2, 5}, {2, 9}})));
}

TEST(ProtectedRingModelTest, NoExpressTopologyYieldsNoYRing) {
    const auto model = derive(projection(8, /*end_wrap=*/true, {}));

    EXPECT_FALSE(model.express_enabled());
    EXPECT_TRUE(model.families().empty());
    EXPECT_FALSE(model.has_protected_ring(0, RoutingDimension::Y));
    // The orthogonal X ring is unaffected by the absence of express topology.
    EXPECT_TRUE(model.has_protected_ring(0, RoutingDimension::X));
}

TEST(ProtectedRingModelTest, XRingAbsentWhenNotClosed) {
    const auto p = projection(8, /*end_wrap=*/true, {{2, 5}});
    const auto model = ProtectedRingModel::derive(p, k_num_cols, /*x_ring_closed=*/false);

    EXPECT_FALSE(model.has_protected_ring(0, RoutingDimension::X));
    EXPECT_FALSE(model.is_protected_ring_edge(0, RoutingDirection::E));
    // Y is still derived normally.
    EXPECT_EQ(model.families().size(), 1u);
}

}  // namespace
}  // namespace tt::tt_fabric
