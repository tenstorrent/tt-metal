// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Golden expectations for the ring-family decomposition of each skip-link fixture. Machine-free:
// MeshGraph(ClusterType, path) needs no cluster, no discovery and no topology mapper, so these are
// pure input -> expected-output tables. Edit the literals; the checker is generic.
//
// Everything is in AXIS COORDINATES (rows), not chip ids: chip = row * 4 + col, and the decomposition
// is shared by all four columns (derivation rejects columns that disagree).

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>

#include "cluster.hpp"
#include "fabric_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/skip_ring_topology.hpp"

namespace tt::tt_fabric::skip_ring_tests {
namespace {

constexpr int kNoContinueSrc = SkipRingTopology::kNone;

struct Rings {
    std::vector<int> leaves;                       // ascending
    std::vector<std::pair<int, int>> anchors;      // {leaf, its anchor}
    std::vector<std::pair<int, int>> leaf_pairs;   // {leaf, its paired leaf}
    std::vector<std::vector<int>> forward_cycles;  // domain -> canonical forward row order
    int continue_src_domain = kNoContinueSrc;      // domain allowed to continue into the other
    std::vector<std::pair<int, int>> crossovers;   // ordered {continue-src row, other-family row}
};

// {src, dst, expected next row on the canonical route}
struct Hop {
    int src;
    int dst;
    int next;
};

SkipRingTopology derive(const std::string& fixture) {
    const auto path = std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
                      "tests/tt_metal/tt_fabric/custom_mesh_descriptors" / fixture;
    MeshGraph mesh_graph(tt::tt_metal::ClusterType::BLACKHOLE_GALAXY, path.string());
    auto topo = derive_skip_ring_topology(mesh_graph, MeshId{0});
    EXPECT_TRUE(topo.has_value()) << fixture << " derived no skip rings";
    return topo.value_or(SkipRingTopology{});
}

// A ControlPlane can only be built when the running world matches the descriptor's declared host
// ranks: fewer ranks cannot clear the single-host check, and one host's chips cannot back a mesh
// spanning several.
int world_size() {
    return static_cast<int>(*tt::tt_metal::MetalContext::instance().full_world_distributed_context().size());
}

// The ControlPlane-backed tests below need real channel binding, unlike everything above them.
bool cluster_available() {
    return tt::tt_metal::MetalContext::instance().rtoptions().get_mock_enabled() ||
           tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() ==
               tt::tt_metal::ClusterType::BLACKHOLE_GALAXY;
}

std::unique_ptr<ControlPlane> make_control_plane(
    const std::string& fixture, FabricConfig fabric_config, FabricReliabilityMode reliability_mode) {
    auto& metal = tt::tt_metal::MetalContext::instance();
    const auto path = std::filesystem::path(metal.rtoptions().get_root_dir()) /
                      "tests/tt_metal/tt_fabric/custom_mesh_descriptors" / fixture;
    auto control_plane = std::make_unique<ControlPlane>(
        metal.get_cluster(),
        metal.rtoptions(),
        metal.hal(),
        metal.full_world_distributed_context(),
        path.string(),
        fabric_config,
        reliability_mode);
    control_plane->configure_routing_tables_for_fabric_ethernet_channels();
    return control_plane;
}

void expect_rings(const SkipRingTopology& t, const Rings& want) {
    std::vector<int> leaves;
    for (int r = 0; r < t.axis_len; r++) {
        if (t.is_leaf(r)) {
            leaves.push_back(r);
        }
    }
    EXPECT_EQ(leaves, want.leaves) << "leaf set";
    for (const auto& [leaf, anchor] : want.anchors) {
        ASSERT_NE(t.leaf_run_of[leaf], SkipRingTopology::kNone) << "row " << leaf << " is not a leaf";
        const auto& run = t.leaf_runs[t.leaf_run_of[leaf]];
        // The end a leaf attaches to: index 0 exits before its run, the last index exits after it.
        EXPECT_EQ(t.leaf_index_of[leaf] == 0 ? run.anchor_before : run.anchor_after, anchor)
            << "anchor of leaf " << leaf;
    }
    for (const auto& [leaf, pair] : want.leaf_pairs) {
        ASSERT_NE(t.leaf_run_of[leaf], SkipRingTopology::kNone) << "row " << leaf << " is not a leaf";
        EXPECT_EQ(t.leaf_run_of[leaf], t.leaf_run_of[pair]) << leaf << " and " << pair << " share no run";
        EXPECT_EQ(std::abs(t.leaf_index_of[leaf] - t.leaf_index_of[pair]), 1)
            << leaf << " and " << pair << " are not adjacent in their run";
    }
    EXPECT_EQ(t.forward_cycle, want.forward_cycles) << "canonical forward cycles";
    EXPECT_EQ(t.continue_src_domain, want.continue_src_domain);
    EXPECT_EQ(t.crossovers, want.crossovers) << "ordered crossovers";
}

void expect_hops(const SkipRingTopology& t, const std::vector<Hop>& hops) {
    for (const auto& h : hops) {
        EXPECT_EQ(t.next_row(h.src, h.dst), h.next) << "next row on " << h.src << " -> " << h.dst;
    }
}

}  // namespace

// 8x4: LINE axis (no end wrap), one span-4 chord 2<->5 and one span-8 chord 0<->7, so the two classes
// fuse into a single spanning family over the six transit rows.
TEST(SkipRingTopologyTest, Rings8x4) {
    const Rings want{
        .leaves = {3, 4},
        .anchors = {{3, 2}, {4, 5}},
        .leaf_pairs = {{3, 4}, {4, 3}},
        .forward_cycles = {{0, 1, 2, 5, 6, 7}},
        .continue_src_domain = kNoContinueSrc,
        .crossovers = {},
    };
    const auto topo = derive("skip_links_8x4_mesh_graph_descriptor.textproto");
    expect_rings(topo, want);
    expect_hops(
        topo,
        {
            {1, 6, 2},  // exact 3-vs-3 tie: canonical forward wins (the t[4][24] == S case)
            {2, 5, 5},  // ride the span-4 chord
            {5, 3, 2},  // destination leaf reached via its anchor, never through leaf 4
            {3, 4, 4},  // paired leaves use their direct base edge
            {3, 6, 2},  // source leaf leaves via its anchor
            {0, 4, 1},  // tie again, forward
        });
}

// 16x4: LINE axis, span-4 chords 2<->5, 6<->9, 10<->13 and span-8 chords 0<->7, 8<->15. Single
// spanning family -- note rows 1 and 14 are members despite owning no chord.
TEST(SkipRingTopologyTest, Rings16x4) {
    const Rings want{
        .leaves = {3, 4, 11, 12},
        .anchors = {{3, 2}, {4, 5}, {11, 10}, {12, 13}},
        .leaf_pairs = {{3, 4}, {11, 12}},
        .forward_cycles = {{0, 1, 2, 5, 6, 9, 10, 13, 14, 15, 8, 7}},
        .continue_src_domain = kNoContinueSrc,
        .crossovers = {},
    };
    const auto topo = derive("skip_links_16x4_mesh_graph_descriptor.textproto");
    expect_rings(topo, want);
    EXPECT_FALSE(topo.is_leaf(1)) << "row 1 is a chordless ring member, not a leaf";
    EXPECT_FALSE(topo.is_leaf(14)) << "row 14 is a chordless ring member, not a leaf";
    expect_hops(
        topo,
        {
            {1, 2, 2},   // forward one hop
            {1, 14, 0},  // reverse is shorter (5 hops vs 7)
            {5, 3, 2},   // anchor, not through leaf 4
        });
}

// 24x4: LINE axis, five span-4 and three span-8 chords fused into one spanning family.
TEST(SkipRingTopologyTest, Rings24x4) {
    const Rings want{
        .leaves = {3, 4, 11, 12, 19, 20},
        .anchors = {{3, 2}, {4, 5}, {11, 10}, {12, 13}, {19, 18}, {20, 21}},
        .leaf_pairs = {{3, 4}, {11, 12}, {19, 20}},
        .forward_cycles = {{0, 1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22, 23, 16, 15, 8, 7}},
        .continue_src_domain = kNoContinueSrc,
        .crossovers = {},
    };
    const auto topo = derive("skip_links_24x4_mesh_graph_descriptor.textproto");
    expect_rings(topo, want);
    EXPECT_FALSE(topo.is_leaf(1));
    EXPECT_FALSE(topo.is_leaf(22));
}

// 32x4: RING axis, so each class closes its own family. Domains are ordered by ascending span, so
// domain 0 is the span-4 (ex4) ring and domain 1 the span-8 (ex8) ring; ex8 may continue into ex4
// and the reverse crossing is terminal.
TEST(SkipRingTopologyTest, Rings32x4) {
    const Rings want{
        .leaves = {3, 4, 11, 12, 19, 20, 27, 28},
        .anchors = {{3, 2}, {4, 5}, {11, 10}, {12, 13}, {19, 18}, {20, 21}, {27, 26}, {28, 29}},
        .leaf_pairs = {{3, 4}, {11, 12}, {19, 20}, {27, 28}},
        .forward_cycles =
            {
                {1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22, 25, 26, 29, 30},  // ex4
                {0, 7, 8, 15, 16, 23, 24, 31},                                // ex8
            },
        .continue_src_domain = 1,
        .crossovers = {{0, 1}, {7, 6}, {8, 9}, {15, 14}, {16, 17}, {23, 22}, {24, 25}, {31, 30}},
    };
    const auto topo = derive("skip_links_32x4_mesh_graph_descriptor.textproto");
    expect_rings(topo, want);
    expect_hops(
        topo,
        {
            {0, 7, 7},  // ex8 chord
            {0, 2, 1},  // ex8 -> ex4 late exit: cross at (0,1), which is right here
            {2, 0, 1},  // ex4 -> ex8 is terminal: walk ex4 to the paired landing's source side
            {3, 4, 4},  // paired leaves
        });
}

// The quad geometry with only its wide pattern declared, so each block skips six rows rather than
// two. Runs longer than a pair are entered and left by their nearer end, one row at a time -- a run
// member is never handed the anchor directly unless it is adjacent to it.
TEST(SkipRingTopologyTest, LeafRunsOfSix) {
    const auto topo = derive("skip_links_32x4_ex8_only_mesh_graph_descriptor.textproto");

    ASSERT_EQ(topo.leaf_runs.size(), 4u);
    EXPECT_EQ(topo.forward_cycle, (std::vector<std::vector<int>>{{0, 7, 8, 15, 16, 23, 24, 31}}));
    const auto& run = topo.leaf_runs[topo.leaf_run_of[1]];
    EXPECT_EQ(run.rows, (std::vector<int>{1, 2, 3, 4, 5, 6}));
    EXPECT_EQ(run.anchor_before, 0);
    EXPECT_EQ(run.anchor_after, 7);

    expect_hops(
        topo,
        {
            {1, 12, 0},  // index 0 is adjacent to anchor_before, so it exits straight to it
            {2, 12, 1},  // nearer end is before: step along the run, not a jump to the anchor
            {3, 12, 2},
            {4, 12, 5},  // past the midpoint the nearer end is after
            {5, 12, 6},
            {6, 12, 7},  // last index is adjacent to anchor_after
            {0, 2, 1},   // entered from anchor_before, then inward
            {7, 5, 6},   // entered from anchor_after, then inward
            {1, 4, 2},   // both ends inside one run: step toward the destination
        });
}

using tt::tt_fabric::fabric_router_tests::ControlPlaneFixture;

// The canonical logical route query must reconstruct the same path the table walks, and must complete
// the skip axis before turning onto X. chip = row*4 + col on the 8x4 fixture.
TEST_F(ControlPlaneFixture, CanonicalRoute8x4) {
    if (!cluster_available()) {
        GTEST_SKIP() << "needs a Blackhole Galaxy or TT_METAL_MOCK_CLUSTER_DESC_PATH";
    }
    auto control_plane = make_control_plane(
        "skip_links_8x4_mesh_graph_descriptor.textproto",
        FabricConfig::FABRIC_2D_TORUS_X,
        FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);

    const auto node = [](int chip) { return FabricNodeId{MeshId{0}, static_cast<std::uint32_t>(chip)}; };
    const auto chips = [&](const std::vector<FabricNodeId>& route) {
        std::vector<int> out;
        for (const auto& n : route) {
            out.push_back(static_cast<int>(n.chip_id));
        }
        return out;
    };

    EXPECT_TRUE(control_plane->express_routing_enabled(MeshId{0}));

    // Rows 0->5 on the single family [0,1,2,5,6,7]: an exact 3-vs-3 tie taking canonical forward, so
    // the walk is rows 0,1,2 then the span-4 chord to 5.
    EXPECT_EQ(chips(control_plane->get_canonical_intramesh_route(node(0), node(20))), (std::vector<int>{0, 4, 8, 20}));
    // Same Y route, then one X hop -- Y must complete before X.
    EXPECT_EQ(
        chips(control_plane->get_canonical_intramesh_route(node(0), node(21))), (std::vector<int>{0, 4, 8, 20, 21}));
    // Destination leaf row 3 is entered from its anchor row 2, never through paired leaf row 4.
    EXPECT_EQ(chips(control_plane->get_canonical_intramesh_route(node(20), node(12))), (std::vector<int>{20, 8, 12}));
}

// The node/direction predicates on the two-family fixture: ex8 may continue into ex4, the reverse
// crossing is terminal, and leaves carry no protected ring.
TEST_F(ControlPlaneFixture, RingPredicates32x4) {
    if (!cluster_available()) {
        GTEST_SKIP() << "needs a Blackhole Galaxy or TT_METAL_MOCK_CLUSTER_DESC_PATH";
    }
    if (world_size() != 4) {
        GTEST_SKIP() << "skip_links_32x4 declares 4 host ranks; run under tt-run with 4 ranks";
    }
    auto control_plane = make_control_plane(
        "skip_links_32x4_mesh_graph_descriptor.textproto",
        FabricConfig::FABRIC_2D_TORUS_XY,
        FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);

    using D = RoutingDirection;
    using Dim = RoutingDimension;
    const auto row = [](int r) {  // column 0
        return FabricNodeId{MeshId{0}, static_cast<std::uint32_t>(r * 4)};
    };

    EXPECT_TRUE(control_plane->express_routing_enabled(MeshId{0}));
    EXPECT_TRUE(control_plane->has_protected_ring(row(2), Dim::Y));   // ex4 member
    EXPECT_TRUE(control_plane->has_protected_ring(row(0), Dim::Y));   // ex8 member
    EXPECT_FALSE(control_plane->has_protected_ring(row(3), Dim::Y));  // leaf
    EXPECT_TRUE(control_plane->has_protected_ring(row(3), Dim::X));   // X closes at every row

    // Row 0 rides ex8 over its chord to row 7; its S neighbour row 1 is ex4, so that edge is a
    // crossover and belongs to neither ring.
    EXPECT_TRUE(control_plane->is_protected_ring_edge(row(0), D::Z));
    EXPECT_FALSE(control_plane->is_protected_ring_edge(row(0), D::S));

    // Row 2 arrives from row 1 and leaves over the ex4 chord to row 5 -- one orientation, one ring.
    EXPECT_TRUE(control_plane->are_same_directed_ring_edges(row(2), D::N, D::Z));

    // ex8 -> ex4 may continue; ex4 -> ex8 is terminal.
    EXPECT_TRUE(control_plane->continuation_allowed(row(1), D::N, D::S));
    EXPECT_FALSE(control_plane->continuation_allowed(row(0), D::S, D::Z));
}

}  // namespace tt::tt_fabric::skip_ring_tests
