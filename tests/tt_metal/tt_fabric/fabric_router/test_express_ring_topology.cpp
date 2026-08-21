// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Golden expectations for the ring-family decomposition of each express-link fixture. Machine-free:
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
#include "utils.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/express_ring_topology.hpp"

namespace tt::tt_fabric::express_ring_tests {
namespace {

constexpr int kNoContinueSrc = ExpressRingTopology::kNone;

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

std::string fixture_path(const std::string& fixture) {
    return (std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
            "tests/tt_metal/tt_fabric/custom_mesh_descriptors" / fixture)
        .string();
}

ExpressRingTopology derive(const std::string& fixture) {
    MeshGraph mesh_graph(tt::tt_metal::ClusterType::BLACKHOLE_GALAXY, fixture_path(fixture));
    auto topo = derive_express_ring_topology(mesh_graph, MeshId{0});
    EXPECT_TRUE(topo.has_value()) << fixture << " derived no express rings";
    return topo.value_or(ExpressRingTopology{});
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
    const std::string& fixture, FabricReliabilityMode reliability_mode, FabricConfig fabric_config) {
    auto& metal = tt::tt_metal::MetalContext::instance();
    auto control_plane = std::make_unique<ControlPlane>(
        metal.get_cluster(),
        metal.rtoptions(),
        metal.hal(),
        metal.full_world_distributed_context(),
        fixture_path(fixture),
        fabric_config,
        reliability_mode);
    control_plane->configure_routing_tables_for_fabric_ethernet_channels();
    return control_plane;
}

void expect_rings(const ExpressRingTopology& t, const Rings& want) {
    std::vector<int> leaves;
    for (int r = 0; r < t.axis_len; r++) {
        if (t.is_leaf(r)) {
            leaves.push_back(r);
        }
    }
    EXPECT_EQ(leaves, want.leaves) << "leaf set";
    for (const auto& [leaf, anchor] : want.anchors) {
        ASSERT_NE(t.leaf_run_of[leaf], ExpressRingTopology::kNone) << "row " << leaf << " is not a leaf";
        const auto& run = t.leaf_runs[t.leaf_run_of[leaf]];
        // The end a leaf attaches to: index 0 exits before its run, the last index exits after it.
        EXPECT_EQ(t.leaf_index_of[leaf] == 0 ? run.anchor_before : run.anchor_after, anchor)
            << "anchor of leaf " << leaf;
    }
    for (const auto& [leaf, pair] : want.leaf_pairs) {
        ASSERT_NE(t.leaf_run_of[leaf], ExpressRingTopology::kNone) << "row " << leaf << " is not a leaf";
        EXPECT_EQ(t.leaf_run_of[leaf], t.leaf_run_of[pair]) << leaf << " and " << pair << " share no run";
        EXPECT_EQ(std::abs(t.leaf_index_of[leaf] - t.leaf_index_of[pair]), 1)
            << leaf << " and " << pair << " are not adjacent in their run";
    }
    EXPECT_EQ(t.forward_cycle, want.forward_cycles) << "canonical forward cycles";
    EXPECT_EQ(t.continue_src_domain, want.continue_src_domain);
    EXPECT_EQ(t.crossovers, want.crossovers) << "ordered crossovers";
}

void expect_hops(const ExpressRingTopology& t, const std::vector<Hop>& hops) {
    for (const auto& h : hops) {
        EXPECT_EQ(t.next_row(h.src, h.dst), h.next) << "next row on " << h.src << " -> " << h.dst;
    }
}

}  // namespace

// 8x4: RING axis with a single span-4 chord 2<->5, its class closing one ring over the six transit
// rows through the ordinary wrap. The pattern declares wrap: LINE, so no chord straddles the boundary.
TEST(ExpressRingTopologyTest, Rings8x4) {
    const Rings want{
        .leaves = {3, 4},
        .anchors = {{3, 2}, {4, 5}},
        .leaf_pairs = {{3, 4}, {4, 3}},
        .forward_cycles = {{0, 1, 2, 5, 6, 7}},
        .continue_src_domain = kNoContinueSrc,
        .crossovers = {},
    };
    const auto topo = derive("express_links_8x4_mesh_graph_descriptor.textproto");
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
TEST(ExpressRingTopologyTest, Rings16x4) {
    const Rings want{
        .leaves = {3, 4, 11, 12},
        .anchors = {{3, 2}, {4, 5}, {11, 10}, {12, 13}},
        .leaf_pairs = {{3, 4}, {11, 12}},
        .forward_cycles = {{0, 1, 2, 5, 6, 9, 10, 13, 14, 15, 8, 7}},
        .continue_src_domain = kNoContinueSrc,
        .crossovers = {},
    };
    const auto topo = derive("express_links_16x4_mesh_graph_descriptor.textproto");
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
TEST(ExpressRingTopologyTest, Rings24x4) {
    const Rings want{
        .leaves = {3, 4, 11, 12, 19, 20},
        .anchors = {{3, 2}, {4, 5}, {11, 10}, {12, 13}, {19, 18}, {20, 21}},
        .leaf_pairs = {{3, 4}, {11, 12}, {19, 20}},
        .forward_cycles = {{0, 1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22, 23, 16, 15, 8, 7}},
        .continue_src_domain = kNoContinueSrc,
        .crossovers = {},
    };
    const auto topo = derive("express_links_24x4_mesh_graph_descriptor.textproto");
    expect_rings(topo, want);
    EXPECT_FALSE(topo.is_leaf(1));
    EXPECT_FALSE(topo.is_leaf(22));
}

// 32x4: RING axis, so each class closes its own family. Domains are ordered by ascending span, so
// domain 0 is the span-4 (ex4) ring and domain 1 the span-8 (ex8) ring; ex8 may continue into ex4
// and the reverse crossing is terminal.
TEST(ExpressRingTopologyTest, Rings32x4) {
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
    const auto topo = derive("express_links_32x4_mesh_graph_descriptor.textproto");
    expect_rings(topo, want);
    expect_hops(
        topo,
        {
            {0, 7, 7},  // ex8 chord
            {0, 2, 1},  // ex8 -> ex4 late exit: cross at (0,1), which is right here
            {2, 0, 1},  // ex4 -> ex8 is terminal: walk ex4 to the paired landing's source side
            {3, 4, 4},  // paired leaves
        });
    // Every nontrivial dependency cycle is a protected ring; the two families and their crossovers
    // introduce none outside them.
    EXPECT_TRUE(topo.cyclic_non_ring_hops().empty()) << "unprotected dependency cycle on the two-family fixture";
}

using tt::tt_fabric::fabric_router_tests::ControlPlaneFixture;
using tt::tt_fabric::fabric_router_tests::write_temp_descriptor;

// The canonical logical route query must reconstruct the same path the table walks, and must complete
// the express axis before turning onto X. chip = row*4 + col on the 8x4 fixture.
TEST_F(ControlPlaneFixture, TestExpressCanonicalRoute8x4) {
    if (!cluster_available()) {
        GTEST_SKIP() << "needs a Blackhole Galaxy or TT_METAL_MOCK_CLUSTER_DESC_PATH";
    }
    auto control_plane = make_control_plane(
        "express_links_8x4_mesh_graph_descriptor.textproto",
        FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE,
        FabricConfig::FABRIC_2D_TORUS_XY);

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
TEST_F(ControlPlaneFixture, TestExpressRingPredicates32x4) {
    if (!cluster_available()) {
        GTEST_SKIP() << "needs a Blackhole Galaxy or TT_METAL_MOCK_CLUSTER_DESC_PATH";
    }
    if (world_size() != 4) {
        GTEST_SKIP() << "express_links_32x4 declares 4 host ranks; run under tt-run with 4 ranks";
    }
    auto control_plane = make_control_plane(
        "express_links_32x4_mesh_graph_descriptor.textproto",
        FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE,
        FabricConfig::FABRIC_2D_TORUS_XY);

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

// --- Ported predicate regressions (formerly ProtectedRingModelTest) ---
//
// These pin the ControlPlane ring predicates the builder consumes, over the same fixtures the old
// builder-side model used (the descriptors' chord sets are identical). The derivation goldens live
// in the Rings* tests above; what remains here is predicate behavior per (row, direction) and the
// axis-level query semantics behind mesh_has_protected_ring_in_axis_of.

TEST(ExpressRingTopologyTest, DoubleChordPerRowIsRejected) {
    // A row terminating two chords (span 4 and span 8 both landing on row 2) cannot be identified
    // by a bare Z command; derivation must refuse it rather than guess.
    const auto path = write_temp_descriptor("express_links_16x4_double_chord.textproto", R"(
mesh_descriptors {
  name: "M0"
  arch: BLACKHOLE
  device_topology { dims: [16, 4] dim_types: [LINE, RING] }
  host_topology   { dims: [2, 1] }
  channels { count: 2 policy: RELAXED }
  express_links { dim_idx: 0  pattern { start: 2  step: 4 } }
  express_links { dim_idx: 0  pattern { start: 2  step: 8 } }
}
top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
)");
    MeshGraph mesh_graph(tt::tt_metal::ClusterType::BLACKHOLE_GALAXY, path);
    EXPECT_ANY_THROW(derive_express_ring_topology(mesh_graph, MeshId{0}));
}

// A single wide pattern on the 32-row RING axis: each block skips six interior rows, so leaves form
// runs of six rather than pairs. Exercises entering and leaving a long run one row at a time: each
// hop must land on a physically adjacent row.
TEST(ExpressRingTopologyTest, Rings32x4Ex8Only) {
    MeshGraph mesh_graph(
        tt::tt_metal::ClusterType::BLACKHOLE_GALAXY,
        fixture_path("express_links_32x4_ex8_only_mesh_graph_descriptor.textproto"));
    const auto topo = derive_express_ring_topology(mesh_graph, MeshId{0});
    ASSERT_TRUE(topo.has_value());

    // One ring over the block endpoints; four runs of six between them.
    EXPECT_EQ(topo->forward_cycle, (std::vector<std::vector<int>>{{0, 7, 8, 15, 16, 23, 24, 31}}));
    ASSERT_EQ(topo->leaf_runs.size(), 4u);
    for (const auto& run : topo->leaf_runs) {
        EXPECT_EQ(run.rows.size(), 6u) << "run bounded by " << run.anchor_before << "," << run.anchor_after;
    }

    // Every ordered pair must route to its destination in bounded hops over edges that exist.
    const auto& conn = mesh_graph.get_intra_mesh_connectivity()[0];
    const auto adjacent = [&](int a, int b) {
        const auto chip = [&](int row) {
            return static_cast<int>(mesh_graph.coordinate_to_chip(MeshId{0}, MeshCoordinate(row, 0)));
        };
        return conn[chip(a)].find(chip(b)) != conn[chip(a)].end();
    };
    for (int src = 0; src < topo->axis_len; src++) {
        for (int dst = 0; dst < topo->axis_len; dst++) {
            if (src == dst) {
                continue;
            }
            int cur = src;
            int hops = 0;
            for (; cur != dst && hops <= topo->axis_len; hops++) {
                const int next = topo->next_row(cur, dst);
                ASSERT_TRUE(adjacent(cur, next))
                    << src << "->" << dst << ": " << cur << "->" << next << " not adjacent";
                cur = next;
            }
            EXPECT_EQ(cur, dst) << "route " << src << "->" << dst << " did not converge";
        }
    }

    // No dependency cycle may form through the run's leaf links.
    EXPECT_TRUE(topo->cyclic_non_ring_hops().empty()) << "unprotected dependency cycle in the run-of-6 routes";
}

// An open X dimension yields no X ring state even alongside a live express (Y) axis: the ordinary
// ring derivation reports "not closed", so the express mesh's X ring state stays empty.
TEST(ExpressRingTopologyTest, OpenXRingReportsNoXAxisRing) {
    const auto path = write_temp_descriptor("express_links_8x4_open_x.textproto", R"(
mesh_descriptors {
  name: "M0"
  arch: BLACKHOLE
  device_topology { dims: [8, 4] dim_types: [RING, LINE] }
  host_topology   { dims: [1, 1] }
  channels { count: 2 }
  express_links { dim_idx: 0  pattern { start: 2  step: 4 } wrap: LINE}
}
top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
)");
    MeshGraph mesh_graph(tt::tt_metal::ClusterType::BLACKHOLE_GALAXY, path);
    // Against the derivations directly, not through RoutingTableGenerator: the generator only stores
    // what these return, and reaching it from a test would need a MeshGraph-only constructor that
    // exists for no other caller.
    EXPECT_TRUE(derive_express_ring_topology(mesh_graph, MeshId{0}).has_value());  // Y carries the express ring
    EXPECT_FALSE(derive_ordinary_ring_topology(mesh_graph, MeshId{0}, /*axis=*/1).has_value());  // X does not close
}

// A mesh that declares no express links derives no express ring state, which is what leaves it on the
// base dimension-order policy. Its X ring state stays underived too, but that is the generator's
// laziness (x_rings are only derived for express meshes) rather than a property of the derivation --
// X closes on this fixture, so the ordinary derivation would answer here. The non-express X
// flow-control answer comes from the topology token (see
// FabricContext::need_deadlock_avoidance_support's fallthrough), not from these predicates.
TEST(ExpressRingTopologyTest, NoExpressTopologyYieldsNoRingState) {
    const auto path = std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
                      "tests/tt_metal/tt_fabric/custom_mesh_descriptors" /
                      "bh_galaxy_single_4x4_subtorus_topology_mesh_graph_descriptor.textproto";
    MeshGraph mesh_graph(tt::tt_metal::ClusterType::BLACKHOLE_GALAXY, path.string());
    EXPECT_FALSE(derive_express_ring_topology(mesh_graph, MeshId{0}).has_value());
}

TEST_F(ControlPlaneFixture, TestExpressRow2OutputIsBothTransitAndAcquisition) {
    if (!cluster_available()) {
        GTEST_SKIP() << "needs a Blackhole Galaxy or TT_METAL_MOCK_CLUSTER_DESC_PATH";
    }
    if (world_size() != 4) {
        GTEST_SKIP() << "express_links_32x4 declares 4 host ranks; run under tt-run with 4 ranks";
    }
    auto control_plane = make_control_plane(
        "express_links_32x4_mesh_graph_descriptor.textproto",
        FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE,
        FabricConfig::FABRIC_2D_TORUS_XY);
    using D = RoutingDirection;
    const auto row = [](int r) { return FabricNodeId{MeshId{0}, static_cast<std::uint32_t>(r * 4)}; };

    // The Z egress is a protected ex4 resource either way.
    EXPECT_TRUE(control_plane->is_protected_ring_edge(row(2), D::Z));
    // N-face producer carries the ex4 hop 1->2, so continuing onto 2->5 is same-ring transit.
    EXPECT_TRUE(control_plane->are_same_directed_ring_edges(row(2), D::N, D::Z));
    // S-face producer is leaf 3 over an anchor edge, so the same Z output is an acquisition. This
    // is the case an axis-turn heuristic gets wrong: both producers share an axis pair.
    EXPECT_FALSE(control_plane->are_same_directed_ring_edges(row(2), D::S, D::Z));
    EXPECT_TRUE(control_plane->continuation_allowed(row(2), D::S, D::Z));
}

TEST_F(ControlPlaneFixture, TestExpressRow2ReverseDomainIsSymmetric) {
    if (!cluster_available()) {
        GTEST_SKIP() << "needs a Blackhole Galaxy or TT_METAL_MOCK_CLUSTER_DESC_PATH";
    }
    if (world_size() != 4) {
        GTEST_SKIP() << "express_links_32x4 declares 4 host ranks; run under tt-run with 4 ranks";
    }
    auto control_plane = make_control_plane(
        "express_links_32x4_mesh_graph_descriptor.textproto",
        FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE,
        FabricConfig::FABRIC_2D_TORUS_XY);
    using D = RoutingDirection;
    const auto row = [](int r) { return FabricNodeId{MeshId{0}, static_cast<std::uint32_t>(r * 4)}; };

    // e(2->1) is the reverse-orientation cardinal output.
    EXPECT_TRUE(control_plane->is_protected_ring_edge(row(2), D::N));
    // Z-face transit remains in the reverse ring.
    EXPECT_TRUE(control_plane->are_same_directed_ring_edges(row(2), D::Z, D::N));
    // Leaf attachment enters it.
    EXPECT_TRUE(control_plane->continuation_allowed(row(2), D::S, D::N));
}

TEST_F(ControlPlaneFixture, TestExpressLeafRowHasNoYRingButKeepsXRing) {
    if (!cluster_available()) {
        GTEST_SKIP() << "needs a Blackhole Galaxy or TT_METAL_MOCK_CLUSTER_DESC_PATH";
    }
    if (world_size() != 4) {
        GTEST_SKIP() << "express_links_32x4 declares 4 host ranks; run under tt-run with 4 ranks";
    }
    auto control_plane = make_control_plane(
        "express_links_32x4_mesh_graph_descriptor.textproto",
        FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE,
        FabricConfig::FABRIC_2D_TORUS_XY);
    using D = RoutingDirection;
    using Dim = RoutingDimension;
    const auto row = [](int r) { return FabricNodeId{MeshId{0}, static_cast<std::uint32_t>(r * 4)}; };

    // Cardinal N/S out of leaf row 3 is not an ex4 acquisition.
    EXPECT_FALSE(control_plane->is_protected_ring_edge(row(3), D::N));
    EXPECT_FALSE(control_plane->is_protected_ring_edge(row(3), D::S));
    EXPECT_FALSE(control_plane->has_protected_ring(row(3), Dim::Y));
    // But E/W still ride the X ring, so flow control can never be decided per chip.
    EXPECT_TRUE(control_plane->has_protected_ring(row(3), Dim::X));
    EXPECT_TRUE(control_plane->is_protected_ring_edge(row(3), D::E));
}

TEST_F(ControlPlaneFixture, TestExpressCrossFamilyContinueIsAllowedButLandOnlyIsNot) {
    if (!cluster_available()) {
        GTEST_SKIP() << "needs a Blackhole Galaxy or TT_METAL_MOCK_CLUSTER_DESC_PATH";
    }
    if (world_size() != 4) {
        GTEST_SKIP() << "express_links_32x4 declares 4 host ranks; run under tt-run with 4 ranks";
    }
    auto control_plane = make_control_plane(
        "express_links_32x4_mesh_graph_descriptor.textproto",
        FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE,
        FabricConfig::FABRIC_2D_TORUS_XY);
    using D = RoutingDirection;
    const auto row = [](int r) { return FabricNodeId{MeshId{0}, static_cast<std::uint32_t>(r * 4)}; };

    // CONTINUE: 0 (ex8) -> 1 (land) -> 2 (first ex4-forward cyclic edge). The hop 0->1 arrives at
    // row 1's N-facing port; the egress toward 2 is S.
    EXPECT_TRUE(control_plane->continuation_allowed(row(1), D::N, D::S));
    // LAND_ONLY: 6 (ex4) -> 7 (land) -> 8 (first ex8-forward cyclic edge). Terminal in Y.
    EXPECT_FALSE(control_plane->continuation_allowed(row(7), D::N, D::S));
}

TEST_F(ControlPlaneFixture, TestExpressOrientationReversalIsNeverAllowed) {
    if (!cluster_available()) {
        GTEST_SKIP() << "needs a Blackhole Galaxy or TT_METAL_MOCK_CLUSTER_DESC_PATH";
    }
    if (world_size() != 4) {
        GTEST_SKIP() << "express_links_32x4 declares 4 host ranks; run under tt-run with 4 ranks";
    }
    auto control_plane = make_control_plane(
        "express_links_32x4_mesh_graph_descriptor.textproto",
        FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE,
        FabricConfig::FABRIC_2D_TORUS_XY);
    using D = RoutingDirection;
    const auto row = [](int r) { return FabricNodeId{MeshId{0}, static_cast<std::uint32_t>(r * 4)}; };

    // Arriving on the ex4 forward hop 1->2 and leaving back toward 1 would join the two
    // orientation views of one ring, which is the dependency arc the proof assumes absent.
    EXPECT_FALSE(control_plane->are_same_directed_ring_edges(row(2), D::N, D::N));
    EXPECT_FALSE(control_plane->continuation_allowed(row(2), D::N, D::N));
}

// The keystone for the per-mesh axis query: distinct from the per-node query, row 3 is a leaf and
// sits on no Y ring, but the axis still carries one. Answering per node here would disable flow
// control on that chip's Y routers, and that flag also gates first-level ACK and the credit path.
TEST_F(ControlPlaneFixture, TestExpressAxisLevelQueryDoesNotElideOnLeaves) {
    if (!cluster_available()) {
        GTEST_SKIP() << "needs a Blackhole Galaxy or TT_METAL_MOCK_CLUSTER_DESC_PATH";
    }
    if (world_size() != 4) {
        GTEST_SKIP() << "express_links_32x4 declares 4 host ranks; run under tt-run with 4 ranks";
    }
    auto control_plane = make_control_plane(
        "express_links_32x4_mesh_graph_descriptor.textproto",
        FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE,
        FabricConfig::FABRIC_2D_TORUS_XY);
    using D = RoutingDirection;
    using Dim = RoutingDimension;
    const auto row = [](int r) { return FabricNodeId{MeshId{0}, static_cast<std::uint32_t>(r * 4)}; };

    EXPECT_FALSE(control_plane->has_protected_ring(row(3), Dim::Y));
    EXPECT_TRUE(control_plane->mesh_has_protected_ring_in_axis_of(MeshId{0}, D::N));
    // The X axis closes on this fixture, so it reports a ring too.
    EXPECT_TRUE(control_plane->mesh_has_protected_ring_in_axis_of(MeshId{0}, D::E));
}

// The case the topology token gets wrong: a LINE Y axis (no cardinal end wrap) whose retained
// express chords still close one spanning protected ring. Deciding from the topology enum would
// drop the bubble on a ring that needs it.
TEST_F(ControlPlaneFixture, TestExpressCarveOutWithoutEndWrapStillReportsAYRing) {
    if (!cluster_available()) {
        GTEST_SKIP() << "needs a Blackhole Galaxy or TT_METAL_MOCK_CLUSTER_DESC_PATH";
    }
    if (world_size() != 2) {
        GTEST_SKIP() << "express_links_16x4 declares 2 host ranks; run under tt-run with 2 ranks";
    }
    auto control_plane = make_control_plane(
        "express_links_16x4_mesh_graph_descriptor.textproto",
        FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE,
        FabricConfig::FABRIC_2D_TORUS_X);
    using D = RoutingDirection;
    using Dim = RoutingDimension;
    const auto row = [](int r) { return FabricNodeId{MeshId{0}, static_cast<std::uint32_t>(r * 4)}; };

    EXPECT_TRUE(control_plane->express_routing_enabled(MeshId{0}));
    EXPECT_TRUE(control_plane->mesh_has_protected_ring_in_axis_of(MeshId{0}, D::N));
    EXPECT_TRUE(control_plane->mesh_has_protected_ring_in_axis_of(MeshId{0}, D::E));  // X is RING here
    // Row 3 is a leaf amid a ringed axis: the per-node answer elides it, the per-mesh one doesn't.
    EXPECT_FALSE(control_plane->has_protected_ring(row(3), Dim::Y));
}

}  // namespace tt::tt_fabric::express_ring_tests
