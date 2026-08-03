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
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/fabric/mesh_graph.hpp>

#include "cluster.hpp"
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

// The MeshGraph is local, so the dump is taken while it is still alive.
SkipRingTopology derive(const std::string& fixture, std::string* dump = nullptr) {
    const auto path = std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
                      "tests/tt_metal/tt_fabric/custom_mesh_descriptors" / fixture;
    MeshGraph mesh_graph(tt::tt_metal::ClusterType::BLACKHOLE_GALAXY, path.string());
    auto topo = derive_skip_ring_topology(mesh_graph, MeshId{0});
    EXPECT_TRUE(topo.has_value()) << fixture << " derived no skip rings";
    if (topo.has_value() && dump != nullptr) {
        *dump = describe_skip_rings(mesh_graph, MeshId{0}, *topo);
    }
    return topo.value_or(SkipRingTopology{});
}

std::string describe_expected(const Rings& want) {
    std::ostringstream out;
    out << "expected:\n";
    for (std::size_t domain = 0; domain < want.forward_cycles.size(); domain++) {
        out << "  domain " << domain << ":";
        for (int row : want.forward_cycles[domain]) {
            out << " " << row;
        }
        out << "\n";
    }
    out << "  leaves:";
    for (int leaf : want.leaves) {
        out << " " << leaf;
    }
    out << "\n  anchors:";
    for (const auto& [leaf, anchor] : want.anchors) {
        out << " " << leaf << "->" << anchor;
    }
    out << "\n  continue_src_domain: " << want.continue_src_domain << "\n  crossovers:";
    for (const auto& [a, b] : want.crossovers) {
        out << " " << a << "->" << b;
    }
    out << "\n";
    return out.str();
}

void report(const std::string& dump, const Rings& want) { std::cout << dump << describe_expected(want) << std::flush; }

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
    std::string dump;
    const auto topo = derive("skip_links_8x4_mesh_graph_descriptor.textproto", &dump);
    report(dump, want);
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
    std::string dump;
    const auto topo = derive("skip_links_16x4_mesh_graph_descriptor.textproto", &dump);
    report(dump, want);
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
    std::string dump;
    const auto topo = derive("skip_links_24x4_mesh_graph_descriptor.textproto", &dump);
    report(dump, want);
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
    std::string dump;
    const auto topo = derive("skip_links_32x4_mesh_graph_descriptor.textproto", &dump);
    report(dump, want);
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

// Runs of ONE: every leaf is equidistant from its two anchors, so every exit and entry is a tie.
// The tie resolves to the anchor_after end, matching "exact ties take the canonical forward
// orientation" -- rows are stored anchor_before side first, so forward means toward anchor_after.
TEST(SkipRingTopologyTest, LeafRunsOfOneTieBreak) {
    std::string dump;
    const auto topo = derive("skip_links_runs_of_1_mesh_graph_descriptor.textproto", &dump);
    std::cout << dump;

    ASSERT_EQ(topo.leaf_runs.size(), 8u);
    for (int leaf : {1, 4, 7, 10, 13, 16, 19, 22}) {
        ASSERT_TRUE(topo.is_leaf(leaf)) << "row " << leaf;
        EXPECT_EQ(topo.leaf_runs[topo.leaf_run_of[leaf]].rows.size(), 1u) << "run at row " << leaf;
    }
    EXPECT_EQ(
        topo.forward_cycle,
        (std::vector<std::vector<int>>{{0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23}}));

    // Leaving row 1: anchors 0 and 2 are both one hop away, and the tie takes 2.
    EXPECT_EQ(topo.next_row(1, 10), 2);
    // Entering row 1: also from anchor 2, so the triangle 0 -> 1 -> 2 is never traversed.
    EXPECT_EQ(topo.next_row(2, 1), 1);
    // Consequence worth pinning: even standing on anchor 0, the route goes the long way round to 2.
    EXPECT_EQ(topo.next_row(0, 1), 2);
}

// Runs of THREE: the middle leaf is the tie, the two outer ones are not.
TEST(SkipRingTopologyTest, LeafRunsOfThreeTieBreak) {
    std::string dump;
    const auto topo = derive("skip_links_runs_of_3_mesh_graph_descriptor.textproto", &dump);
    std::cout << dump;

    ASSERT_EQ(topo.leaf_runs.size(), 4u);
    EXPECT_EQ(topo.forward_cycle, (std::vector<std::vector<int>>{{0, 4, 5, 9, 10, 14, 15, 19}}));
    const auto& run = topo.leaf_runs[topo.leaf_run_of[1]];
    EXPECT_EQ(run.rows, (std::vector<int>{1, 2, 3}));
    EXPECT_EQ(run.anchor_before, 0);
    EXPECT_EQ(run.anchor_after, 4);

    // Exits, toward a destination in another run. Outer leaves take their own end; the middle leaf
    // ties and steps toward anchor_after -- one row at a time, never jumping to the anchor.
    EXPECT_EQ(topo.next_row(1, 12), 0);
    EXPECT_EQ(topo.next_row(2, 12), 3);
    EXPECT_EQ(topo.next_row(3, 12), 4);

    // Entries mirror the same choice of end, then walk inward.
    EXPECT_EQ(topo.next_row(0, 1), 1);
    EXPECT_EQ(topo.next_row(4, 2), 3);
    EXPECT_EQ(topo.next_row(3, 2), 2);
}

// Exploratory: point TT_SKIP_RING_PROBE_DIR at a directory of descriptors to see what decomposition
// and routing they produce, or how they are rejected. Asserts nothing -- it reports.
TEST(SkipRingTopologyTest, ProbeDirectory) {
    const char* dir = std::getenv("TT_SKIP_RING_PROBE_DIR");
    if (dir == nullptr || !std::filesystem::is_directory(dir)) {
        GTEST_SKIP() << "set TT_SKIP_RING_PROBE_DIR to a directory of probe descriptors";
    }
    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().extension() == ".textproto") {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());

    for (const auto& path : paths) {
        std::cout << "\n===== " << path.filename().string() << "\n";
        try {
            MeshGraph mesh_graph(tt::tt_metal::ClusterType::BLACKHOLE_GALAXY, path.string());
            const auto topo = derive_skip_ring_topology(mesh_graph, MeshId{0});
            if (!topo.has_value()) {
                std::cout << "no skip links declared; base routing unchanged\n";
                continue;
            }
            std::cout << describe_skip_rings(mesh_graph, MeshId{0}, *topo);

            // Line 0 is representative: derivation already proved every line carries the same edges.
            const auto& conn = mesh_graph.get_intra_mesh_connectivity()[0];
            const auto adjacent = [&](int row_a, int row_b) {
                const auto coord = [&](int row) {
                    return topo->axis_dim == 0 ? MeshCoordinate(static_cast<std::uint32_t>(row), 0)
                                               : MeshCoordinate(0, static_cast<std::uint32_t>(row));
                };
                const auto chip_a = mesh_graph.coordinate_to_chip(MeshId{0}, coord(row_a));
                const auto chip_b = mesh_graph.coordinate_to_chip(MeshId{0}, coord(row_b));
                return conn[chip_a].find(chip_b) != conn[chip_a].end();
            };

            // Walk every ordered pair so unreachable, non-converging or non-adjacent hops surface.
            int worst_hops = 0;
            std::string worst_path;
            int failures = 0;
            for (int src = 0; src < topo->axis_len; src++) {
                for (int dst = 0; dst < topo->axis_len; dst++) {
                    if (src == dst) {
                        continue;
                    }
                    std::ostringstream trace;
                    trace << src;
                    int cur = src;
                    int hops = 0;
                    try {
                        while (cur != dst && hops <= topo->axis_len + 4) {
                            const int next = topo->next_row(cur, dst);
                            trace << "->" << next;
                            if (!adjacent(cur, next)) {
                                std::cout << "  route " << src << "->" << dst << " hops to a non-neighbour: "
                                          << trace.str() << "\n";
                                failures++;
                                break;
                            }
                            cur = next;
                            hops++;
                        }
                    } catch (const std::exception& e) {
                        std::cout << "  route " << src << "->" << dst << " threw: " << e.what() << "\n";
                        failures++;
                        continue;
                    }
                    if (cur != dst) {
                        std::cout << "  route " << src << "->" << dst << " did not converge: " << trace.str() << "\n";
                        failures++;
                    } else if (hops > worst_hops) {
                        worst_hops = hops;
                        worst_path = trace.str();
                    }
                }
            }
            std::cout << "routing: " << (topo->axis_len * (topo->axis_len - 1)) << " ordered pairs, " << failures
                      << " failed, longest " << worst_hops << " hops: " << worst_path << "\n";
        } catch (const std::exception& e) {
            std::cout << "REJECTED: " << e.what() << "\n";
        }
    }
}

}  // namespace tt::tt_fabric::skip_ring_tests
