// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reverse-tree multicast: tree construction, the arborescence gate over every root on both axes,
// packing, the encode pass, and the per-chip L1 embed.
//
// Machine-free: MeshGraph and AxisRouteTopology need no cluster, so multi-host mesh shapes are
// covered here on a single machine.
//
// Rows are axis coordinates, not chip ids: chip = row * 4 + col.

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <ios>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/fabric/mesh_graph.hpp>

// mesh_graph.hpp only forward-declares ClusterType; the MeshGraph constructor takes it by value.
#include "cluster.hpp"
#include "hostdevcommon/fabric_common.h"
#include "llrt/rtoptions.hpp"
#include "tt_metal/fabric/axis_route_topology.hpp"
#include "tt_metal/fabric/mcast_reverse_tree.hpp"
#include "utils.hpp"

namespace tt::tt_fabric::mcast_reverse_tree_tests {
namespace {

MeshGraph load(const std::string& fixture) {
    // A local RunTimeOptions rather than MetalContext::instance(): the context eagerly builds a
    // Cluster, which throws when no chips are present.
    const tt::llrt::RunTimeOptions rtoptions;
    const auto path =
        std::filesystem::path(rtoptions.get_root_dir()) / "tests/tt_metal/tt_fabric/custom_mesh_descriptors" / fixture;
    return MeshGraph(tt::tt_metal::ClusterType::BLACKHOLE_GALAXY, path.string());
}

// union(R(root, dst)) recomputed from next_row, independently of the generator.
std::map<int, std::pair<int, RoutingDirection>> canonical_edges(
    const MeshGraph& mesh_graph, MeshId mesh_id, const AxisRouteTopology& topo, int root) {
    std::map<int, std::pair<int, RoutingDirection>> by_child;
    for (int dst = 0; dst < topo.axis_len; dst++) {
        if (dst == root) {
            continue;
        }
        int cur = root;
        for (int guard = 0; cur != dst && guard <= topo.axis_len; guard++) {
            const int next = topo.next_row(cur, dst);
            const auto direction = axis_edge_direction(mesh_graph, mesh_id, topo.axis_dim, 0, cur, next);
            EXPECT_TRUE(direction.has_value()) << "no declared edge for hop " << cur << " -> " << next;
            by_child[next] = {cur, direction.value_or(RoutingDirection::NONE)};
            cur = next;
        }
    }
    return by_child;
}

// The actions each row drives on the canonical routes to the requested targets, walked from
// next_row. This is the definition the prune has to reproduce.
std::vector<std::uint8_t> expected_actions(
    const MeshGraph& mesh_graph,
    MeshId mesh_id,
    const AxisRouteTopology& topo,
    int root,
    const std::vector<bool>& targets) {
    std::vector<std::uint8_t> want(topo.axis_len, 0);
    for (int dst = 0; dst < topo.axis_len; dst++) {
        if (dst == root || !targets[dst]) {
            continue;
        }
        int cur = root;
        for (int guard = 0; cur != dst && guard <= topo.axis_len; guard++) {
            const int next = topo.next_row(cur, dst);
            const auto direction = axis_edge_direction(mesh_graph, mesh_id, topo.axis_dim, 0, cur, next);
            want[cur] |= mcast_action_bit(direction.value_or(RoutingDirection::NONE));
            cur = next;
        }
    }
    return want;
}

// The action bit a packed 2-bit output code stands for, resolved independently of the packer.
std::uint8_t expected_action_for_code(int axis_dim, std::uint8_t code) {
    if (axis_dim == 0) {
        switch (code) {
            case Routing2DCodec::Y2_NORTH: return Routing2DCodec::ACTION_NORTH;
            case Routing2DCodec::Y2_SOUTH: return Routing2DCodec::ACTION_SOUTH;
            case Routing2DCodec::Y2_Z: return Routing2DCodec::ACTION_Z;
            default: return 0;
        }
    }
    switch (code) {
        case Routing2DCodec::X2_EAST: return Routing2DCodec::ACTION_EAST;
        case Routing2DCodec::X2_WEST: return Routing2DCodec::ACTION_WEST;
        default: return 0;
    }
}

std::string describe(const std::vector<bool>& targets) {
    std::string out;
    for (std::size_t row = 0; row < targets.size(); row++) {
        if (targets[row]) {
            out += (out.empty() ? "" : ",") + std::to_string(row);
        }
    }
    return out.empty() ? "<none>" : out;
}

void check_prune(
    const MeshGraph& mesh_graph,
    MeshId mesh_id,
    const AxisRouteTopology& topo,
    const McastReverseTree& tree,
    const std::vector<bool>& targets,
    const std::string& where) {
    const auto got = encode_mcast_axis_actions(tree, targets);
    const auto want = expected_actions(mesh_graph, mesh_id, topo, tree.root, targets);
    ASSERT_EQ(got.size(), want.size()) << where;
    for (std::size_t row = 0; row < want.size(); row++) {
        EXPECT_EQ(got[row], want[row]) << where << " targets {" << describe(targets) << "}: row " << row << " action 0x"
                                       << std::hex << static_cast<int>(got[row]) << " expected 0x"
                                       << static_cast<int>(want[row]) << std::dec;
    }
}

// Every singleton, every pair, the full set, and deterministic pseudo-random subsets. Pairs are
// where a wrong `needed` propagation shows up: a branch taken for one target and lost for the other.
void check_prune_over_target_sets(
    const MeshGraph& mesh_graph,
    MeshId mesh_id,
    const AxisRouteTopology& topo,
    const McastReverseTree& tree,
    const std::string& where) {
    const int len = topo.axis_len;

    for (int a = 0; a < len; a++) {
        std::vector<bool> targets(len, false);
        targets[a] = true;
        check_prune(mesh_graph, mesh_id, topo, tree, targets, where);
        if (::testing::Test::HasFailure()) {
            return;
        }
        for (int b = a + 1; b < len; b++) {
            targets[b] = true;
            check_prune(mesh_graph, mesh_id, topo, tree, targets, where);
            targets[b] = false;
            if (::testing::Test::HasFailure()) {
                return;
            }
        }
    }

    check_prune(mesh_graph, mesh_id, topo, tree, std::vector<bool>(len, true), where);

    std::uint32_t state = 0x2545f491u + static_cast<std::uint32_t>(tree.root);
    for (int trial = 0; trial < 16 && !::testing::Test::HasFailure(); trial++) {
        std::vector<bool> targets(len, false);
        for (int row = 0; row < len; row++) {
            state = state * 1664525u + 1013904223u;
            targets[row] = ((state >> 16) & 1u) != 0u;
        }
        check_prune(mesh_graph, mesh_id, topo, tree, targets, where);
    }
}

void check_axis(const MeshGraph& mesh_graph, const AxisRouteTopology& topo, const std::string& label) {
    const MeshId mesh_id{0};
    const auto gate = run_mcast_arborescence_gate(mesh_graph, mesh_id, topo);

    ASSERT_TRUE(gate.passed) << label << ": root " << gate.failing_root << " failed the gate -- " << gate.failure;
    ASSERT_EQ(static_cast<int>(gate.trees.size()), topo.axis_len) << label << ": expected one tree per root";

    for (const auto& tree : gate.trees) {
        const std::string where = label + " T(" + std::to_string(tree.root) + ")";

        EXPECT_EQ(static_cast<int>(tree.edges.size()), topo.axis_len - 1) << where << ": edge count";

        // Represents exactly the canonical routes: same edge set, same parent command on each.
        const auto want = canonical_edges(mesh_graph, mesh_id, topo, tree.root);
        EXPECT_EQ(tree.edges.size(), want.size()) << where << ": edge set differs from union(R(root, dst))";
        for (const auto& edge : tree.edges) {
            const auto it = want.find(edge.child);
            ASSERT_NE(it, want.end()) << where << ": edge into row " << edge.child << " is on no canonical route";
            EXPECT_EQ(edge.parent, it->second.first) << where << ": parent of row " << edge.child;
            EXPECT_EQ(edge.parent_output, it->second.second) << where << ": command into row " << edge.child;
        }

        // Indegree 1 everywhere but the root.
        std::set<int> children;
        for (const auto& edge : tree.edges) {
            EXPECT_TRUE(children.insert(edge.child).second) << where << ": row " << edge.child << " has two parents";
            EXPECT_NE(edge.child, tree.root) << where << ": the root has a parent";
        }

        // Descendants before ancestors: the ordering the worker's single reverse pass depends on.
        std::map<int, std::size_t> edge_index_by_child;
        for (std::size_t i = 0; i < tree.edges.size(); i++) {
            edge_index_by_child[tree.edges[i].child] = i;
        }
        for (std::size_t i = 0; i < tree.edges.size(); i++) {
            const auto it = edge_index_by_child.find(tree.edges[i].parent);
            if (it != edge_index_by_child.end()) {
                EXPECT_LT(i, it->second) << where << ": edge into row " << tree.edges[i].child
                                         << " must precede the edge into its parent " << tree.edges[i].parent;
            }
        }

        check_prune_over_target_sets(mesh_graph, mesh_id, topo, tree, where);
        if (::testing::Test::HasFailure()) {
            return;
        }

        // Packs, and survives the round trip with the direction intact.
        std::string pack_failure;
        const auto packed = pack_mcast_reverse_tree(tree, &pack_failure);
        ASSERT_TRUE(packed.has_value()) << where << ": packing rejected the tree -- " << pack_failure;
        ASSERT_EQ(packed->size(), tree.edges.size()) << where << ": one descriptor per edge";
        for (std::size_t i = 0; i < packed->size(); i++) {
            const std::uint16_t word = packed->at(i);
            EXPECT_EQ(mcast_tree_edge_child(word), tree.edges[i].child) << where << ": child at " << i;
            EXPECT_EQ(mcast_tree_edge_parent(word), tree.edges[i].parent) << where << ": parent at " << i;
            EXPECT_EQ(
                mcast_action_bit(tree.edges[i].parent_output),
                expected_action_for_code(topo.axis_dim, mcast_tree_edge_output(word)))
                << where << ": direction at " << i << " survived the 2-bit encoding";
            EXPECT_EQ(word >> 14, 0) << where << ": reserved bits at " << i;
        }
    }
}

void check_fixture(const std::string& fixture) {
    const auto mesh_graph = load(fixture);

    const auto express = derive_express_ring_topology(mesh_graph, MeshId{0});
    ASSERT_TRUE(express.has_value()) << fixture << ": derived no express rings";
    check_axis(mesh_graph, *express, fixture + " Y");

    // X is the ordinary four-column ring: E/W only, no chords. A multicast encodes both axes, so
    // both are gated.
    const auto ordinary_x = derive_ordinary_ring_topology(mesh_graph, MeshId{0}, 1);
    ASSERT_TRUE(ordinary_x.has_value()) << fixture << ": X dimension does not close into a ring";
    check_axis(mesh_graph, *ordinary_x, fixture + " X");
}

// Packing an ancestor before its descendant is silent on device, dropping branches from the encoded
// map rather than faulting, so the packer must reject it.
TEST(McastReverseTreeTest, PackingRejectsAncestorBeforeDescendant) {
    McastReverseTree tree;
    tree.root = 0;
    tree.axis_dim = 0;
    tree.axis_len = 3;
    // 0 -> 1 -> 2, serialized root-first, which is backwards.
    tree.edges = {
        McastTreeEdge{1, 0, RoutingDirection::S},
        McastTreeEdge{2, 1, RoutingDirection::S},
    };

    std::string failure;
    EXPECT_FALSE(pack_mcast_reverse_tree(tree, &failure).has_value());
    EXPECT_NE(failure.find("descendants before ancestors"), std::string::npos) << "actual: " << failure;

    std::reverse(tree.edges.begin(), tree.edges.end());
    EXPECT_TRUE(pack_mcast_reverse_tree(tree, &failure).has_value()) << failure;
}

// The host/device layout contract: the trees land at the offsets the loader reads, and nothing
// outside that span is touched. The sentinel fill is what makes the untouched region checkable.
void check_embed(const std::string& fixture) {
    const auto mesh_graph = load(fixture);
    const auto y_topo = derive_express_ring_topology(mesh_graph, MeshId{0});
    ASSERT_TRUE(y_topo.has_value()) << fixture << ": derived no express rings";
    const auto x_topo = derive_ordinary_ring_topology(mesh_graph, MeshId{0}, 1);
    ASSERT_TRUE(x_topo.has_value()) << fixture << ": X dimension does not close into a ring";

    const auto y_size = static_cast<std::uint32_t>(y_topo->axis_len);
    const auto x_size = static_cast<std::uint32_t>(x_topo->axis_len);
    ASSERT_TRUE(Routing2DCodec::route_table_regions_fit(y_size, x_size)) << fixture;

    const auto tree_bytes = Routing2DCodec::mcast_tree_region_bytes(y_size, x_size);
    constexpr std::uint8_t kSentinel = 0xA5;

    const std::array<std::pair<std::uint32_t, std::uint32_t>, 4> corners = {
        std::pair{0u, 0u},
        std::pair{0u, x_size - 1},
        std::pair{y_size - 1, 0u},
        std::pair{y_size - 1, x_size - 1},
    };
    for (const auto& [my_y, my_x] : corners) {
        const std::string where = fmt::format("{} chip ({},{})", fixture, my_y, my_x);
        std::vector<std::uint8_t> trees(Routing2DCodec::MCAST_TREE_CAPACITY_BYTES, kSentinel);

        std::string failure;
        ASSERT_TRUE(embed_mcast_reverse_trees(
            mesh_graph,
            MeshId{0},
            *y_topo,
            *x_topo,
            static_cast<int>(my_y),
            static_cast<int>(my_x),
            trees.data(),
            &failure))
            << where << ": " << failure;

        for (std::uint32_t i = tree_bytes; i < trees.size(); i++) {
            ASSERT_EQ(trees[i], kSentinel) << where << ": wrote outside the live tree span at byte " << i;
        }

        // The embedded words must match what the generator and packer produce standalone, for
        // this chip's own row and column and no other.
        const auto y_packed =
            pack_mcast_reverse_tree(*build_mcast_reverse_tree(mesh_graph, MeshId{0}, *y_topo, static_cast<int>(my_y)));
        const auto x_packed =
            pack_mcast_reverse_tree(*build_mcast_reverse_tree(mesh_graph, MeshId{0}, *x_topo, static_cast<int>(my_x)));
        ASSERT_TRUE(y_packed.has_value() && x_packed.has_value()) << where;

        const std::uint8_t* y_region = trees.data();
        for (std::uint32_t i = 0; i < y_packed->size(); i++) {
            EXPECT_EQ(Routing2DCodec::get_mcast_tree_edge(y_region, i), (*y_packed)[i]) << where << ": Y edge " << i;
        }
        const std::uint8_t* x_region = trees.data() + Routing2DCodec::mcast_tree_x_offset(y_size);
        for (std::uint32_t i = 0; i < x_packed->size(); i++) {
            EXPECT_EQ(Routing2DCodec::get_mcast_tree_edge(x_region, i), (*x_packed)[i]) << where << ": X edge " << i;
        }
    }
}

// The rectangle the client asked for, as row/column target sets.
std::vector<bool> target_rows(int axis_len, int root, int before_hops, int after_hops) {
    std::vector<bool> targets(axis_len, false);
    if (before_hops == 0 && after_hops == 0) {
        targets[root] = true;
        return targets;
    }
    for (int k = 1; k <= before_hops; k++) {
        targets[(root + axis_len - (k % axis_len)) % axis_len] = true;
    }
    for (int k = 1; k <= after_hops; k++) {
        targets[(root + k) % axis_len] = true;
    }
    return targets;
}

// Golden map: trace the canonical route to every target and OR the actions, then copy the X-root
// teeth and local delivery onto the target rows. The golden walks routes forward and never touches a
// reverse tree, so agreeing with the encoder means the tree does stand for the routes.
void check_encode(const std::string& fixture, bool expect_multi_output_roots) {
    const auto mesh_graph = load(fixture);
    const auto y_topo = derive_express_ring_topology(mesh_graph, MeshId{0});
    ASSERT_TRUE(y_topo.has_value()) << fixture;
    const auto x_topo = derive_ordinary_ring_topology(mesh_graph, MeshId{0}, 1);
    ASSERT_TRUE(x_topo.has_value()) << fixture;

    const int y_len = y_topo->axis_len;
    const int x_len = x_topo->axis_len;
    const auto y_size = static_cast<std::uint32_t>(y_len);
    const auto x_size = static_cast<std::uint32_t>(x_len);

    // Generic encoder coverage includes combined extents that the public source-injection API does
    // not accept as one branch, plus legal one-sided branches that can require cardinal+Z fanout.
    const std::vector<std::pair<int, int>> y_extents = {{0, 0}, {1, 0}, {0, 1}, {2, 2}, {y_len / 2, 0}, {0, y_len / 2}};
    const std::vector<std::pair<int, int>> x_extents = {{0, 0}, {1, 0}, {0, 1}, {1, x_len - 2}};

    int multi_output_roots = 0;

    for (int root_y = 0; root_y < y_len; root_y++) {
        for (int root_x = 0; root_x < x_len; root_x++) {
            std::vector<std::uint8_t> trees(Routing2DCodec::MCAST_TREE_CAPACITY_BYTES, 0);
            std::string failure;
            ASSERT_TRUE(embed_mcast_reverse_trees(
                mesh_graph, MeshId{0}, *y_topo, *x_topo, root_y, root_x, trees.data(), &failure))
                << failure;

            for (const auto& [n_hops, s_hops] : y_extents) {
                for (const auto& [w_hops, e_hops] : x_extents) {
                    const std::string where = fmt::format(
                        "{} root ({},{}) N{} S{} W{} E{}", fixture, root_y, root_x, n_hops, s_hops, w_hops, e_hops);

                    std::vector<std::uint8_t> got(y_size + x_size, 0);
                    encode_2d_mcast_maps(
                        got.data(),
                        trees.data(),
                        y_size,
                        x_size,
                        static_cast<std::uint32_t>(root_y),
                        static_cast<std::uint32_t>(root_x),
                        static_cast<std::uint32_t>(n_hops),
                        static_cast<std::uint32_t>(s_hops),
                        static_cast<std::uint32_t>(e_hops),
                        static_cast<std::uint32_t>(w_hops));

                    const auto targets_y = target_rows(y_len, root_y, n_hops, s_hops);
                    const auto targets_x = target_rows(x_len, root_x, w_hops, e_hops);
                    // The anchor column always delivers; target_rows only stands in the root when an
                    // axis has no extent at all.
                    auto want_x = expected_actions(mesh_graph, MeshId{0}, *x_topo, root_x, targets_x);
                    for (int x = 0; x < x_len; x++) {
                        if (targets_x[x] || x == root_x) {
                            want_x[x] |= Routing2DCodec::ACTION_LOCAL_DELIVER;
                        }
                    }
                    auto want_y = expected_actions(mesh_graph, MeshId{0}, *y_topo, root_y, targets_y);
                    const std::uint8_t x_root_action = want_x[root_x];
                    const std::uint8_t teeth =
                        x_root_action & (Routing2DCodec::ACTION_EAST | Routing2DCodec::ACTION_WEST);
                    const std::uint8_t deliver = x_root_action & Routing2DCodec::ACTION_LOCAL_DELIVER;
                    for (int y = 0; y < y_len; y++) {
                        if (targets_y[y]) {
                            want_y[y] |= teeth | deliver;
                        }
                    }

                    for (int y = 0; y < y_len; y++) {
                        ASSERT_EQ(got[y], want_y[y]) << where << ": route_buffer_y[" << y << "]";
                    }
                    for (int x = 0; x < x_len; x++) {
                        ASSERT_EQ(got[y_size + x], want_x[x]) << where << ": route_buffer_x[" << x << "]";
                    }

                    const std::uint8_t root_outputs = got[root_y] & ~Routing2DCodec::ACTION_LOCAL_DELIVER;
                    const bool has_vertical_trunk = n_hops != 0 || s_hops != 0;
                    const bool is_one_branch =
                        has_vertical_trunk ? ((n_hops != 0) != (s_hops != 0)) : ((e_hops != 0) != (w_hops != 0));
                    if (is_one_branch && root_outputs != 0 && (root_outputs & (root_outputs - 1)) != 0) {
                        multi_output_roots++;
                    }
                    // A root with no eth outputs is legal and means deliver locally only.
                    if (n_hops == 0 && s_hops == 0 && e_hops == 0 && w_hops == 0) {
                        EXPECT_EQ(root_outputs, 0) << where << ": local-only mcast must leave no eth output";
                        EXPECT_NE(got[root_y] & Routing2DCodec::ACTION_LOCAL_DELIVER, 0)
                            << where << ": local-only mcast must still deliver at the source";
                    }
                }
            }
        }
    }

    // A one-hop branch reaches an adjacent row or column, which no canonical route needs a chord for,
    // so its root leaves on exactly one edge and the single-connection API remains sufficient.
    for (int root_y = 0; root_y < y_len; root_y++) {
        for (int root_x = 0; root_x < x_len; root_x++) {
            std::vector<std::uint8_t> trees(Routing2DCodec::MCAST_TREE_CAPACITY_BYTES, 0);
            ASSERT_TRUE(
                embed_mcast_reverse_trees(mesh_graph, MeshId{0}, *y_topo, *x_topo, root_y, root_x, trees.data()));
            const std::vector<std::array<int, 4>> one_hop = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
            for (const auto& [n, s, e, w] : one_hop) {
                std::vector<std::uint8_t> got(y_size + x_size, 0);
                encode_2d_mcast_maps(
                    got.data(),
                    trees.data(),
                    y_size,
                    x_size,
                    static_cast<std::uint32_t>(root_y),
                    static_cast<std::uint32_t>(root_x),
                    n,
                    s,
                    e,
                    w);
                const std::uint8_t outputs = got[root_y] & Routing2DCodec::ACTION_ETH_MASK;
                EXPECT_EQ(std::popcount(static_cast<unsigned>(outputs)), 1)
                    << fixture << " root (" << root_y << "," << root_x << ") N" << n << " S" << s << " E" << e << " W"
                    << w << ": one-hop range must leave on a single edge";
            }
        }
    }

    // A legal one-direction branch can have a multi-output root on an express axis, which is why the
    // manager source-injection API submits one encoded branch through cardinal plus Z.
    if (expect_multi_output_roots) {
        EXPECT_GT(multi_output_roots, 0) << fixture
                                         << ": expected express routing to produce a multi-output one-branch root";
    }
}

// One-direction branches that route over a chord from a single-output root. A root only needs
// multi-inject when it is itself a chord tail whose branch reaches that chord's head; a root further
// from the chord leaves on one edge and lets a transit router take it.
//
// The column is not swept: with no E/W extent the X map is local delivery only and contributes no
// teeth, so the Y map depends on the root row alone.
void check_chord_ranges(const std::string& fixture, bool expect_candidates) {
    const auto mesh_graph = load(fixture);
    const auto y_topo = derive_express_ring_topology(mesh_graph, MeshId{0});
    ASSERT_TRUE(y_topo.has_value()) << fixture;
    const auto x_topo = derive_ordinary_ring_topology(mesh_graph, MeshId{0}, 1);
    ASSERT_TRUE(x_topo.has_value()) << fixture;

    const int y_len = y_topo->axis_len;
    const auto y_size = static_cast<std::uint32_t>(y_len);
    const auto x_size = static_cast<std::uint32_t>(x_topo->axis_len);

    std::vector<std::vector<std::uint8_t>> tables(y_len);
    for (int root_y = 0; root_y < y_len; root_y++) {
        tables[root_y].assign(Routing2DCodec::MCAST_TREE_CAPACITY_BYTES, 0);
        std::string failure;
        ASSERT_TRUE(embed_mcast_reverse_trees(
            mesh_graph, MeshId{0}, *y_topo, *x_topo, root_y, 0, tables[root_y].data(), &failure))
            << failure;
    }

    int candidates = 0;
    for (int n_hops = 0; n_hops < y_len; n_hops++) {
        for (int s_hops = 0; n_hops + s_hops < y_len; s_hops++) {
            if ((n_hops != 0) == (s_hops != 0)) {
                continue;
            }
            for (int root_y = 0; root_y < y_len; root_y++) {
                std::vector<std::uint8_t> got(y_size + x_size, 0);
                encode_2d_mcast_maps(
                    got.data(),
                    tables[root_y].data(),
                    y_size,
                    x_size,
                    static_cast<std::uint32_t>(root_y),
                    0,
                    static_cast<std::uint32_t>(n_hops),
                    static_cast<std::uint32_t>(s_hops),
                    0,
                    0);

                const std::uint8_t outputs = got[root_y] & Routing2DCodec::ACTION_ETH_MASK;
                if (std::popcount(static_cast<unsigned>(outputs)) != 1) {
                    continue;
                }
                for (int y = 0; y < y_len; y++) {
                    if ((got[y] & Routing2DCodec::ACTION_Z) != 0) {
                        candidates++;
                        break;
                    }
                }
            }
        }
    }

    if (expect_candidates) {
        EXPECT_GT(candidates, 0) << fixture
                                 << ": no one-direction branch routes over a chord from a single-output root";
    }
}

// 8x4 carries no expectation: its express pattern is coarser and may admit no such range.
TEST(McastReverseTreeTest, ChordRanges8x4) {
    check_chord_ranges("express_links_8x4_mesh_graph_descriptor.textproto", /*expect_candidates=*/false);
}
TEST(McastReverseTreeTest, ChordRanges32x4) {
    check_chord_ranges("express_links_32x4_mesh_graph_descriptor.textproto", /*expect_candidates=*/true);
}

// Generic encoder stress case: every chip roots combined extents covering the whole mesh. This is
// not one legal source-injection branch; it verifies only the action-map structure and output bound.
//
// The root action is route_buffer_y[root_y] alone, not that OR'd with route_buffer_x[root_x]. The
// X-root E/W teeth are copied onto the target rows, and a range with a Y extent excludes the source
// row, so the source never picks up the teeth and its outputs are a subset of N/S/Z. (An X-only
// range is the other case: the source row is itself a target, so the action is a subset of E/W.)
void check_full_extent_roots(const std::string& fixture) {
    const auto mesh_graph = load(fixture);
    const auto y_topo = derive_express_ring_topology(mesh_graph, MeshId{0});
    ASSERT_TRUE(y_topo.has_value()) << fixture;
    const auto x_topo = derive_ordinary_ring_topology(mesh_graph, MeshId{0}, 1);
    ASSERT_TRUE(x_topo.has_value()) << fixture;

    const int y_len = y_topo->axis_len;
    const int x_len = x_topo->axis_len;
    const auto y_size = static_cast<std::uint32_t>(y_len);
    const auto x_size = static_cast<std::uint32_t>(x_len);

    // The whole mesh minus the source: N and S together cover every other row, likewise E and W.
    const int n_hops = y_len / 2;
    const int s_hops = y_len - 1 - n_hops;
    const int e_hops = x_len / 2;
    const int w_hops = x_len - 1 - e_hops;

    for (int root_y = 0; root_y < y_len; root_y++) {
        for (int root_x = 0; root_x < x_len; root_x++) {
            std::vector<std::uint8_t> trees(Routing2DCodec::MCAST_TREE_CAPACITY_BYTES, 0);
            std::string failure;
            ASSERT_TRUE(embed_mcast_reverse_trees(
                mesh_graph, MeshId{0}, *y_topo, *x_topo, root_y, root_x, trees.data(), &failure))
                << failure;

            std::vector<std::uint8_t> got(y_size + x_size, 0);
            encode_2d_mcast_maps(
                got.data(),
                trees.data(),
                y_size,
                x_size,
                static_cast<std::uint32_t>(root_y),
                static_cast<std::uint32_t>(root_x),
                static_cast<std::uint32_t>(n_hops),
                static_cast<std::uint32_t>(s_hops),
                static_cast<std::uint32_t>(e_hops),
                static_cast<std::uint32_t>(w_hops));

            const std::uint8_t outputs = got[root_y] & Routing2DCodec::ACTION_ETH_MASK;
            const int count = std::popcount(static_cast<unsigned>(outputs));

            // With a Y extent the source row is not a target row, so no teeth reach it.
            const std::uint8_t teeth = outputs & (Routing2DCodec::ACTION_EAST | Routing2DCodec::ACTION_WEST);
            EXPECT_EQ(teeth, 0) << fixture << ": root (" << root_y << "," << root_x
                                << ") has an E/W output on a range with a Y extent, where the source row is not a"
                                << " target";
            ASSERT_LE(count, 3) << fixture << ": root (" << root_y << "," << root_x << ") has " << count
                                << " outputs; a root on a range with a Y extent is bounded to the subset N/S/Z";
        }
    }
}

TEST(McastReverseTreeTest, FullExtentRoots8x4) {
    check_full_extent_roots("express_links_8x4_mesh_graph_descriptor.textproto");
}
TEST(McastReverseTreeTest, FullExtentRoots32x4) {
    check_full_extent_roots("express_links_32x4_mesh_graph_descriptor.textproto");
}

TEST(McastReverseTreeTest, Encode8x4) {
    check_encode("express_links_8x4_mesh_graph_descriptor.textproto", /*expect_multi_output_roots=*/false);
}
TEST(McastReverseTreeTest, Encode32x4) {
    check_encode("express_links_32x4_mesh_graph_descriptor.textproto", /*expect_multi_output_roots=*/true);
}

TEST(McastReverseTreeTest, Embed8x4) { check_embed("express_links_8x4_mesh_graph_descriptor.textproto"); }
TEST(McastReverseTreeTest, Embed32x4) { check_embed("express_links_32x4_mesh_graph_descriptor.textproto"); }

void check_maximum_embed(
    const std::string& name, const std::string& descriptor, std::uint32_t expected_y, std::uint32_t expected_x) {
    const auto path = fabric_router_tests::write_temp_descriptor(name, descriptor);
    const MeshGraph mesh_graph(tt::tt_metal::ClusterType::BLACKHOLE_GALAXY, path);
    const auto y_topo = derive_axis_topology(mesh_graph, MeshId{0}, 0);
    const auto x_topo = derive_axis_topology(mesh_graph, MeshId{0}, 1);

    ASSERT_EQ(y_topo.axis_len, expected_y);
    ASSERT_EQ(x_topo.axis_len, expected_x);
    ASSERT_EQ(
        Routing2DCodec::mcast_tree_region_bytes(expected_y, expected_x), Routing2DCodec::MCAST_TREE_CAPACITY_BYTES);

    std::vector<std::uint8_t> trees(Routing2DCodec::MCAST_TREE_CAPACITY_BYTES, 0);
    std::string failure;
    EXPECT_TRUE(embed_mcast_reverse_trees(
        mesh_graph,
        MeshId{0},
        y_topo,
        x_topo,
        static_cast<int>(expected_y - 1),
        static_cast<int>(expected_x - 1),
        trees.data(),
        &failure))
        << failure;
}

TEST(McastReverseTreeTest, EmbedMaximum64x4Shape) {
    check_maximum_embed(
        "fabric_64x4_line.textproto",
        R"(
mesh_descriptors {
  name: "M0"
  arch: BLACKHOLE
  device_topology { dims: [64, 4] dim_types: [LINE, LINE] }
  host_topology   { dims: [1, 1] }
  channels { count: 2 }
}
top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
)",
        64,
        4);
}

TEST(McastReverseTreeTest, EmbedMaximum4x64Shape) {
    check_maximum_embed(
        "fabric_4x64_line.textproto",
        R"(
mesh_descriptors {
  name: "M0"
  arch: BLACKHOLE
  device_topology { dims: [4, 64] dim_types: [LINE, LINE] }
  host_topology   { dims: [1, 1] }
  channels { count: 2 }
}
top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
)",
        4,
        64);
}

TEST(McastReverseTreeTest, Gate8x4) { check_fixture("express_links_8x4_mesh_graph_descriptor.textproto"); }
TEST(McastReverseTreeTest, Gate16x4) { check_fixture("express_links_16x4_mesh_graph_descriptor.textproto"); }
TEST(McastReverseTreeTest, Gate24x4) { check_fixture("express_links_24x4_mesh_graph_descriptor.textproto"); }
TEST(McastReverseTreeTest, Gate32x4) { check_fixture("express_links_32x4_mesh_graph_descriptor.textproto"); }

}  // namespace
}  // namespace tt::tt_fabric::mcast_reverse_tree_tests
