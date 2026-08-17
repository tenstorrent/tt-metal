// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The §5.7.1 arborescence gate over every express fixture, for every root on both axes.
//
// This is the precondition the whole indexed-multicast representation rests on, and it is
// all-or-nothing by design: V1 runs one encoder, so a single failing root rejects the mesh with no
// fallback. Checking it is cheap and machine-free -- MeshGraph(ClusterType, path) and
// ExpressRingTopology::next_row need no cluster, no discovery, and no matching host world -- so the
// multi-host shapes are covered here on one machine rather than only on a four-host run.
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
#include "tt_metal/fabric/express_ring_topology.hpp"
#include "tt_metal/fabric/mcast_reverse_tree.hpp"

namespace tt::tt_fabric::mcast_reverse_tree_tests {
namespace {

MeshGraph load(const std::string& fixture) {
    // A local RunTimeOptions rather than MetalContext::instance(): the context eagerly builds a
    // Cluster, which throws when no chips are present and would make these checks require a machine
    // for nothing but a path lookup.
    const tt::llrt::RunTimeOptions rtoptions;
    const auto path =
        std::filesystem::path(rtoptions.get_root_dir()) / "tests/tt_metal/tt_fabric/custom_mesh_descriptors" / fixture;
    return MeshGraph(tt::tt_metal::ClusterType::BLACKHOLE_GALAXY, path.string());
}

// Recompute union(R(root, dst)) straight from next_row, independently of the generator, so the tree
// is checked against the routes it claims to represent rather than against its own bookkeeping.
std::map<int, std::pair<int, RoutingDirection>> canonical_edges(
    const MeshGraph& mesh_graph, MeshId mesh_id, const ExpressRingTopology& topo, int root) {
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

// union over the requested targets of the actions each row drives on the canonical route to them,
// walked straight from next_row. This is the definition the prune has to reproduce.
std::vector<std::uint8_t> expected_actions(
    const MeshGraph& mesh_graph,
    MeshId mesh_id,
    const ExpressRingTopology& topo,
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

// The action bit a packed 2-bit output code stands for, resolved independently of the packer so the
// round trip checks the encoding rather than restating it.
std::uint8_t expected_action_for_code(int axis_dim, std::uint8_t code) {
    if (axis_dim == 0) {
        switch (code) {
            case IndexedMeshRoutingFields::Y2_NORTH: return IndexedMeshRoutingFields::ACTION_NORTH;
            case IndexedMeshRoutingFields::Y2_SOUTH: return IndexedMeshRoutingFields::ACTION_SOUTH;
            case IndexedMeshRoutingFields::Y2_Z: return IndexedMeshRoutingFields::ACTION_Z;
            default: return 0;
        }
    }
    switch (code) {
        case IndexedMeshRoutingFields::X2_EAST: return IndexedMeshRoutingFields::ACTION_EAST;
        case IndexedMeshRoutingFields::X2_WEST: return IndexedMeshRoutingFields::ACTION_WEST;
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
    const ExpressRingTopology& topo,
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

// Every singleton, every pair, the full set, and deterministic pseudo-random subsets. Singletons pin
// each individual route, pairs are where a wrong `needed` propagation first shows up as a branch that
// is taken for one target and lost for the other, and the full set exercises maximum fan-out.
void check_prune_over_target_sets(
    const MeshGraph& mesh_graph,
    MeshId mesh_id,
    const ExpressRingTopology& topo,
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

void check_axis(const MeshGraph& mesh_graph, const ExpressRingTopology& topo, const std::string& label) {
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

        // Every row present once as a child, and never the root: indegree 1 everywhere but the root.
        std::set<int> children;
        for (const auto& edge : tree.edges) {
            EXPECT_TRUE(children.insert(edge.child).second) << where << ": row " << edge.child << " has two parents";
            EXPECT_NE(edge.child, tree.root) << where << ": the root has a parent";
        }

        // Descendants before ancestors -- the ordering the worker's single reverse pass depends on. An
        // edge hanging off row r must be visited before the edge that enters r, or the propagation
        // that marks r needed would run after r's own edge had already been read.
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

    // X is the ordinary four-column ring: E/W only, no chords. Cheap, and it is part of the same
    // mesh-wide gate, since a multicast encodes both axes.
    const auto ordinary_x = derive_ordinary_ring_topology(mesh_graph, MeshId{0}, 1);
    ASSERT_TRUE(ordinary_x.has_value()) << fixture << ": X dimension does not close into a ring";
    check_axis(mesh_graph, *ordinary_x, fixture + " X");
}

// The packing order obligation is worth a negative test: it is the one §5.7.1 requirement whose
// violation is silent on device, dropping branches from the encoded map rather than faulting.
TEST(McastReverseTreeTest, PackingRejectsAncestorBeforeDescendant) {
    McastReverseTree tree;
    tree.root = 0;
    tree.axis_dim = 0;
    tree.axis_len = 3;
    // 0 -> 1 -> 2, serialized root-first, which is exactly backwards.
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

// The embed is a host<->device layout contract, so what matters is not only that the right tree lands
// but that it lands where the loader will look and nowhere else. A sentinel-filled table makes both
// halves of that checkable: the written span and, just as importantly, the untouched one.
void check_embed(const std::string& fixture) {
    const auto mesh_graph = load(fixture);
    const auto y_topo = derive_express_ring_topology(mesh_graph, MeshId{0});
    ASSERT_TRUE(y_topo.has_value()) << fixture << ": derived no express rings";
    const auto x_topo = derive_ordinary_ring_topology(mesh_graph, MeshId{0}, 1);
    ASSERT_TRUE(x_topo.has_value()) << fixture << ": X dimension does not close into a ring";

    const auto y_size = static_cast<std::uint32_t>(y_topo->axis_len);
    const auto x_size = static_cast<std::uint32_t>(x_topo->axis_len);
    ASSERT_TRUE(IndexedMeshRoutingFields::hybrid_region_fits(y_size, x_size)) << fixture;

    const auto tree_offset = IndexedMeshRoutingFields::mcast_tree_offset_bytes(y_size, x_size);
    const auto tree_bytes = IndexedMeshRoutingFields::mcast_tree_region_bytes(y_size, x_size);
    constexpr std::uint8_t kSentinel = 0xA5;

    for (std::uint32_t my_y = 0; my_y < y_size; my_y++) {
        for (std::uint32_t my_x = 0; my_x < x_size; my_x++) {
            const std::string where = fmt::format("{} chip ({},{})", fixture, my_y, my_x);
            std::vector<std::uint8_t> table(IndexedMeshRoutingFields::INDEXED_VECTOR_TABLE_BYTES, kSentinel);

            std::string failure;
            ASSERT_TRUE(embed_mcast_reverse_trees(
                mesh_graph,
                MeshId{0},
                *y_topo,
                *x_topo,
                static_cast<int>(my_y),
                static_cast<int>(my_x),
                table.data(),
                &failure))
                << where << ": " << failure;

            for (std::uint32_t i = 0; i < table.size(); i++) {
                if (i >= tree_offset && i < tree_offset + tree_bytes) {
                    continue;
                }
                ASSERT_EQ(table[i], kSentinel) << where << ": wrote outside the tree region at byte " << i;
            }

            // The embedded words must be the same artifact the generator and packer produce standalone,
            // for this chip's own row and column and no other.
            const auto y_packed = pack_mcast_reverse_tree(
                *build_mcast_reverse_tree(mesh_graph, MeshId{0}, *y_topo, static_cast<int>(my_y)));
            const auto x_packed = pack_mcast_reverse_tree(
                *build_mcast_reverse_tree(mesh_graph, MeshId{0}, *x_topo, static_cast<int>(my_x)));
            ASSERT_TRUE(y_packed.has_value() && x_packed.has_value()) << where;

            const std::uint8_t* y_region = table.data() + IndexedMeshRoutingFields::mcast_tree_y_offset(y_size, x_size);
            for (std::uint32_t i = 0; i < y_packed->size(); i++) {
                EXPECT_EQ(IndexedMeshRoutingFields::get_mcast_tree_edge(y_region, i), (*y_packed)[i])
                    << where << ": Y edge " << i;
            }
            const std::uint8_t* x_region = table.data() + IndexedMeshRoutingFields::mcast_tree_x_offset(y_size, x_size);
            for (std::uint32_t i = 0; i < x_packed->size(); i++) {
                EXPECT_EQ(IndexedMeshRoutingFields::get_mcast_tree_edge(x_region, i), (*x_packed)[i])
                    << where << ": X edge " << i;
            }
        }
    }
}

// The rectangle the client asked for, as row/column target sets. Shared with the encoder by
// definition rather than by implementation: what the golden below checks independently is the
// routing, which is the part a tree can get wrong.
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

// Golden §5.6 map: trace the canonical route to every target and OR the actions, then apply the §5.5
// teeth and delivery rules. This walks routes forward and never touches a reverse tree, so agreeing
// with the device encoder means the tree really does stand for the routes.
//
// Multicast is where a wrong map is most expensive -- it does not fail, it delivers to the wrong set
// of chips -- so this checks the encoder the device actually runs, not a host restatement of it.
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

    // Extents worth covering: none, one row either way, a two-sided range, a one-sided reach far
    // enough to cross a ring boundary, and a range spanning most of the axis.
    const std::vector<std::pair<int, int>> y_extents = {{0, 0}, {1, 0}, {0, 1}, {2, 2}, {y_len / 2, 0}, {0, y_len / 2}};
    const std::vector<std::pair<int, int>> x_extents = {{0, 0}, {1, 0}, {0, 1}, {1, x_len - 2}};

    int multi_output_roots = 0;

    for (int root_y = 0; root_y < y_len; root_y++) {
        for (int root_x = 0; root_x < x_len; root_x++) {
            std::vector<std::uint8_t> table(IndexedMeshRoutingFields::INDEXED_VECTOR_TABLE_BYTES, 0);
            std::string failure;
            ASSERT_TRUE(embed_mcast_reverse_trees(
                mesh_graph, MeshId{0}, *y_topo, *x_topo, root_y, root_x, table.data(), &failure))
                << failure;

            for (const auto& [n_hops, s_hops] : y_extents) {
                for (const auto& [w_hops, e_hops] : x_extents) {
                    const std::string where = fmt::format(
                        "{} root ({},{}) N{} S{} W{} E{}", fixture, root_y, root_x, n_hops, s_hops, w_hops, e_hops);

                    std::vector<std::uint8_t> got(y_size + x_size, 0);
                    encode_indexed_mcast_maps(
                        got.data(),
                        table.data(),
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
                            want_x[x] |= IndexedMeshRoutingFields::ACTION_LOCAL_DELIVER;
                        }
                    }
                    auto want_y = expected_actions(mesh_graph, MeshId{0}, *y_topo, root_y, targets_y);
                    const std::uint8_t x_root_action = want_x[root_x];
                    const std::uint8_t teeth =
                        x_root_action & (IndexedMeshRoutingFields::ACTION_EAST | IndexedMeshRoutingFields::ACTION_WEST);
                    const std::uint8_t deliver = x_root_action & IndexedMeshRoutingFields::ACTION_LOCAL_DELIVER;
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

                    const std::uint8_t root_outputs = got[root_y] & ~IndexedMeshRoutingFields::ACTION_LOCAL_DELIVER;
                    if (root_outputs != 0 && (root_outputs & (root_outputs - 1)) != 0) {
                        multi_output_roots++;
                    }
                    // §7.3.1 states N/S/Z source fanout fits the existing four-slot connection
                    // manager, which is why multi-inject needs no transport capacity work. If a root
                    // ever wanted more outputs than there are slots, that conclusion would be wrong.
                    EXPECT_LE(std::popcount(static_cast<unsigned>(root_outputs)), 4)
                        << where << ": root wants more outputs than a connection manager holds";
                    // §7.3.1: a root with no eth outputs is legal and means deliver locally only.
                    // A rectangle of nothing but the anchor is exactly that case, and the producer
                    // must not treat it as an error.
                    if (n_hops == 0 && s_hops == 0 && e_hops == 0 && w_hops == 0) {
                        EXPECT_EQ(root_outputs, 0) << where << ": local-only mcast must leave no eth output";
                        EXPECT_NE(got[root_y] & IndexedMeshRoutingFields::ACTION_LOCAL_DELIVER, 0)
                            << where << ": local-only mcast must still deliver at the source";
                    }
                }
            }
        }
    }

    // A one-hop range reaches an adjacent row or column, which no canonical route should need a chord
    // to do, so its root should leave on exactly one edge from every source. The express multicast
    // YAML relies on that: those are the ranges the test infra can drive before source multi-inject
    // exists, and a failure here is how we find out on the host instead of on the cluster.
    for (int root_y = 0; root_y < y_len; root_y++) {
        for (int root_x = 0; root_x < x_len; root_x++) {
            std::vector<std::uint8_t> table(IndexedMeshRoutingFields::INDEXED_VECTOR_TABLE_BYTES, 0);
            ASSERT_TRUE(
                embed_mcast_reverse_trees(mesh_graph, MeshId{0}, *y_topo, *x_topo, root_y, root_x, table.data()));
            const std::vector<std::array<int, 4>> one_hop = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
            for (const auto& [n, s, e, w] : one_hop) {
                std::vector<std::uint8_t> got(y_size + x_size, 0);
                encode_indexed_mcast_maps(
                    got.data(),
                    table.data(),
                    y_size,
                    x_size,
                    static_cast<std::uint32_t>(root_y),
                    static_cast<std::uint32_t>(root_x),
                    n,
                    s,
                    e,
                    w);
                const std::uint8_t outputs = got[root_y] & IndexedMeshRoutingFields::ACTION_ETH_MASK;
                EXPECT_EQ(std::popcount(static_cast<unsigned>(outputs)), 1)
                    << fixture << " root (" << root_y << "," << root_x << ") N" << n << " S" << s << " E" << e << " W"
                    << w << ": one-hop range must leave on a single edge";
            }
        }
    }

    // §7.3.1 claims multi-output roots are ordinary on an express axis rather than a corner case, and
    // that claim is the whole reason express multicast still needs source multi-inject. If it were
    // false here, the blocker would not be real and the plan would be wrong.
    if (expect_multi_output_roots) {
        EXPECT_GT(multi_output_roots, 0) << fixture << ": expected express routing to produce multi-output roots";
    }
}

struct ChordRange {
    int n_hops = 0;
    int s_hops = 0;
    std::vector<int> roots;  // single-output roots whose map drives a chord
    int multi_output_roots = 0;
};

// Which ranges exercise a chord on hardware today, before source multi-inject exists.
//
// A one-hop range never routes over a chord -- reaching an adjacent row needs no shortcut -- so the
// ranges the express multicast YAML can currently drive prove the codec end to end without ever
// crossing an express edge. That gap is narrower than it looks. A multi-output root is forced only
// when the root is itself a chord tail whose range reaches that chord's head; a root that is not a
// tail can leave on a single edge and have a router further along take the chord, because transit
// routers clone through the source RX and need no multi-inject at all.
//
// So the useful question is not "which extents are single-output everywhere" -- for any extent wide
// enough to reach a chord head the tail roots are multi-output, so the answer there is none -- but
// "for a given extent, which roots are single-output and still drive a chord". Those roots are
// exactly the senders the YAML can list, one entry per device, and they are what turns the express
// multicast test from a codec smoke test into express coverage.
//
// The column is not swept: with no E/W extent the X map is local delivery only and contributes no
// teeth, so the Y map depends on the root row alone and every column of a reported row behaves the
// same.
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
        tables[root_y].assign(IndexedMeshRoutingFields::INDEXED_VECTOR_TABLE_BYTES, 0);
        std::string failure;
        ASSERT_TRUE(embed_mcast_reverse_trees(
            mesh_graph, MeshId{0}, *y_topo, *x_topo, root_y, 0, tables[root_y].data(), &failure))
            << failure;
    }

    std::vector<ChordRange> found;
    for (int n_hops = 0; n_hops < y_len; n_hops++) {
        for (int s_hops = 0; n_hops + s_hops < y_len; s_hops++) {
            ChordRange range{n_hops, s_hops, {}, 0};
            for (int root_y = 0; root_y < y_len; root_y++) {
                std::vector<std::uint8_t> got(y_size + x_size, 0);
                encode_indexed_mcast_maps(
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

                const std::uint8_t outputs = got[root_y] & IndexedMeshRoutingFields::ACTION_ETH_MASK;
                if (std::popcount(static_cast<unsigned>(outputs)) > 1) {
                    range.multi_output_roots++;
                    continue;
                }
                if (std::popcount(static_cast<unsigned>(outputs)) != 1) {
                    continue;
                }
                for (int y = 0; y < y_len; y++) {
                    if ((got[y] & IndexedMeshRoutingFields::ACTION_Z) != 0) {
                        range.roots.push_back(root_y);
                        break;
                    }
                }
            }
            if (!range.roots.empty()) {
                found.push_back(std::move(range));
            }
        }
    }

    if (expect_candidates) {
        EXPECT_FALSE(found.empty()) << fixture << ": no range routes over a chord from a single-output root, so"
                                    << " express multicast coverage really does have to wait for multi-inject";
    }
}

// 32x4 has multi-output roots, so it certainly has chords to reach; whether any of them sit
// downstream of a single-output root is the question, and an empty result would be a real finding.
// 8x4 is reported without an expectation because its express pattern is coarser and may admit none.
TEST(McastReverseTreeTest, ChordRanges8x4) {
    check_chord_ranges("express_links_8x4_mesh_graph_descriptor.textproto", /*expect_candidates=*/false);
}
TEST(McastReverseTreeTest, ChordRanges32x4) {
    check_chord_ranges("express_links_32x4_mesh_graph_descriptor.textproto", /*expect_candidates=*/true);
}

// Whether an all-to-all is reachable, asked without hardware: every chip roots a range covering the
// whole mesh, so the question is how many outputs a full-extent root has and therefore how many
// copies the sender must inject (§7.3.1).
//
// The root action is route_buffer_y[root_y] alone. It is NOT that OR'd with route_buffer_x[root_x],
// which is the mistake this test was first written with. §5.5 copies the X-root E/W teeth onto the
// target rows, and §7.3.1 records that a range with a Y extent has target rows excluding the source
// row -- so the source never picks up the teeth, and its outputs are a subset of N/S/Z. Reading the X
// map here invents outputs the source does not have and reports five where the contract allows three.
//
// An X-only range is the other case: with no Y extent the source row is itself a target, so the teeth
// do land on it and the action is a subset of E/W. Either way route_buffer_y[root_y] is the whole
// answer, which is why mcast_root_output_directions reads only that byte.
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

    // The whole mesh minus the source: split the axis so N and S together cover every other row, and
    // likewise E and W. This is the widest range a client can ask for.
    const int n_hops = y_len / 2;
    const int s_hops = y_len - 1 - n_hops;
    const int e_hops = x_len / 2;
    const int w_hops = x_len - 1 - e_hops;

    for (int root_y = 0; root_y < y_len; root_y++) {
        for (int root_x = 0; root_x < x_len; root_x++) {
            std::vector<std::uint8_t> table(IndexedMeshRoutingFields::INDEXED_VECTOR_TABLE_BYTES, 0);
            std::string failure;
            ASSERT_TRUE(embed_mcast_reverse_trees(
                mesh_graph, MeshId{0}, *y_topo, *x_topo, root_y, root_x, table.data(), &failure))
                << failure;

            std::vector<std::uint8_t> got(y_size + x_size, 0);
            encode_indexed_mcast_maps(
                got.data(),
                table.data(),
                y_size,
                x_size,
                static_cast<std::uint32_t>(root_y),
                static_cast<std::uint32_t>(root_x),
                static_cast<std::uint32_t>(n_hops),
                static_cast<std::uint32_t>(s_hops),
                static_cast<std::uint32_t>(e_hops),
                static_cast<std::uint32_t>(w_hops));

            const std::uint8_t outputs = got[root_y] & IndexedMeshRoutingFields::ACTION_ETH_MASK;
            const int count = std::popcount(static_cast<unsigned>(outputs));

            // §7.3.1: with a Y extent the source row is not a target row, so no teeth reach it. A
            // root that sets E or W here means §5.5's copy landed on the source row after all, and
            // the sender would need a connection the contract says it never needs.
            const std::uint8_t teeth =
                outputs & (IndexedMeshRoutingFields::ACTION_EAST | IndexedMeshRoutingFields::ACTION_WEST);
            EXPECT_EQ(teeth, 0) << fixture << ": root (" << root_y << "," << root_x
                                << ") has an E/W output on a range with a Y extent, which §7.3.1 excludes";
            ASSERT_LE(count, 3) << fixture << ": root (" << root_y << "," << root_x << ") has " << count
                                << " outputs; §7.3.1 bounds a Y-extent root to the subset N/S/Z";
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

// [64,4] is the documented growth boundary, and the embed must say so rather than run off the slot.
// The fit check reads only the axis lengths, so a bare topology is enough to reach it.
TEST(McastReverseTreeTest, EmbedRejectsShapeThatDoesNotFit) {
    const auto mesh_graph = load("express_links_32x4_mesh_graph_descriptor.textproto");
    ExpressRingTopology y_topo;
    y_topo.axis_dim = 0;
    y_topo.axis_len = 64;
    ExpressRingTopology x_topo;
    x_topo.axis_dim = 1;
    x_topo.axis_len = 4;

    std::vector<std::uint8_t> table(IndexedMeshRoutingFields::INDEXED_VECTOR_TABLE_BYTES, 0);
    std::string failure;
    EXPECT_FALSE(embed_mcast_reverse_trees(mesh_graph, MeshId{0}, y_topo, x_topo, 0, 0, table.data(), &failure));
    EXPECT_NE(failure.find("union slot"), std::string::npos) << "actual: " << failure;
}

TEST(McastReverseTreeTest, Gate8x4) { check_fixture("express_links_8x4_mesh_graph_descriptor.textproto"); }
TEST(McastReverseTreeTest, Gate16x4) { check_fixture("express_links_16x4_mesh_graph_descriptor.textproto"); }
TEST(McastReverseTreeTest, Gate24x4) { check_fixture("express_links_24x4_mesh_graph_descriptor.textproto"); }
TEST(McastReverseTreeTest, Gate32x4) { check_fixture("express_links_32x4_mesh_graph_descriptor.textproto"); }

}  // namespace
}  // namespace tt::tt_fabric::mcast_reverse_tree_tests
