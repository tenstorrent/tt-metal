// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sweeps AxisRouteTopology over *every* mesh graph descriptor in the tree, on both axes of every
// mesh, and checks the routing invariants that make a derived topology usable.
//
// Why this exists: the per-axis derivation used to run only for express meshes, and the existing
// coverage (test_mcast_reverse_tree.cpp, test_express_ring_topology.cpp) sweeps only the four
// express_links_*x4 fixtures. The codec unification made derivation unconditional -- express, then
// ring, then LINE -- so the shapes with no coverage at all are exactly the ones the change added:
// plain non-express meshes, and meshes wider than X = 4.
//
// The invariant that matters most is that a hop must cross a **declared edge**. A line has no
// closing edge, so a topology that reasoned modularly would route "the short way" from row 0 to the
// last row over a link that does not exist. On device that is a hang, not a wrong answer.
//
// Machine-free: MeshGraph(ClusterType, path) needs no cluster, no discovery and no topology mapper.

#include <gtest/gtest.h>

#include <filesystem>
#include <set>
#include <string>
#include <vector>

#include <tt-metalium/experimental/fabric/mesh_graph.hpp>

// mesh_graph.hpp only forward-declares ClusterType; the MeshGraph constructor takes it by value.
#include "cluster.hpp"
#include "hostdevcommon/fabric_common.h"
#include "llrt/rtoptions.hpp"
#include "tt_metal/fabric/axis_route_topology.hpp"

namespace tt::tt_fabric::axis_topology_sweep_tests {
namespace {

// Both descriptor homes: the shipped set and the test fixtures.
std::vector<std::filesystem::path> all_descriptors() {
    const tt::llrt::RunTimeOptions rtoptions;
    const std::filesystem::path root(rtoptions.get_root_dir());
    std::vector<std::filesystem::path> out;
    for (const auto& dir : {
             std::filesystem::path("tt_metal/fabric/mesh_graph_descriptors"),
             std::filesystem::path("tests/tt_metal/tt_fabric/custom_mesh_descriptors"),
         }) {
        std::error_code ec;
        for (const auto& entry : std::filesystem::directory_iterator(root / dir, ec)) {
            if (entry.is_regular_file() && entry.path().extension() == ".textproto") {
                out.push_back(entry.path());
            }
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

// A descriptor that will not load is not this test's business -- several need rank bindings, a
// specific cluster type, or a live topology mapper. They are counted, not asserted on, and the
// suite fails if too few load (see MinimumCoverage) so it cannot quietly become vacuous.
struct SweepResult {
    int descriptors_loaded = 0;
    int descriptors_skipped = 0;
    int axes_checked = 0;
    int lines = 0;
    int rings_or_express = 0;
    int derivations_failed = 0;
};

// Walks the canonical route src -> dst one hop at a time and checks each hop.
void check_route(
    const MeshGraph& mesh_graph,
    MeshId mesh_id,
    const AxisRouteTopology& topo,
    int src,
    int dst,
    const std::string& where) {
    int cur = src;
    std::set<int> visited{src};
    for (int guard = 0; cur != dst; guard++) {
        ASSERT_LE(guard, topo.axis_len) << where << ": route " << src << " -> " << dst << " did not converge";

        const int next = topo.next_row(cur, dst);

        // In range, and never a self-loop.
        ASSERT_GE(next, 0) << where << ": hop " << cur << " -> " << next << " is out of range";
        ASSERT_LT(next, topo.axis_len) << where << ": hop " << cur << " -> " << next << " is out of range";
        ASSERT_NE(next, cur) << where << ": row " << cur << " routes to itself for destination " << dst;

        // THE check: the hop must cross an edge the descriptor actually declares. This is what a
        // line stepping over its absent wrap edge fails.
        const auto direction = axis_edge_direction(mesh_graph, mesh_id, topo.axis_dim, 0, cur, next);
        ASSERT_TRUE(direction.has_value())
            << where << ": hop " << cur << " -> " << next << " (routing to " << dst << ") crosses an edge that the "
            << "descriptor does not declare";

        // No row may repeat: a revisit is a cycle, which on device is a livelock.
        ASSERT_TRUE(visited.insert(next).second)
            << where << ": route " << src << " -> " << dst << " revisits row " << next;
        cur = next;
    }
}

SweepResult sweep(bool exhaustive_routes) {
    SweepResult r;
    for (const auto& path : all_descriptors()) {
        std::unique_ptr<MeshGraph> mesh_graph;
        try {
            mesh_graph = std::make_unique<MeshGraph>(tt::tt_metal::ClusterType::BLACKHOLE_GALAXY, path.string());
        } catch (...) {
            r.descriptors_skipped++;
            continue;
        }
        r.descriptors_loaded++;

        for (const auto mesh_id : mesh_graph->get_mesh_ids()) {
            const auto shape = mesh_graph->get_mesh_shape(mesh_id);
            for (int axis = 0; axis < 2; axis++) {
                const int axis_len = static_cast<int>(shape[axis]);
                if (axis_len < 2) {
                    continue;  // degenerate axis: no hops to make
                }
                AxisRouteTopology topo;
                try {
                    topo = derive_axis_topology(*mesh_graph, mesh_id, axis);
                } catch (const std::exception& e) {
                    // Only the route-checking pass owns derivation failures. MinimumCoverage is a
                    // counting guard and must not re-report the same fault as a second red test.
                    if (exhaustive_routes) {
                        ADD_FAILURE() << path.filename().string() << " mesh " << *mesh_id << " axis " << axis
                                      << ": derivation threw: " << e.what();
                    }
                    r.derivations_failed++;
                    continue;
                }
                r.axes_checked++;
                (topo.wraps ? r.rings_or_express : r.lines)++;

                const std::string where =
                    path.filename().string() + " mesh " + std::to_string(*mesh_id) + " axis " + std::to_string(axis);

                // Structural sanity: every row is placed, and placements are self-consistent.
                // EXPECT, not ASSERT: this helper returns a value, and ASSERT_* only compiles in a
                // void function. A structural mismatch here is reported and the sweep continues.
                EXPECT_EQ(static_cast<int>(topo.domain_of.size()), axis_len) << where << ": domain_of is undersized";
                EXPECT_EQ(topo.axis_len, axis_len) << where << ": axis_len disagrees with the mesh shape";

                if (!exhaustive_routes) {
                    continue;
                }
                for (int src = 0; src < axis_len; src++) {
                    for (int dst = 0; dst < axis_len; dst++) {
                        if (src != dst) {
                            check_route(*mesh_graph, mesh_id, topo, src, dst, where);
                            if (::testing::Test::HasFatalFailure()) {
                                return r;
                            }
                        }
                    }
                }
            }
        }
    }
    return r;
}

}  // namespace

// Every hop of every canonical route, on every axis of every mesh of every descriptor that loads,
// crosses a declared edge and converges without revisiting a row.
TEST(AxisTopologySweep, EveryRouteCrossesOnlyDeclaredEdges) {
    const auto r = sweep(/*exhaustive_routes=*/true);
    RecordProperty("descriptors_loaded", r.descriptors_loaded);
    RecordProperty("descriptors_skipped", r.descriptors_skipped);
    RecordProperty("axes_checked", r.axes_checked);
    RecordProperty("line_axes", r.lines);
    RecordProperty("ring_or_express_axes", r.rings_or_express);

    // Both families must actually appear, or the sweep is not exercising what it claims to.
    EXPECT_GT(r.lines, 0) << "no LINE axis was swept -- the line fallback is going untested";
    EXPECT_GT(r.rings_or_express, 0) << "no ring/express axis was swept";
}

// Guards against the sweep silently degrading to nothing if descriptor loading breaks or the
// directories move. The floor is deliberately far below the real count.
TEST(AxisTopologySweep, MinimumCoverage) {
    const auto r = sweep(/*exhaustive_routes=*/false);
    EXPECT_GE(r.descriptors_loaded, 20) << "only " << r.descriptors_loaded << " descriptors loaded ("
                                        << r.descriptors_skipped << " skipped) -- the sweep is nearly vacuous";
    EXPECT_GE(r.axes_checked, 40) << "only " << r.axes_checked << " axes checked";
    // Reported, not asserted: EveryRouteCrossesOnlyDeclaredEdges is the test that fails on these.
    RecordProperty("derivations_failed", r.derivations_failed);
}

// Every mesh shape that a descriptor declares must be representable by the indexed codec, or the
// control plane's validation would reject it at startup. Shapes that exceed the packet header's
// route buffer are reported rather than asserted: that is a separate bound (issue #32237) and the
// host fatals with a clear message, so it is a known limit, not a defect.
TEST(AxisTopologySweep, EveryDeclaredShapeIsIndexable) {
    int checked = 0;
    std::vector<std::string> over_header_bound;

    for (const auto& path : all_descriptors()) {
        std::unique_ptr<MeshGraph> mesh_graph;
        try {
            mesh_graph = std::make_unique<MeshGraph>(tt::tt_metal::ClusterType::BLACKHOLE_GALAXY, path.string());
        } catch (...) {
            continue;
        }
        for (const auto mesh_id : mesh_graph->get_mesh_ids()) {
            const auto shape = mesh_graph->get_mesh_shape(mesh_id);
            const auto y = static_cast<uint32_t>(shape[0]);
            const auto x = static_cast<uint32_t>(shape[1]);
            checked++;

            EXPECT_TRUE(IndexedMeshRoutingFields::shape_is_indexable(y, x))
                << path.filename().string() << " mesh " << *mesh_id << " shape " << y << "x" << x
                << " cannot be packed into the indexed vectors";

            constexpr uint32_t kMaxRouteBytes = 67;  // see fabric_edm_packet_header.hpp
            if (y + x > kMaxRouteBytes) {
                over_header_bound.push_back(
                    path.filename().string() + " (" + std::to_string(y) + "x" + std::to_string(x) + ")");
            }
        }
    }
    EXPECT_GT(checked, 0) << "no mesh shapes were checked";

    // Informational: these need issue #32237 before they could run 2D fabric.
    for (const auto& s : over_header_bound) {
        RecordProperty("exceeds_header_route_buffer", s);
    }
}

}  // namespace tt::tt_fabric::axis_topology_sweep_tests
