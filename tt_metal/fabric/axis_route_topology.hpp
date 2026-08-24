// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/fabric/mesh_graph.hpp>

namespace tt::tt_fabric {

// Ring-domain decomposition of a mesh's express axis, constructed from the declared express-link patterns.
// Membership is cycle membership: a member need not own a chord. Indexed by axis coordinate (row), so
// one decomposition serves every line; derivation confirms every line carries the edges it implies.
struct AxisRouteTopology {
    static constexpr int kNone = -1;

    // A maximal run of skipped rows, bridging the two transit rows either side of it. The run is a
    // path parallel to the chord it bypasses, so traversing it end to end would close an unprotected
    // cycle; routes enter from one end and walk inward to a destination inside it, never through.
    struct LeafRun {
        std::vector<int> rows;      // increasing-row order, anchor_before side first
        int anchor_before = kNone;  // transit row preceding rows.front()
        int anchor_after = kNone;   // transit row following rows.back()
    };

    int axis_dim = 0;  // dimension the chords jump along; the domains' dimension
    int axis_len = 0;
    // Whether the axis closes. False only for a LINE axis (derive_line_axis_topology), where there is
    // no edge joining the last coordinate back to the first. next_row()'s step() otherwise picks a
    // direction by comparing forward against backward distance modulo the cycle length, which on a
    // line would hand back the far end as if one hop away -- a route over an edge that does not
    // exist. Defaults true so express and ordinary-ring behaviour is unchanged.
    bool wraps = true;
    std::vector<int> domain_of;      // row -> domain, kNone for a leaf
    std::vector<int> leaf_run_of;    // row -> leaf run id, kNone for a transit row
    std::vector<int> leaf_index_of;  // leaf row -> its index within that run
    std::vector<LeafRun> leaf_runs;
    std::vector<std::vector<int>> forward_cycle;  // domain -> canonical forward node order
    std::vector<int> pos_in_domain;               // row -> index in its domain's forward cycle

    // Two-family axis only: the larger-span family, which may CONTINUE into the other; the reverse
    // direction is terminal. Crossovers are oriented (a in continue_src, b in the other family) and
    // ordered by a's forward position.
    int continue_src_domain = kNone;
    std::vector<std::pair<int, int>> crossovers;

    bool is_leaf(int row) const { return domain_of[row] == kNone; }
    int ring_distance(int domain, int from, int to) const;
    // Which end a run is entered by depends only on `dst`, which is what keeps routes suffix-consistent.
    int next_row(int src, int dst) const;

    // Directed row-hops that take part in a dependency cycle across the generated routes yet are not
    // protected-ring edges. Empty means no unprotected cycle can form (the CDG/SCC condition); a
    // non-empty result is a deadlock risk. Built by walking next_row over every ordered row pair,
    // forming the edge-level control-dependency graph, and running SCC over it.
    std::vector<std::pair<int, int>> cyclic_non_ring_hops() const;
};

// port_direction of the axis edge between two rows of one line, or nullopt when there is none. `axis`
// selects which coordinate the rows index and `ortho` fixes the line; a Z chord answers here exactly
// like a cardinal edge, which is what lets callers name a hop's command without a chord lookup.
std::optional<RoutingDirection> axis_edge_direction(
    const MeshGraph& mesh_graph, MeshId mesh_id, int axis, int ortho, int row_a, int row_b);

// std::nullopt when the mesh declares no express links, leaving base routing untouched. Throws on any
// topology this cut does not define; each check carries its own message.
std::optional<AxisRouteTopology> derive_express_ring_topology(const MeshGraph& mesh_graph, MeshId mesh_id);

// The ordinary ring along `axis`, or std::nullopt when that dimension does not close. One domain over
// every node in coordinate order, no chords, so both axes answer ring queries through one path.
std::optional<AxisRouteTopology> derive_ordinary_ring_topology(const MeshGraph& mesh_graph, MeshId mesh_id, int axis);

// The plain line along `axis`: one domain in coordinate order, no chords, and no closing edge
// (wraps == false). Always derivable, so it is the fallback that guarantees every 2D mesh has an axis
// topology on both dimensions. Trivially an arborescence from every root, so the multicast
// one-feeder-per-row gate always passes on a line. Throws only if an interior edge is missing, which
// means the axis is not a line either.
AxisRouteTopology derive_line_axis_topology(const MeshGraph& mesh_graph, MeshId mesh_id, int axis);

// The topology governing `axis`, in precedence order: express chords where declared for that axis,
// the ordinary ring where the axis closes, else the plain line. Never returns nothing, so callers can
// rely on a non-null answer for any 2D mesh -- which is what keeps a missing axis topology from
// silently disabling multicast.
AxisRouteTopology derive_axis_topology(const MeshGraph& mesh_graph, MeshId mesh_id, int axis);
}  // namespace tt::tt_fabric
