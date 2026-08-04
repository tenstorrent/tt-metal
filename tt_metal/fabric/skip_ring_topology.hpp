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

// Ring-domain decomposition of a mesh's skip axis, constructed from the declared skip-link patterns.
// Membership is cycle membership: a member need not own a chord. Indexed by axis coordinate (row), so
// one decomposition serves every line; derivation confirms every line carries the edges it implies.
struct SkipRingTopology {
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
    std::vector<int> domain_of;      // row -> domain, kNone for a leaf
    std::vector<int> leaf_run_of;    // row -> leaf run id, kNone for a transit row
    std::vector<int> leaf_index_of;  // leaf row -> its index within that run
    std::vector<LeafRun> leaf_runs;
    std::vector<std::vector<int>> forward_cycle;  // domain -> canonical forward node order
    std::vector<int> pos_in_domain;    // row -> index in its domain's forward cycle

    // Two-family axis only: the larger-span family, which may CONTINUE into the other; the reverse
    // direction is terminal. Crossovers are oriented (a in continue_src, b in the other family) and
    // ordered by a's forward position.
    int continue_src_domain = kNone;
    std::vector<std::pair<int, int>> crossovers;

    bool is_leaf(int row) const { return domain_of[row] == kNone; }
    int ring_distance(int domain, int from, int to) const;
    // Which end a run is entered by depends only on `dst`, which is what keeps routes suffix-consistent.
    int next_row(int src, int dst) const;
};

// std::nullopt when the mesh declares no skip links, leaving base routing untouched. Throws on any
// topology this cut does not define; each check carries its own message.
std::optional<SkipRingTopology> derive_skip_ring_topology(const MeshGraph& mesh_graph, MeshId mesh_id);

// The ordinary ring along `axis`, or std::nullopt when that dimension does not close. One domain over
// every node in coordinate order, no chords, so both axes answer ring queries through one path.
std::optional<SkipRingTopology> derive_ordinary_ring_topology(const MeshGraph& mesh_graph, MeshId mesh_id, int axis);

// Human-readable dump of the stored decomposition plus the declared patterns it was built from and
// the materialized edges it was checked against.
std::string describe_skip_rings(const MeshGraph& mesh_graph, MeshId mesh_id, const SkipRingTopology& topo);

}  // namespace tt::tt_fabric
