// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "express_ring_topology.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <enchantum/enchantum.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp>

#include "protobuf/mesh_graph_descriptor.pb.h"

namespace tt::tt_fabric {

std::optional<RoutingDirection> axis_edge_direction(
    const MeshGraph& mesh_graph, MeshId mesh_id, int axis, int ortho, int row_a, int row_b) {
    const auto coord = [&](int row) { return axis == 0 ? MeshCoordinate(row, ortho) : MeshCoordinate(ortho, row); };
    const auto& conn = mesh_graph.get_intra_mesh_connectivity()[*mesh_id];
    const ChipId chip_a = mesh_graph.coordinate_to_chip(mesh_id, coord(row_a));
    const auto it = conn[chip_a].find(mesh_graph.coordinate_to_chip(mesh_id, coord(row_b)));
    return it == conn[chip_a].end() ? std::nullopt : std::optional{it->second.port_direction};
}

namespace {

constexpr int kNone = ExpressRingTopology::kNone;

struct Pattern {
    int step = 0;
    int start = 0;
    // Declared tiling wrap, unset when the descriptor leaves it to the dimension.
    std::optional<bool> declared_wrap;
    bool wraps = false;
    std::vector<std::pair<int, int>> blocks;  // {first, last} row of each block, in block order
};

// Block tiling for one declared pattern. Mirrors expand_express_link_edges in mesh_graph.cpp: a block
// straddling the boundary wraps when the pattern wraps and is dropped otherwise.
void fill_blocks(Pattern& pattern, MeshId mesh_id, int len) {
    TT_FATAL(
        pattern.step >= 2 && len % pattern.step == 0,
        "ExpressRingTopology: mesh M{} express step {} does not tile axis length {}",
        *mesh_id,
        pattern.step,
        len);
    for (int k = 0; k < len / pattern.step; k++) {
        const int first = pattern.start + k * pattern.step;
        const int last = first + pattern.step - 1;
        if (!pattern.wraps && last >= len) {
            continue;
        }
        pattern.blocks.emplace_back(first % len, last % len);
    }
    TT_FATAL(
        !pattern.blocks.empty(),
        "ExpressRingTopology: mesh M{} express pattern (start {}, step {}) tiles no block",
        *mesh_id,
        pattern.start,
        pattern.step);
}

// Declared express-link patterns for one mesh, ascending by step. Empty when the mesh declares none.
std::vector<Pattern> read_patterns(const MeshGraph& mesh_graph, MeshId mesh_id, int& axis) {
    std::vector<Pattern> patterns;
    if (!mesh_graph.get_mesh_graph_descriptor_path().has_value()) {
        return patterns;
    }
    const auto& mgd = mesh_graph.get_mesh_graph_descriptor();
    const proto::MeshDescriptor* desc = nullptr;
    for (const auto& id : mgd.all_meshes()) {
        const auto& instance = mgd.get_instance(id);
        if (instance.local_id == *mesh_id && std::holds_alternative<const proto::MeshDescriptor*>(instance.desc)) {
            desc = std::get<const proto::MeshDescriptor*>(instance.desc);
            break;
        }
    }
    if (desc == nullptr) {
        return patterns;
    }
    for (const auto& express : desc->express_links()) {
        const int declared_axis = static_cast<int>(express.dim_idx());
        TT_FATAL(
            axis == kNone || axis == declared_axis,
            "ExpressRingTopology: mesh M{} declares express links on more than one dimension",
            *mesh_id);
        axis = declared_axis;
        Pattern pattern;
        pattern.step = static_cast<int>(express.pattern().step());
        pattern.start = static_cast<int>(express.pattern().start());
        if (express.wrap() != proto::TorusTopology::INVALID_TYPE) {
            pattern.declared_wrap = express.wrap() == proto::TorusTopology::RING;
        }
        patterns.push_back(std::move(pattern));
    }
    std::sort(patterns.begin(), patterns.end(), [](const Pattern& a, const Pattern& b) { return a.step < b.step; });
    return patterns;
}

// Whether the axis closes at the mesh level. Only an ORDINARY end edge counts: a chord may also join
// the first and last row (a block of width len does exactly that) and that is not a cardinal wrap.
bool axis_wraps(const MeshGraph& mesh_graph, MeshId mesh_id, int axis, int len) {
    if (len <= 2) {
        return false;
    }
    const auto dir = axis_edge_direction(mesh_graph, mesh_id, axis, 0, 0, len - 1);
    return dir.has_value() && *dir != RoutingDirection::Z;
}

std::vector<int> canonicalize(std::vector<int> cycle) {
    std::rotate(cycle.begin(), std::min_element(cycle.begin(), cycle.end()), cycle.end());
    std::vector<int> reflected{cycle.front()};
    reflected.insert(reflected.end(), cycle.rbegin(), cycle.rend() - 1);
    return std::min(cycle, reflected);
}

// RING axis: the ring is the rows this pattern does not skip, in coordinate order. Consecutive members
// are joined either by an ordinary edge or by the chord across the block between them, and the axis
// wrap closes it. Block endpoints alone would only work where the blocks tile the whole axis.
std::vector<int> pattern_cycle(const Pattern& pattern, int len) {
    std::vector<bool> interior(len, false);
    for (const auto& [first, last] : pattern.blocks) {
        for (int row = (first + 1) % len; row != last; row = (row + 1) % len) {
            interior[row] = true;
        }
    }
    std::vector<int> cycle;
    for (int row = 0; row < len; row++) {
        if (!interior[row]) {
            cycle.push_back(row);
        }
    }
    return cycle;
}

// LINE axis: with no mesh-level wrap no pattern can close on its own, so the classes fuse into one
// ring -- up through the small-step endpoints, across an ordinary run, back down the large-step ones.
std::vector<int> merged_cycle(const Pattern& small, const Pattern& large) {
    const int head = large.blocks.front().first;
    std::vector<int> cycle{head};
    for (int r = head + 1; r < small.blocks.front().first; r++) {
        cycle.push_back(r);
    }
    for (const auto& [first, last] : small.blocks) {
        cycle.push_back(first);
        cycle.push_back(last);
    }
    for (int r = small.blocks.back().second + 1; r < large.blocks.back().second; r++) {
        cycle.push_back(r);
    }
    for (int i = static_cast<int>(large.blocks.size()) - 1; i >= 0; i--) {
        cycle.push_back(large.blocks[i].second);
        if (large.blocks[i].first != head) {
            cycle.push_back(large.blocks[i].first);
        }
    }
    return cycle;
}

}  // namespace

int ExpressRingTopology::ring_distance(int domain, int from, int to) const {
    const int n = static_cast<int>(forward_cycle[domain].size());
    const int d = (pos_in_domain[to] - pos_in_domain[from] + n) % n;
    return std::min(d, n - d);
}

int ExpressRingTopology::next_row(int src, int dst) const {
    const auto step = [&](int domain, int from, int to) {
        const int n = static_cast<int>(forward_cycle[domain].size());
        const int p = pos_in_domain[from];
        const int fwd = (pos_in_domain[to] - p + n) % n;
        return fwd <= n - fwd ? forward_cycle[domain][(p + 1) % n] : forward_cycle[domain][(p - 1 + n) % n];
    };

    // A run is left, and entered, by its nearer end -- the half of the run closer to that anchor.
    // The end is a function of the leaf alone, so every suffix of the route agrees on it.
    const auto nearer_end_is_before = [&](int leaf) {
        const auto& run = leaf_runs[leaf_run_of[leaf]];
        return leaf_index_of[leaf] < static_cast<int>(run.rows.size()) - 1 - leaf_index_of[leaf];
    };
    // Same-run pairs on the same side of the run's midpoint walk the leaf links directly. Pairs that
    // straddle the midpoint do NOT: routing them across the middle would chain the run's forward edges
    // into a cycle with the anchors' chord (an unprotected loop). They fall through to egress+ingress
    // instead, exiting to one anchor and re-entering from the other, which leaves the middle edge
    // unused and keeps the run cycle-free.
    if (is_leaf(src) && leaf_run_of[src] == leaf_run_of[dst] &&
        nearer_end_is_before(src) == nearer_end_is_before(dst)) {
        const auto& run = leaf_runs[leaf_run_of[src]];
        const int i = leaf_index_of[src];
        return run.rows[leaf_index_of[dst] > i ? i + 1 : i - 1];
    }
    if (is_leaf(src)) {
        const auto& run = leaf_runs[leaf_run_of[src]];
        const int i = leaf_index_of[src];
        if (nearer_end_is_before(src)) {
            return i == 0 ? run.anchor_before : run.rows[i - 1];
        }
        return i + 1 < static_cast<int>(run.rows.size()) ? run.rows[i + 1] : run.anchor_after;
    }
    if (is_leaf(dst)) {
        const auto& run = leaf_runs[leaf_run_of[dst]];
        const bool from_before = nearer_end_is_before(dst);
        const int entry = from_before ? run.anchor_before : run.anchor_after;
        return src == entry ? (from_before ? run.rows.front() : run.rows.back()) : next_row(src, entry);
    }
    const int src_domain = domain_of[src];
    const int dst_domain = domain_of[dst];
    if (src_domain == dst_domain) {
        return step(src_domain, src, dst);
    }
    if (src_domain == continue_src_domain) {
        // Late exit: the crossover landing closest to the destination in its own family wins.
        std::size_t best = 0;
        std::pair<int, int> best_key{std::numeric_limits<int>::max(), 0};
        for (std::size_t i = 0; i < crossovers.size(); i++) {
            const auto& [a, b] = crossovers[i];
            const int dst_dist = ring_distance(dst_domain, b, dst);
            const std::pair<int, int> key{dst_dist, ring_distance(src_domain, src, a) + 1 + dst_dist};
            if (key < best_key) {
                best_key = key;
                best = i;
            }
        }
        const auto& [a, b] = crossovers[best];
        return src == a ? b : step(src_domain, src, a);
    }
    // Terminal landing: cross only at the crossover paired with the destination.
    for (const auto& [a, b] : crossovers) {
        if (a == dst) {
            return src == b ? dst : step(src_domain, src, b);
        }
    }
    TT_THROW("ExpressRingTopology: no paired landing for row {} from row {}", dst, src);
}

std::vector<std::pair<int, int>> ExpressRingTopology::cyclic_non_ring_hops() const {
    const int len = axis_len;
    const auto hop_id = [len](int a, int b) { return a * len + b; };
    const auto is_ring_edge = [&](int a, int b) {
        return !is_leaf(a) && !is_leaf(b) && domain_of[a] == domain_of[b] && ring_distance(domain_of[a], a, b) == 1;
    };

    // Control-dependency graph over directed row-hops: an edge hop -> successor for every pair of
    // consecutive hops on a route. Homogeneity (checked separately) makes one column representative,
    // so this is built purely on rows.
    std::vector<std::vector<int>> succ(len * len);
    std::vector<char> used(len * len, 0);
    for (int s = 0; s < len; s++) {
        for (int d = 0; d < len; d++) {
            if (s == d) {
                continue;
            }
            int cur = s;
            int prev = -1;
            for (int guard = 0; cur != d && guard <= len; guard++) {
                const int nxt = next_row(cur, d);
                const int h = hop_id(cur, nxt);
                used[h] = 1;
                if (prev >= 0) {
                    succ[prev].push_back(h);
                }
                prev = h;
                cur = nxt;
            }
        }
    }

    // Tarjan SCC over the hop graph; a hop in a nontrivial SCC takes part in a dependency cycle.
    std::vector<int> index(len * len, -1);
    std::vector<int> low(len * len, 0);
    std::vector<char> on_stack(len * len, 0);
    std::vector<int> stk;
    int next_index = 0;
    std::vector<std::pair<int, int>> bad;

    const std::function<void(int)> strongconnect = [&](int v) {
        index[v] = low[v] = next_index++;
        stk.push_back(v);
        on_stack[v] = 1;
        for (int w : succ[v]) {
            if (index[w] == -1) {
                strongconnect(w);
                low[v] = std::min(low[v], low[w]);
            } else if (on_stack[w]) {
                low[v] = std::min(low[v], index[w]);
            }
        }
        if (low[v] == index[v]) {
            std::vector<int> component;
            int w = -1;
            do {
                w = stk.back();
                stk.pop_back();
                on_stack[w] = 0;
                component.push_back(w);
            } while (w != v);
            // Nontrivial = more than one hop, or a single hop that depends on itself.
            const bool nontrivial =
                component.size() > 1 || std::find(succ[v].begin(), succ[v].end(), v) != succ[v].end();
            if (nontrivial) {
                for (int h : component) {
                    if (!is_ring_edge(h / len, h % len)) {
                        bad.emplace_back(h / len, h % len);
                    }
                }
            }
        }
    };
    for (int v = 0; v < len * len; v++) {
        if (used[v] && index[v] == -1) {
            strongconnect(v);
        }
    }
    std::sort(bad.begin(), bad.end());
    return bad;
}

std::optional<ExpressRingTopology> derive_express_ring_topology(const MeshGraph& mesh_graph, MeshId mesh_id) {
    int axis = kNone;
    auto patterns = read_patterns(mesh_graph, mesh_id, axis);
    if (patterns.empty()) {
        return std::nullopt;  // no express links: base routing is unchanged
    }
    TT_FATAL(
        axis == 0,
        "ExpressRingTopology: mesh M{} declares express links along dimension {}; this cut supports dimension 0 only",
        *mesh_id,
        axis);
    TT_FATAL(
        patterns.size() <= 2,
        "ExpressRingTopology: mesh M{} declares {} express patterns; only one or two are defined",
        *mesh_id,
        patterns.size());

    const auto shape = mesh_graph.get_mesh_shape(mesh_id);
    const int len = static_cast<int>(shape[axis]);
    const int ortho_len = static_cast<int>(shape[1 - axis]);
    // Baseline fact: whether the ordinary grid closes the axis. Distinct from a pattern's tiling
    // wrap, which the overlay declares and which only decides which blocks exist.
    const bool wraps = axis_wraps(mesh_graph, mesh_id, axis, len);

    // Closing a ring needs either the ordinary axis wrap, which joins the ends of a single pattern's
    // class, or a second pattern whose blocks merge with the first into one cycle. With neither, this
    // mesh has no express decomposition: it routes on the base grid and the declared chords go unused.
    //
    // This is a configuration outcome, not a malformed descriptor, so it degrades rather than throws.
    // Connectivity is built from the effective fabric type, which a supplied FabricConfig replaces
    // outright (mesh_graph.cpp), and get_fabric_type() maps everything that is not an explicit torus
    // request to FabricType::MESH. A RING-declared axis therefore has no wrap edges whenever the
    // control plane is constructed at a non-torus config -- which every caller does at least once
    // before any torus config is selected, since the config change is what rebuilds the control plane.
    if (!wraps && patterns.size() != 2) {
        log_warning(
            tt::LogFabric,
            "Mesh M{} declares express links but its dim-{} axis has no end wrap under the current fabric "
            "config, and one pattern cannot close a ring on its own. Express routing is off for this mesh; "
            "it routes on the base grid. Select a torus fabric config along that axis to enable it.",
            *mesh_id,
            axis);
        return std::nullopt;
    }

    for (auto& pattern : patterns) {
        pattern.wraps = pattern.declared_wrap.value_or(wraps);
        fill_blocks(pattern, mesh_id, len);
    }

    ExpressRingTopology topo;
    topo.axis_dim = axis;
    topo.axis_len = len;
    topo.domain_of.assign(len, kNone);
    topo.leaf_run_of.assign(len, kNone);
    topo.leaf_index_of.assign(len, kNone);
    topo.pos_in_domain.assign(len, kNone);

    std::vector<bool> endpoint(len, false);
    for (const auto& pattern : patterns) {
        for (const auto& [first, last] : pattern.blocks) {
            endpoint[first] = true;
            endpoint[last] = true;
        }
    }

    // Leaves are the smallest pattern's skipped rows, except where a wider chord claims one as an
    // endpoint. How many rows a block skips follows from its step.
    const Pattern& leaf_pattern = patterns.front();
    std::vector<bool> leaf(len, false);
    for (const auto& block : leaf_pattern.blocks) {
        for (int j = 1; j <= leaf_pattern.step - 2; j++) {
            const int row = (block.first + j) % len;
            if (!endpoint[row]) {
                leaf[row] = true;
            }
        }
    }
    // Group the skipped rows into maximal runs. A run needs a transit row either side to attach to.
    const auto neighbour = [&](int row, int delta) {
        const int peer = row + delta;
        return (peer < 0 || peer >= len) ? (wraps ? ((peer % len) + len) % len : kNone) : peer;
    };
    for (int row = 0; row < len; row++) {
        const int before = neighbour(row, -1);
        if (!leaf[row] || (before != kNone && leaf[before])) {
            continue;  // not a leaf, or not the first row of its run
        }
        ExpressRingTopology::LeafRun run;
        run.anchor_before = before;
        for (int r = row; r != kNone && leaf[r] && static_cast<int>(run.rows.size()) < len; r = neighbour(r, 1)) {
            topo.leaf_run_of[r] = static_cast<int>(topo.leaf_runs.size());
            topo.leaf_index_of[r] = static_cast<int>(run.rows.size());
            run.rows.push_back(r);
            run.anchor_after = neighbour(r, 1);
        }
        TT_FATAL(
            run.anchor_before != kNone && run.anchor_after != kNone && !leaf[run.anchor_before] &&
                !leaf[run.anchor_after],
            "ExpressRingTopology: mesh M{} skipped run starting at row {} has no transit row on both sides",
            *mesh_id,
            row);
        topo.leaf_runs.push_back(std::move(run));
    }

    std::vector<std::vector<int>> cycles;
    if (wraps) {
        for (const auto& pattern : patterns) {
            cycles.push_back(canonicalize(pattern_cycle(pattern, len)));
        }
    } else {
        // Two patterns, guaranteed by the early return above: without the axis wrap the only way to
        // close a cycle is to merge both patterns' blocks.
        cycles.push_back(canonicalize(merged_cycle(patterns.front(), patterns.back())));
    }
    for (std::size_t domain = 0; domain < cycles.size(); domain++) {
        for (std::size_t p = 0; p < cycles[domain].size(); p++) {
            const int row = cycles[domain][p];
            TT_FATAL(
                topo.domain_of[row] == kNone && topo.leaf_run_of[row] == kNone,
                "ExpressRingTopology: mesh M{} row {} is a leaf or already in another ring",
                *mesh_id,
                row);
            topo.domain_of[row] = static_cast<int>(domain);
            topo.pos_in_domain[row] = static_cast<int>(p);
        }
    }
    topo.forward_cycle = cycles;
    for (int row = 0; row < len; row++) {
        TT_FATAL(
            (topo.domain_of[row] == kNone) == (topo.leaf_run_of[row] != kNone),
            "ExpressRingTopology: mesh M{} row {} is neither a ring member nor a leaf",
            *mesh_id,
            row);
    }

    if (cycles.size() == 2) {
        // The wider class continues into the narrower one; the reverse crossing is terminal.
        topo.continue_src_domain = patterns[0].step > patterns[1].step ? 0 : 1;
        for (int row = 0; row < len; row++) {
            const int next = (row + 1) % len;
            if ((!wraps && next == 0) || topo.is_leaf(row) || topo.is_leaf(next) ||
                topo.domain_of[row] == topo.domain_of[next]) {
                continue;
            }
            const bool row_continues = topo.domain_of[row] == topo.continue_src_domain;
            topo.crossovers.emplace_back(row_continues ? row : next, row_continues ? next : row);
        }
        std::sort(topo.crossovers.begin(), topo.crossovers.end(), [&](const auto& x, const auto& y) {
            return std::pair{topo.pos_in_domain[x.first], x} < std::pair{topo.pos_in_domain[y.first], y};
        });
        std::vector<int> landings;
        for (const auto& [a, b] : topo.crossovers) {
            landings.push_back(a);
        }
        std::vector<int> expected = topo.forward_cycle[topo.continue_src_domain];
        std::sort(landings.begin(), landings.end());
        std::sort(expected.begin(), expected.end());
        TT_FATAL(
            landings == expected,
            "ExpressRingTopology: mesh M{} has {} paired landings for {} continuing-family members",
            *mesh_id,
            landings.size(),
            expected.size());
    }

    // Everything above is arithmetic on the declared patterns, so confirm the mesh actually carries
    // the edges it implies -- on every line, which is also the row/column uniformity precondition.
    const auto require_edge = [&](int ortho, int row_a, int row_b, const char* what) {
        TT_FATAL(
            axis_edge_direction(mesh_graph, mesh_id, axis, ortho, row_a, row_b).has_value(),
            "ExpressRingTopology: mesh M{} line {} is missing the {} edge {}-{}",
            *mesh_id,
            ortho,
            what,
            row_a,
            row_b);
    };
    int declared_chords = 0;
    for (const auto& pattern : patterns) {
        declared_chords += static_cast<int>(pattern.blocks.size());
    }
    for (int ortho = 0; ortho < ortho_len; ortho++) {
        for (const auto& cycle : topo.forward_cycle) {
            for (std::size_t p = 0; p < cycle.size(); p++) {
                require_edge(ortho, cycle[p], cycle[(p + 1) % cycle.size()], "ring");
            }
        }
        for (const auto& [a, b] : topo.crossovers) {
            require_edge(ortho, a, b, "crossover");
        }
        for (const auto& run : topo.leaf_runs) {
            require_edge(ortho, run.anchor_before, run.rows.front(), "anchor");
            require_edge(ortho, run.rows.back(), run.anchor_after, "anchor");
            for (std::size_t i = 1; i < run.rows.size(); i++) {
                require_edge(ortho, run.rows[i - 1], run.rows[i], "leaf-run");
            }
        }
        int chords = 0;
        for (int row = 0; row < len; row++) {
            for (int peer = row + 1; peer < len; peer++) {
                const auto dir = axis_edge_direction(mesh_graph, mesh_id, axis, ortho, row, peer);
                if (dir.has_value() && *dir == RoutingDirection::Z) {
                    chords++;
                }
            }
        }
        TT_FATAL(
            chords == declared_chords,
            "ExpressRingTopology: mesh M{} line {} carries {} chords but the declared patterns imply {}",
            *mesh_id,
            ortho,
            chords,
            declared_chords);
    }

    return topo;
}

std::optional<ExpressRingTopology> derive_ordinary_ring_topology(
    const MeshGraph& mesh_graph, MeshId mesh_id, int axis) {
    const auto shape = mesh_graph.get_mesh_shape(mesh_id);
    const int len = static_cast<int>(shape[axis]);
    if (!axis_wraps(mesh_graph, mesh_id, axis, len)) {
        return std::nullopt;
    }

    ExpressRingTopology topo;
    topo.axis_dim = axis;
    topo.axis_len = len;
    topo.domain_of.assign(len, 0);
    topo.leaf_run_of.assign(len, kNone);
    topo.leaf_index_of.assign(len, kNone);
    topo.pos_in_domain.resize(len);
    topo.forward_cycle.emplace_back();
    for (int coord = 0; coord < len; coord++) {
        topo.pos_in_domain[coord] = coord;
        topo.forward_cycle.front().push_back(coord);
    }

    for (int ortho = 0; ortho < static_cast<int>(shape[1 - axis]); ortho++) {
        for (int coord = 0; coord < len; coord++) {
            const int next = (coord + 1) % len;
            TT_FATAL(
                axis_edge_direction(mesh_graph, mesh_id, axis, ortho, coord, next).has_value(),
                "Mesh M{} dim {} line {} is missing the ordinary edge {}-{}",
                *mesh_id,
                axis,
                ortho,
                coord,
                next);
        }
    }
    return topo;
}

}  // namespace tt::tt_fabric
