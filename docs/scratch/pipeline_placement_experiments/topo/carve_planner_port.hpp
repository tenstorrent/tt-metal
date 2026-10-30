// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// A C++ port of tt-blaze's ring carve planner (pipeline_builder/carve_planner.py, plan_ring_carve) for the
// benchmarks: the design's section 3.2 ("port the carve planner to C++") does not exist in the repo yet, so
// this is the reference for what that port would cost per solve. Same algorithm, same orders, same budgets:
//
//   phase 2  prune slices of degree < 2, contract degree-2 runs into chains between branch slices
//   phase 3  depth-first search for simple cycles of exactly `need` slices over the contracted graph,
//            bounded by kMaxSearchExpansions expansions and kMaxCandidateCycles cycles
//   phase 4  sort the cycles by mesh crossings then slice keys; for each, try the canonical walk, its
//            reverse, and every rotation, cutting consecutive runs into the stage rectangles
//
// First fit: the first cycle and alignment that groups is the plan. Slices are in slice units (every
// HostSlice is 1x1), which is what blaze's uniform-slice check reduces to.
//
// Differences from the Python, all deliberate: the plan does not append the filler slots for unused
// slices (the harness adds them where blaze would), and the "stage 0 must be one slice" rule can be
// switched off (allow_wide_stage0) to measure the algorithm on rings whose first stage is wider.
#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace carve_planner {

constexpr long kMaxSearchExpansions = 200000;
constexpr size_t kMaxCandidateCycles = 64;

struct HostSlice {
    int mesh_id, row0, col0, nrows = 1, ncols = 1;
    std::tuple<int, int, int> key() const { return {mesh_id, row0, col0}; }
    bool operator<(const HostSlice& o) const { return key() < o.key(); }
};

struct Rect {
    int mesh_id, row0, col0, rows, cols;
};

struct Chain {
    int a, b;
    std::vector<int> interior;  // strictly between a and b, in a->b order
    size_t weight() const { return interior.size(); }
};

struct PlanResult {
    bool ok = false;
    std::string phase;        // on failure: input-infeasible | no-cycle-of-length-N | budget-exhausted | no-alignment
    std::vector<Rect> slots;  // stage order
    long expansions = 0;
    size_t cycles = 0;
};

using Adj = std::vector<std::set<int>>;

inline std::set<int> prune_leaves(int n, const Adj& adj) {
    std::set<int> alive;
    for (int v = 0; v < n; ++v) {
        alive.insert(v);
    }
    bool changed = true;
    while (changed) {
        changed = false;
        for (int v : std::vector<int>(alive.begin(), alive.end())) {
            size_t deg = 0;
            for (int w : adj[v]) {
                deg += alive.count(w);
            }
            if (deg < 2) {
                alive.erase(v);
                changed = true;
            }
        }
    }
    return alive;
}

inline std::vector<int> live_neighbours(int v, const Adj& adj, const std::set<int>& alive) {
    std::vector<int> out;
    for (int w : adj[v]) {
        if (alive.count(w)) {
            out.push_back(w);
        }
    }
    return out;  // sorted: std::set iteration
}

inline std::pair<std::vector<int>, std::vector<Chain>> contract(const std::set<int>& alive, const Adj& adj) {
    std::vector<int> branch;
    for (int v : alive) {
        if (live_neighbours(v, adj, alive).size() > 2) {
            branch.push_back(v);
        }
    }
    std::vector<Chain> chains;
    if (branch.empty()) {
        return {branch, chains};
    }
    const std::set<int> branch_set(branch.begin(), branch.end());
    std::set<std::pair<int, int>> seen;
    for (int a : branch) {
        for (int first : live_neighbours(a, adj, alive)) {
            if (seen.count({a, first})) {
                continue;
            }
            std::vector<int> interior;
            int prev = a, cur = first;
            while (!branch_set.count(cur)) {
                interior.push_back(cur);
                std::vector<int> nxt;
                for (int w : live_neighbours(cur, adj, alive)) {
                    if (w != prev) {
                        nxt.push_back(w);
                    }
                }
                if (nxt.empty()) {
                    break;
                }
                prev = cur;
                cur = nxt[0];
            }
            const int b = cur;
            seen.insert({a, first});
            const int back = interior.empty() ? a : interior.back();
            seen.insert({b, back});
            chains.push_back({a, b, interior});
        }
    }
    return {branch, chains};
}

inline std::vector<int> expand(
    const std::vector<int>& branch_path, const std::vector<int>& chain_path, const std::vector<Chain>& chains) {
    std::vector<int> out;
    for (size_t i = 0; i < branch_path.size(); ++i) {
        const int a = branch_path[i];
        const Chain& ch = chains[chain_path[i]];
        out.push_back(a);
        if (ch.a == a) {
            out.insert(out.end(), ch.interior.begin(), ch.interior.end());
        } else {
            out.insert(out.end(), ch.interior.rbegin(), ch.interior.rend());
        }
    }
    return out;
}

struct CycleSearch {
    size_t target;
    const std::vector<int>& branch;
    const std::vector<Chain>& chains;
    std::map<int, std::vector<int>> incident;
    std::set<int> order;
    std::vector<std::vector<int>> cycles;
    std::set<size_t> lengths;
    long expansions = 0;

    CycleSearch(size_t t, const std::vector<int>& b, const std::vector<Chain>& c) : target(t), branch(b), chains(c) {
        for (int v : branch) {
            incident[v];
            order.insert(v);
        }
        for (size_t i = 0; i < chains.size(); ++i) {
            incident[chains[i].a].push_back(int(i));
            if (chains[i].b != chains[i].a) {
                incident[chains[i].b].push_back(int(i));
            }
        }
        for (auto& [v, list] : incident) {
            std::sort(list.begin(), list.end());
        }
    }

    void walk(
        int start,
        int head,
        std::set<int>& used_chains,
        std::set<int>& visited,
        std::vector<int>& path,
        std::vector<int>& chain_path,
        size_t total) {
        if (expansions > kMaxSearchExpansions || cycles.size() >= kMaxCandidateCycles) {
            return;
        }
        ++expansions;
        for (int ci : incident[head]) {
            if (used_chains.count(ci)) {
                continue;
            }
            const Chain& ch = chains[ci];
            const int nxt = ch.a == head ? ch.b : ch.a;
            const bool closing = nxt == start;
            if (!closing && (!order.count(nxt) || visited.count(nxt))) {
                continue;
            }
            const size_t new_total = total + ch.weight() + (closing ? 0 : 1);
            if (new_total > target) {
                continue;
            }
            if (closing) {
                if (new_total == target && !path.empty()) {
                    chain_path.push_back(ci);
                    cycles.push_back(expand(path, chain_path, chains));
                    chain_path.pop_back();
                }
                lengths.insert(new_total);
                continue;
            }
            used_chains.insert(ci);
            visited.insert(nxt);
            path.push_back(nxt);
            chain_path.push_back(ci);
            walk(start, nxt, used_chains, visited, path, chain_path, new_total);
            chain_path.pop_back();
            path.pop_back();
            visited.erase(nxt);
            used_chains.erase(ci);
        }
    }
};

inline void cycles_no_branch(
    size_t target,
    const std::set<int>& alive,
    const Adj& adj,
    std::vector<std::vector<int>>& cycles,
    std::set<size_t>& lengths) {
    std::set<int> unvisited(alive);
    while (!unvisited.empty()) {
        const int start = *unvisited.begin();
        std::vector<int> seq{start};
        unvisited.erase(start);
        int prev = -1, cur = start;
        while (true) {
            std::vector<int> nxt;
            for (int w : live_neighbours(cur, adj, alive)) {
                if (w != prev) {
                    nxt.push_back(w);
                }
            }
            if (nxt.empty() || nxt[0] == start) {
                break;
            }
            seq.push_back(nxt[0]);
            unvisited.erase(nxt[0]);
            prev = cur;
            cur = nxt[0];
        }
        lengths.insert(seq.size());
        if (seq.size() == target) {
            cycles.push_back(seq);
        }
    }
}

inline std::optional<Rect> as_rectangle(const std::vector<HostSlice>& group) {
    if (group.empty()) {
        return std::nullopt;
    }
    std::set<int> meshes, rows, cols;
    std::set<std::pair<int, int>> cells;
    const int nrows = group[0].nrows, ncols = group[0].ncols;
    for (const auto& s : group) {
        if (s.nrows != nrows || s.ncols != ncols) {
            return std::nullopt;
        }
        meshes.insert(s.mesh_id);
        rows.insert(s.row0);
        cols.insert(s.col0);
        cells.insert({s.row0, s.col0});
    }
    if (meshes.size() != 1 || group.size() != rows.size() * cols.size() || cells.size() != group.size()) {
        return std::nullopt;
    }
    for (auto it = rows.begin(); std::next(it) != rows.end(); ++it) {
        if (*std::next(it) - *it != nrows) {
            return std::nullopt;
        }
    }
    for (auto it = cols.begin(); std::next(it) != cols.end(); ++it) {
        if (*std::next(it) - *it != ncols) {
            return std::nullopt;
        }
    }
    return Rect{group[0].mesh_id, *rows.begin(), *cols.begin(), int(rows.size()) * nrows, int(cols.size()) * ncols};
}

inline std::vector<int> canonical_walk(const std::vector<int>& cycle, const std::vector<HostSlice>& slices) {
    const size_t n = cycle.size();
    size_t start = 0;
    for (size_t i = 1; i < n; ++i) {
        if (slices[cycle[i]].key() < slices[cycle[start]].key()) {
            start = i;
        }
    }
    std::vector<int> seq(cycle.begin() + start, cycle.end());
    seq.insert(seq.end(), cycle.begin(), cycle.begin() + start);
    if (n > 1 && slices[seq[1]].key() > slices[seq.back()].key()) {
        std::reverse(seq.begin() + 1, seq.end());
    }
    return seq;
}

inline std::optional<std::vector<Rect>> group_cycle(
    const std::vector<int>& cycle,
    const std::vector<HostSlice>& slices,
    const std::vector<std::pair<int, int>>& stage_shapes,
    const std::vector<size_t>& stage_counts) {
    const size_t n = cycle.size();
    const std::vector<int> canonical = canonical_walk(cycle, slices);
    const std::vector<int> reversed(canonical.rbegin(), canonical.rend());
    for (const auto* seq : {&canonical, &reversed}) {
        for (size_t offset = 0; offset < n; ++offset) {
            std::vector<int> rotated(seq->begin() + offset, seq->end());
            rotated.insert(rotated.end(), seq->begin(), seq->begin() + offset);
            std::vector<Rect> placements;
            size_t pos = 0;
            bool ok = true;
            for (size_t k = 0; k < stage_shapes.size(); ++k) {
                std::vector<HostSlice> group;
                for (size_t i = pos; i < pos + stage_counts[k] && i < rotated.size(); ++i) {
                    group.push_back(slices[rotated[i]]);
                }
                pos += stage_counts[k];
                const auto rect = as_rectangle(group);
                if (!rect || rect->rows != stage_shapes[k].first || rect->cols != stage_shapes[k].second) {
                    ok = false;
                    break;
                }
                placements.push_back(*rect);
            }
            if (ok) {
                return placements;
            }
        }
    }
    return std::nullopt;
}

// slices: every host slice (slice units, 1x1 each); adjacency: unordered slice-index pairs;
// stage_shapes: (rows, cols) in slices, ring order.
inline PlanResult plan_ring_carve(
    std::vector<HostSlice> slices,
    const std::vector<std::pair<int, int>>& adjacency,
    const std::vector<std::pair<int, int>>& stage_shapes,
    bool allow_wide_stage0 = false) {
    PlanResult res;
    std::sort(slices.begin(), slices.end());
    const int n = int(slices.size());
    std::vector<size_t> stage_counts;
    size_t need = 0;
    for (const auto& [r, c] : stage_shapes) {
        stage_counts.push_back(size_t(r) * c);
        need += size_t(r) * c;
    }
    if (need > size_t(n) || (!allow_wide_stage0 && !stage_counts.empty() && stage_counts[0] != 1)) {
        res.phase = "input-infeasible";
        return res;
    }
    Adj adj(n);
    for (auto [a, b] : adjacency) {
        adj[a].insert(b);
        adj[b].insert(a);
    }
    const std::set<int> alive = prune_leaves(n, adj);
    auto [branch, chains] = contract(alive, adj);
    if (need > alive.size()) {
        res.phase = "no-cycle-of-length-" + std::to_string(need);
        return res;
    }
    std::vector<std::vector<int>> cycles;
    std::set<size_t> lengths;
    long expansions = 0;
    if (branch.empty()) {
        cycles_no_branch(need, alive, adj, cycles, lengths);
    } else {
        CycleSearch search(need, branch, chains);
        for (int s : branch) {
            std::set<int> used, visited{s};
            std::vector<int> path{s}, chain_path;
            search.walk(s, s, used, visited, path, chain_path, 1);
        }
        cycles = std::move(search.cycles);
        lengths = std::move(search.lengths);
        expansions = search.expansions;
    }
    res.expansions = expansions;
    res.cycles = cycles.size();
    if (expansions > kMaxSearchExpansions) {
        res.phase = "budget-exhausted";
        return res;
    }
    if (cycles.empty()) {
        res.phase = "no-cycle-of-length-" + std::to_string(need);
        return res;
    }
    auto crossings = [&](const std::vector<int>& cycle) {
        int c = 0;
        for (size_t i = 0; i < cycle.size(); ++i) {
            c += slices[cycle[i]].mesh_id != slices[cycle[(i + 1) % cycle.size()]].mesh_id;
        }
        return c;
    };
    auto key_of = [&](const std::vector<int>& cycle) {
        std::vector<std::tuple<int, int, int>> k;
        for (int i : cycle) {
            k.push_back(slices[i].key());
        }
        return k;
    };
    std::stable_sort(cycles.begin(), cycles.end(), [&](const auto& x, const auto& y) {
        const int cx = crossings(x), cy = crossings(y);
        return cx != cy ? cx < cy : key_of(x) < key_of(y);
    });
    for (const auto& cycle : cycles) {
        if (auto placements = group_cycle(cycle, slices, stage_shapes, stage_counts)) {
            res.ok = true;
            res.slots = std::move(*placements);
            return res;
        }
    }
    res.phase = "no-alignment";
    return res;
}

}  // namespace carve_planner
