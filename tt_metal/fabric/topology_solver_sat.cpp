// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include "topology_solver_sat_solver.hpp"
#include <tt-metalium/experimental/fabric/topology_solver.hpp>

namespace tt::tt_fabric::detail {

static constexpr int kHostCapConflictBudget = 300'000;

struct SatSearchBackend::Impl {
    TopologySatSolver solver;
    TopologySatHardEncoding enc;
    bool cap_active = false;
    size_t solve_calls = 0;
    int symmetry_lit = 0;
    int preferred_lit = 0;
    int minimize_lit = 0;
    std::vector<std::vector<int>> stages;
    size_t stage = 0;
    bool unique_shapes = false;
};

SatSearchBackend::SatSearchBackend() = default;
SatSearchBackend::~SatSearchBackend() = default;
SatSearchBackend::SatSearchBackend(SatSearchBackend&&) noexcept = default;
SatSearchBackend& SatSearchBackend::operator=(SatSearchBackend&&) noexcept = default;

void SatSearchBackend::reset() { impl_.reset(); }

// ── Adjacency and Edge Helpers ────────────────────────────────────────────────
namespace {

bool are_globals_adjacent(const TopologySatGraphView& graph_data, size_t global_i, size_t global_j) {
    if (global_i >= graph_data.n_global || global_j >= graph_data.n_global) {
        return false;
    }
    const auto& adj = graph_data.global_adj_idx[global_i];
    return std::binary_search(adj.begin(), adj.end(), global_j);
}

bool topology_sat_check_edge_feasibility(
    const TopologySatGraphView& graph_data, const std::vector<int>& mapping, size_t target_idx, size_t global_idx) {
    for (size_t tn : graph_data.target_adj_idx[target_idx]) {
        if (mapping[tn] < 0) {
            continue;
        }
        const size_t gg = static_cast<size_t>(mapping[tn]);
        if (!are_globals_adjacent(graph_data, global_idx, gg)) {
            return false;
        }
    }
    return true;
}

// ── Preferred-Hit Bound Helpers ───────────────────────────────────────────────

size_t topology_sat_preferred_upper_bound(
    const TopologySatConstraintView& constraint_data,
    const TopologySatHardEncoding& enc,
    const std::vector<bool>& used_global,
    size_t ti_start,
    size_t n_target) {
    size_t out = 0;
    for (size_t u = ti_start; u < n_target; ++u) {
        if (u >= constraint_data.preferred_global_indices.size()) {
            continue;
        }
        const auto& pref = constraint_data.preferred_global_indices[u];
        if (pref.empty()) {
            continue;
        }
        bool can = false;
        for (size_t g : enc.allowed_global_idx[u]) {
            if (g >= used_global.size() || used_global[g]) {
                continue;
            }
            if (std::binary_search(pref.begin(), pref.end(), g)) {
                can = true;
                break;
            }
        }
        if (can) {
            ++out;
        }
    }
    return out;
}

// Lower bound on maximum simultaneously satisfiable preferred targets, using the same per-target allowed globals as
// the SAT encoding.  Explores partial assignments with edge checks + injective constraint; pruning uses a simple
// upper bound on remaining preferred-capable targets.  When max_nodes is huge (n_target <= kExactLbMaxTargets), this
// becomes an exhaustive search -> exact optimum for small instances; otherwise it stops after max_nodes expansions and
// returns the best complete mapping found (still a safe lower bound for at-least-k).
size_t topology_sat_preferred_exact_lower_bound(
    const TopologySatGraphView& graph_data,
    const TopologySatConstraintView& constraint_data,
    const TopologySatHardEncoding& enc,
    size_t max_nodes) {
    const size_t nt = graph_data.n_target;
    std::vector<int> mapping(nt, -1);
    std::vector<bool> used(graph_data.n_global, false);
    size_t best = 0;
    size_t explored = 0;

    const auto dfs = [&](auto&& self, size_t ti, size_t pref_so_far) -> void {
        if (explored >= max_nodes) {
            return;
        }
        ++explored;
        if (pref_so_far + topology_sat_preferred_upper_bound(constraint_data, enc, used, ti, nt) <= best) {
            return;
        }
        if (ti == nt) {
            best = std::max(best, pref_so_far);
            return;
        }

        struct Cand {
            size_t g;
            bool is_pref;
        };
        std::vector<Cand> cands;
        cands.reserve(enc.allowed_global_idx[ti].size());
        for (size_t g : enc.allowed_global_idx[ti]) {
            if (g >= used.size() || used[g]) {
                continue;
            }
            bool is_pref = false;
            if (ti < constraint_data.preferred_global_indices.size()) {
                const auto& pv = constraint_data.preferred_global_indices[ti];
                is_pref = !pv.empty() && std::binary_search(pv.begin(), pv.end(), g);
            }
            cands.push_back({g, is_pref});
        }
        std::stable_partition(cands.begin(), cands.end(), [](const Cand& c) { return c.is_pref; });

        for (const Cand& cand : cands) {
            const size_t g = cand.g;
            if (!topology_sat_check_edge_feasibility(graph_data, mapping, ti, g)) {
                continue;
            }
            mapping[ti] = static_cast<int>(g);
            used[g] = true;
            self(self, ti + 1, pref_so_far + (cand.is_pref ? 1u : 0u));
            used[g] = false;
            mapping[ti] = -1;
        }
    };

    dfs(dfs, 0, 0);
    return best;
}

size_t topology_sat_preferred_greedy_lower_bound(
    const TopologySatGraphView& graph_data,
    const TopologySatConstraintView& constraint_data,
    const TopologySatHardEncoding& enc) {
    const size_t nt = graph_data.n_target;
    std::vector<std::vector<size_t>> orders;
    orders.reserve(40);
    {
        std::vector<size_t> id(nt);
        std::iota(id.begin(), id.end(), 0);
        orders.push_back(std::move(id));
    }
    {
        std::vector<size_t> rev(nt);
        for (size_t i = 0; i < nt; ++i) {
            rev[i] = nt - 1 - i;
        }
        orders.push_back(std::move(rev));
    }
    {
        std::vector<size_t> inter;
        inter.reserve(nt);
        for (size_t i = 0; i < nt; i += 2) {
            inter.push_back(i);
        }
        for (size_t i = 1; i < nt; i += 2) {
            inter.push_back(i);
        }
        orders.push_back(std::move(inter));
    }
    // Cyclic rotations (helps ring / path-like target orders); skip r==0 (same as identity).
    const size_t nrot = std::min(nt, size_t(32));
    for (size_t r = 1; r < nrot; ++r) {
        std::vector<size_t> ord;
        ord.reserve(nt);
        for (size_t i = 0; i < nt; ++i) {
            ord.push_back((r + i) % nt);
        }
        orders.push_back(std::move(ord));
    }

    size_t best = 0;
    for (const auto& ord : orders) {
        std::vector<int> mapping(nt, -1);
        std::vector<bool> used(graph_data.n_global, false);
        size_t pref_so_far = 0;
        bool ok = true;
        for (size_t ti : ord) {
            struct Cand {
                size_t g;
                bool is_pref;
            };
            std::vector<Cand> cands;
            cands.reserve(enc.allowed_global_idx[ti].size());
            for (size_t g : enc.allowed_global_idx[ti]) {
                if (g >= used.size() || used[g]) {
                    continue;
                }
                bool is_pref = false;
                if (ti < constraint_data.preferred_global_indices.size()) {
                    const auto& pv = constraint_data.preferred_global_indices[ti];
                    is_pref = !pv.empty() && std::binary_search(pv.begin(), pv.end(), g);
                }
                cands.push_back({g, is_pref});
            }
            std::stable_partition(cands.begin(), cands.end(), [](const Cand& c) { return c.is_pref; });
            bool placed = false;
            for (const Cand& cand : cands) {
                if (!topology_sat_check_edge_feasibility(graph_data, mapping, ti, cand.g)) {
                    continue;
                }
                mapping[ti] = static_cast<int>(cand.g);
                used[cand.g] = true;
                if (cand.is_pref) {
                    ++pref_so_far;
                }
                placed = true;
                break;
            }
            if (!placed) {
                ok = false;
                break;
            }
        }
        if (ok) {
            best = std::max(best, pref_so_far);
        }
    }
    return best;
}

// Returns false if the clause would be empty (no literal can be true to map outside shape_set).
bool topology_sat_build_shape_blocking_clause(
    const TopologySatHardEncoding& enc, const std::vector<int>& shape_sorted, std::vector<int>& clause_out) {
    clause_out.clear();
    const size_t nt = enc.assign_lit.size();
    for (size_t t = 0; t < nt; ++t) {
        const auto& globs = enc.allowed_global_idx[t];
        const auto& lits = enc.assign_lit[t];
        for (size_t k = 0; k < globs.size(); ++k) {
            const int g = static_cast<int>(globs[k]);
            if (!std::binary_search(shape_sorted.begin(), shape_sorted.end(), g)) {
                clause_out.push_back(lits[k]);
            }
        }
    }
    return !clause_out.empty();
}

void topology_sat_add_shape_clause_or_unsat(
    TopologySatSolver& solver, const TopologySatHardEncoding& enc, std::vector<int>& clause_working) {
    if (clause_working.empty()) {
        if (!enc.assign_lit.empty() && !enc.assign_lit[0].empty()) {
            const int lit = enc.assign_lit[0][0];
            solver.add(lit);
            solver.add(0);
            solver.add(-lit);
            solver.add(0);
        } else {
            // No variables were ever declared — declare one now so CaDiCaL's strict variable check
            // (factor=1, enabled by default in CaDiCaL 3.0.0) accepts the literal.
            const int v = solver.declare_one_more_variable();
            solver.add(v);
            solver.add(0);
            solver.add(-v);
            solver.add(0);
        }
        return;
    }
    for (int lit : clause_working) {
        solver.add(lit);
    }
    solver.add(0);
}

// Exclude one complete assignment (or its image-set shape when unique_shapes).
bool topology_sat_add_blocking_clause_for_mapping_impl(
    TopologySatSolver& solver, TopologySatHardEncoding& enc, const std::vector<int>& raw_mapping, bool unique_shapes) {
    if (unique_shapes) {
        const auto shape_key = topology_mapping_shape_key(raw_mapping);
        std::vector<int> shape_clause;
        if (!topology_sat_build_shape_blocking_clause(enc, shape_key, shape_clause)) {
            if (!enc.assign_lit.empty() && !enc.assign_lit[0].empty()) {
                const int lit = enc.assign_lit[0][0];
                solver.add(lit);
                solver.add(0);
                solver.add(-lit);
                solver.add(0);
            } else {
                const int v = solver.declare_one_more_variable();
                solver.add(v);
                solver.add(0);
                solver.add(-v);
                solver.add(0);
            }
        } else {
            for (int lit : shape_clause) {
                solver.add(lit);
            }
            solver.add(0);
        }
        return true;
    }
    const size_t nt = enc.assign_lit.size();
    std::vector<int> new_blocking;
    new_blocking.reserve(nt);
    for (size_t t = 0; t < nt; ++t) {
        const int chosen_global = raw_mapping[t];
        if (chosen_global < 0) {
            return false;
        }
        const auto& globs = enc.allowed_global_idx[t];
        const auto& lits = enc.assign_lit[t];
        bool found_k = false;
        for (size_t k = 0; k < globs.size(); ++k) {
            if (static_cast<int>(globs[k]) == chosen_global) {
                new_blocking.push_back(-lits[k]);
                found_k = true;
                break;
            }
        }
        if (!found_k) {
            return false;
        }
    }
    for (int lit : new_blocking) {
        solver.add(lit);
    }
    solver.add(0);
    return true;
}

}  // namespace

bool topology_sat_add_blocking_clause_for_mapping(
    TopologySatSolver& solver, TopologySatHardEncoding& enc, const std::vector<int>& raw_mapping, bool unique_shapes) {
    return topology_sat_add_blocking_clause_for_mapping_impl(solver, enc, raw_mapping, unique_shapes);
}

// ── Cardinality Encoding Primitives ──────────────────────────────────────────

inline bool topology_sat_combinations_exceed_limit(size_t n, size_t r, size_t max_combinations) {
    if (r > n) {
        return true;
    }
    if (r == 0 || r == n) {
        return false;
    }
    r = std::min(r, n - r);
    double x = 1.0;
    for (size_t i = 1; i <= r; ++i) {
        x = x * static_cast<double>(n - r + i) / static_cast<double>(i);
        if (x > static_cast<double>(max_combinations)) {
            return true;
        }
    }
    return false;
}

template <typename EmitCombination>
void topology_sat_emit_combinations_indices(size_t n, size_t r, EmitCombination&& emit_combination) {
    std::vector<size_t> cur;
    cur.reserve(r);
    const auto dfs = [&](auto&& self, size_t start) -> void {
        if (cur.size() == r) {
            std::forward<EmitCombination>(emit_combination)(cur);
            return;
        }
        for (size_t i = start; i < n; ++i) {
            if (n - i < r - cur.size()) {
                break;
            }
            cur.push_back(i);
            self(self, i + 1);
            cur.pop_back();
        }
    };
    dfs(dfs, 0);
}

// Sequential counter encoding for at-least-k: O(m*k) clauses + O(m*k) auxiliary variables.
// c[i][j] represents "at least j+1 of lits[0..i] are true"; assert c[m-1][k-1].
// If extra_lit != 0 it is added only to the final assertion, so extra_lit => at-least-k without making the
// cardinality hard (used for the optional preferred objective).
inline void topology_sat_add_at_least_k_counter(
    TopologySatSolver& solver, const std::vector<int>& lits, size_t k, int extra_lit = 0) {
    const size_t m = lits.size();
    std::vector<std::vector<int>> c(m);
    for (size_t i = 0; i < m; ++i) {
        const size_t cols = std::min(k, i + 1);
        c[i].resize(cols);
        for (size_t j = 0; j < cols; ++j) {
            c[i][j] = solver.declare_one_more_variable();
        }
    }
    solver.add(-lits[0]);
    solver.add(c[0][0]);
    solver.add(0);
    solver.add(-c[0][0]);
    solver.add(lits[0]);
    solver.add(0);
    for (size_t i = 1; i < m; ++i) {
        const size_t cols = std::min(k, i + 1);
        for (size_t j = 0; j < cols; ++j) {
            if (j == 0) {
                solver.add(-lits[i]);
                solver.add(c[i][0]);
                solver.add(0);
                solver.add(-c[i - 1][0]);
                solver.add(c[i][0]);
                solver.add(0);
                solver.add(-c[i][0]);
                solver.add(lits[i]);
                solver.add(c[i - 1][0]);
                solver.add(0);
            } else if (j == i) {
                solver.add(-lits[i]);
                solver.add(-c[i - 1][j - 1]);
                solver.add(c[i][j]);
                solver.add(0);
                solver.add(-c[i][j]);
                solver.add(lits[i]);
                solver.add(0);
                solver.add(-c[i][j]);
                solver.add(c[i - 1][j - 1]);
                solver.add(0);
            } else {
                solver.add(-lits[i]);
                solver.add(-c[i - 1][j - 1]);
                solver.add(c[i][j]);
                solver.add(0);
                solver.add(-c[i - 1][j]);
                solver.add(c[i][j]);
                solver.add(0);
                solver.add(-c[i][j]);
                solver.add(c[i - 1][j]);
                solver.add(lits[i]);
                solver.add(0);
                solver.add(-c[i][j]);
                solver.add(c[i - 1][j]);
                solver.add(c[i - 1][j - 1]);
                solver.add(0);
            }
        }
    }
    if (extra_lit != 0) {
        solver.add(extra_lit);
    }
    solver.add(c[m - 1][k - 1]);
    solver.add(0);
}

// At-least-k on independent literals.  Uses the small combinatorial encoding when affordable (O(C(m,m-k+1))
// clauses), otherwise falls back to the sequential counter encoding (O(m*k) clauses + aux vars).
inline bool topology_sat_add_at_least_k_literals(
    TopologySatSolver& solver,
    const std::vector<int>& lits,
    size_t k,
    size_t max_combination_clauses,
    std::string* trivial_reason,
    int extra_lit = 0) {
    const size_t m = lits.size();
    if (k == 0) {
        return true;
    }
    if (k > m) {
        if (trivial_reason != nullptr) {
            *trivial_reason = fmt::format(
                "topology_sat: cardinality needs at least {} satisfied literals but only {} are listed", k, m);
        }
        return false;
    }
    if (k == m) {
        for (int lit : lits) {
            if (extra_lit != 0) {
                solver.add(extra_lit);
            }
            solver.add(lit);
            solver.add(0);
        }
        return true;
    }
    const size_t clause_width = m - k + 1;
    if (topology_sat_combinations_exceed_limit(m, clause_width, max_combination_clauses)) {
        topology_sat_add_at_least_k_counter(solver, lits, k, extra_lit);
        return true;
    }
    topology_sat_emit_combinations_indices(m, clause_width, [&](const std::vector<size_t>& comb) {
        if (extra_lit != 0) {
            solver.add(extra_lit);
        }
        for (size_t idx : comb) {
            solver.add(lits[idx]);
        }
        solver.add(0);
    });
    return true;
}

// Sequential (Sinz 2005) at-most-one encoding: O(n) clauses + O(n) auxiliary register variables instead of the
// O(n^2) pairwise binary clauses.  Sequential encoding improves unit-propagation on large domains.
inline void topology_sat_add_at_most_one_sequential(TopologySatSolver& solver, const std::vector<int>& lits) {
    const size_t n = lits.size();
    if (n <= 1) {
        return;
    }
    if (n == 2) {
        solver.add(-lits[0]);
        solver.add(-lits[1]);
        solver.add(0);
        return;
    }
    std::vector<int> r;
    r.reserve(n - 1);
    for (size_t i = 0; i < n - 1; ++i) {
        r.push_back(solver.declare_one_more_variable());
    }
    solver.add(-lits[0]);
    solver.add(r[0]);
    solver.add(0);
    for (size_t i = 1; i < n - 1; ++i) {
        solver.add(-lits[i]);
        solver.add(r[i]);
        solver.add(0);
        solver.add(-r[i - 1]);
        solver.add(r[i]);
        solver.add(0);
        solver.add(-r[i - 1]);
        solver.add(-lits[i]);
        solver.add(0);
    }
    solver.add(-r[n - 2]);
    solver.add(-lits[n - 1]);
    solver.add(0);
}

// ── Hard host cap: "at most k same-rank host groups occupied" with a full-packing fast path ────
//
// The minimal-host objective is a cardinality constraint over per-host-group OCCUPANCY: "at most k of the same-rank
// global groups (host partitions) are occupied", with the solver free to choose WHICH k. The general encoding uses a
// sequential-counter over the occupancy literals, whose propagation is weak. When a minimal-host packing fills each
// used host COMPLETELY (n_target is a multiple of a uniform group capacity -- e.g. a 16-host ring where every host is
// fully used), an all-or-nothing per-host encoding forces the count with strong unit propagation alone, no counter --
// this is the fast path that lets the solver actually find such packings (see issue #50253: SC16 ring on SC24).

// Build one "occupied" indicator per non-empty host group: occ_g <=> (some target maps into a global of group g).
// When all_or_nothing is true, additionally force occ_g => every (reachable) global of g is used. This is valid ONLY
// when a minimal-host packing fills each used host completely; it eliminates partially-used hosts, which massively
// prunes the at-most-k search. Returns the occupancy indicators (one per non-empty group).
inline void topology_sat_build_group_occupancy(
    TopologySatSolver& solver,
    const TopologySatConstraintView& constraint_data,
    const TopologySatHardEncoding& enc,
    bool all_or_nothing,
    std::vector<int>& occ_out) {
    occ_out.clear();
    const auto& global_to_host = constraint_data.global_to_same_rank_group;
    const size_t num_groups = constraint_data.same_rank_groups.size();
    if (global_to_host.empty() || num_groups == 0) {
        return;
    }

    // Per group, gather the assign literals landing a target on each global mesh of that group.
    std::vector<std::map<size_t, std::vector<int>>> group_mesh_lits(num_groups);
    const size_t nt = enc.assign_lit.size();
    for (size_t t = 0; t < nt; ++t) {
        const auto& globs = enc.allowed_global_idx[t];
        const auto& lits = enc.assign_lit[t];
        for (size_t k = 0; k < globs.size(); ++k) {
            const size_t g = globs[k];
            if (g >= global_to_host.size()) {
                continue;
            }
            const int label = global_to_host[g];
            if (label < 0 || static_cast<size_t>(label) >= num_groups) {
                continue;
            }
            group_mesh_lits[static_cast<size_t>(label)][g].push_back(lits[k]);
        }
    }

    for (size_t p = 0; p < num_groups; ++p) {
        auto& mesh_lits = group_mesh_lits[p];
        if (mesh_lits.empty()) {
            continue;
        }
        // Per-mesh "used" indicator: used_m <=> OR(assign lits that land a target on mesh m).
        std::vector<int> used_m;
        used_m.reserve(mesh_lits.size());
        for (auto& [gidx, lits] : mesh_lits) {
            const int um = solver.declare_one_more_variable();
            solver.add(-um);  // um => OR(lits)
            for (int l : lits) {
                solver.add(l);
            }
            solver.add(0);
            for (int l : lits) {  // each lit => um
                solver.add(-l);
                solver.add(um);
                solver.add(0);
            }
            used_m.push_back(um);
        }
        // occ_g <=> OR(used_m).
        const int occ = solver.declare_one_more_variable();
        solver.add(-occ);
        for (int um : used_m) {
            solver.add(um);
        }
        solver.add(0);
        for (int um : used_m) {
            solver.add(-um);
            solver.add(occ);
            solver.add(0);
        }
        if (all_or_nothing) {
            for (int um : used_m) {  // occ => every reachable mesh of the group is used
                solver.add(-occ);
                solver.add(um);
                solver.add(0);
            }
        }
        occ_out.push_back(occ);
    }
}

// Capacity feasibility: can k same-rank global groups hold n_target placements at all? The k LARGEST groups must sum
// to >= n_target (generalizes ceil(n_target / max_group_size) to non-uniform group sizes).
inline bool topology_sat_max_groups_cap_capacity_feasible(
    const TopologySatConstraintView& constraint_data, size_t n_target, size_t k) {
    if (k == 0 || n_target == 0) {
        return true;
    }
    std::vector<size_t> capacities;
    capacities.reserve(constraint_data.same_rank_groups.size());
    for (const auto& g : constraint_data.same_rank_groups) {
        if (!g.empty()) {
            capacities.push_back(g.size());
        }
    }
    if (capacities.empty()) {
        return true;  // no partition registered; cap is non-binding
    }
    std::sort(capacities.begin(), capacities.end(), std::greater<size_t>());
    size_t reachable_capacity = 0;
    for (size_t i = 0; i < k && i < capacities.size(); ++i) {
        reachable_capacity += capacities[i];
    }
    return reachable_capacity >= n_target;
}

// HARD: at most k_hosts same-rank global groups occupied. Returns true if encoded (or non-binding); false only if the
// underlying cardinality is trivially impossible. `full_packing` == true additionally applies the all-or-nothing
// occupancy tightening (a used host must be completely filled), which lets tight packings propagate by unit
// resolution and is what makes cases like a 16-host ring on a 24-host cluster tractable (issue #50253). The
// at-most-k counter is ALWAYS emitted, though: `full_packing` is derived from the RAW group size, but pinnings /
// degree filtering / AC-3 can leave a group with fewer *reachable* globals than its raw capacity, and then
// n_target == k * capacity can be satisfied across MORE than k partially-reachable groups. Relying on all-or-nothing
// alone would silently exceed the cap; the counter guarantees the bound. The counter is over the per-group
// occupancy indicators only (one literal per group), so it stays cheap and does not reintroduce the old bottleneck.
inline bool topology_sat_encode_at_most_k_groups(
    TopologySatSolver& solver,
    const TopologySatConstraintView& constraint_data,
    const TopologySatHardEncoding& enc,
    size_t k_hosts,
    bool full_packing,
    int extra_lit = 0) {
    std::vector<int> occ;
    topology_sat_build_group_occupancy(solver, constraint_data, enc, /*all_or_nothing=*/full_packing, occ);
    const size_t num_present = occ.size();
    if (num_present == 0 || k_hosts >= num_present) {
        return true;  // not binding
    }
    // "at most k occupied" == "at least (num_present - k) of the negated occupancy literals".
    std::vector<int> neg;
    neg.reserve(num_present);
    for (int o : occ) {
        neg.push_back(-o);
    }
    static constexpr size_t kGroupBudgetCombClauses = 500000;
    std::string reason;
    return topology_sat_add_at_least_k_literals(
        solver, neg, num_present - k_hosts, kGroupBudgetCombClauses, &reason, extra_lit);
}

// ── Hard Constraint Encoding Sub-functions ────────────────────────────────────
//
// The following functions collectively implement topology_sat_encode_hard_constraints,
// broken into one function per constraint type for clarity.  The top-level function
// (at the bottom of this section) is a thin orchestrator that calls them in order.

// Step 1: Build per-target candidate domains by applying degree and constraint filtering.
// Any global whose degree is below the target degree, or that fails is_valid_mapping, is
// excluded.  Returns false (and sets enc.trivial_unsat) if any target has an empty domain.
bool topology_sat_build_initial_domains(
    const TopologySatGraphView& graph_data,
    const TopologySatConstraintView& constraint_data,
    TopologySatHardEncoding& enc,
    std::vector<std::vector<size_t>>& domain_out) {
    const size_t nt = graph_data.n_target;
    const size_t ng = graph_data.n_global;
    domain_out.resize(nt);
    for (size_t t = 0; t < nt; ++t) {
        for (size_t g = 0; g < ng; ++g) {
            if (!constraint_data.is_valid_mapping(t, g)) {
                continue;
            }
            if (graph_data.global_deg[g] < graph_data.target_deg[t]) {
                continue;
            }
            domain_out[t].push_back(g);
        }
        if (domain_out[t].empty()) {
            enc.trivial_unsat = true;
            enc.trivial_reason = fmt::format("topology_sat: no allowed global for target_idx {}", t);
            return false;
        }
    }
    return true;
}

// Step 2: AC-3 arc-consistency propagation.
//
// WHY AC-3: After degree/constraint filtering the domains can still contain globals that
// have no feasible partner for some adjacent target.  AC-3 iteratively removes such
// "unsupported" values.  Smaller domains mean fewer SAT variables and shorter support
// clauses in Step 6, which substantially speeds up the SAT solver on dense instances.
//
// The worklist starts with every arc (t, t_neigh).  Whenever a domain shrinks, all arcs
// pointing INTO t are re-added so their support can be re-checked.  The iteration cap of
// 100 prevents quadratic blow-up on pathological inputs while handling real topologies.
bool topology_sat_apply_arc_consistency(
    const TopologySatGraphView& graph_data,
    [[maybe_unused]] const TopologySatConstraintView& constraint_data,
    ConnectionValidationMode validation_mode,
    TopologySatHardEncoding& enc,
    std::vector<std::vector<size_t>>& domain) {
    const size_t nt = graph_data.n_target;

    // Build membership sets for fast O(1) domain lookup during support checks.
    std::vector<std::unordered_set<size_t>> domain_set(nt);
    for (size_t t = 0; t < nt; ++t) {
        domain_set[t].insert(domain[t].begin(), domain[t].end());
    }

    // has_support(t, g, t_neigh): true iff there exists at least one value g2 in domain[t_neigh]
    // adjacent to g (and meeting channel requirements in STRICT mode).
    auto has_support = [&](size_t t, size_t g, size_t t_neigh) -> bool {
        size_t required_channels = 1;
        if (validation_mode == ConnectionValidationMode::STRICT) {
            if (t < graph_data.target_conn_count.size()) {
                const auto& tc = graph_data.target_conn_count[t];
                const auto itc = tc.find(t_neigh);
                if (itc != tc.end()) {
                    required_channels = itc->second;
                }
            }
        }
        for (size_t g2 : graph_data.global_adj_idx[g]) {
            if (g2 == g) {
                continue;
            }
            if (!domain_set[t_neigh].contains(g2)) {
                continue;
            }
            if (validation_mode == ConnectionValidationMode::STRICT && required_channels > 1) {
                size_t actual = 0;
                if (g < graph_data.global_conn_count.size()) {
                    const auto& gc = graph_data.global_conn_count[g];
                    const auto itg = gc.find(g2);
                    if (itg != gc.end()) {
                        actual = itg->second;
                    }
                }
                if (actual < required_channels) {
                    continue;
                }
            }
            return true;
        }
        return false;
    };

    // Collect all arcs (t, t_neigh) to check.
    std::vector<std::pair<size_t, size_t>> worklist;
    for (size_t t = 0; t < nt; ++t) {
        for (size_t tn : graph_data.target_adj_idx[t]) {
            worklist.emplace_back(t, tn);
        }
    }

    static constexpr size_t kMaxAC3Iterations = 100;
    for (size_t iter = 0; iter < kMaxAC3Iterations && !worklist.empty(); ++iter) {
        std::vector<std::pair<size_t, size_t>> next_worklist;
        for (const auto& [t, tn] : worklist) {
            auto& dom = domain[t];
            size_t before = dom.size();
            dom.erase(
                std::remove_if(dom.begin(), dom.end(), [&](size_t g) { return !has_support(t, g, tn); }), dom.end());
            if (dom.size() < before) {
                domain_set[t].clear();
                domain_set[t].insert(dom.begin(), dom.end());
                if (dom.empty()) {
                    enc.trivial_unsat = true;
                    enc.trivial_reason =
                        fmt::format("topology_sat: arc consistency emptied domain for target_idx {}", t);
                    return false;
                }
                // Re-enqueue arcs into t so their support is re-checked now that domain[t] shrank.
                for (size_t t2 : graph_data.target_adj_idx[t]) {
                    if (t2 != tn) {
                        next_worklist.emplace_back(t2, t);
                    }
                }
            }
        }
        worklist = std::move(next_worklist);
    }

    return true;
}

// Step 3: Allocate one SAT Boolean variable per (target, domain-global) pair and record
// them in enc.assign_lit / enc.allowed_global_idx.  Preferred globals for a target are
// listed first in the row so that the solver's internal variable-order heuristic naturally
// tries preferred assignments first under a single solve (no MaxSAT needed).
void topology_sat_create_assignment_variables(
    TopologySatSolver& solver,
    const TopologySatConstraintView& constraint_data,
    TopologySatHardEncoding& enc,
    std::vector<std::vector<size_t>>& domain) {
    const size_t nt = domain.size();

    // Sort each domain so preferred globals come first.
    for (size_t t = 0; t < nt; ++t) {
        if (t < constraint_data.preferred_global_indices.size() &&
            !constraint_data.preferred_global_indices[t].empty()) {
            const auto& pref = constraint_data.preferred_global_indices[t];
            auto& dom = domain[t];
            std::stable_partition(
                dom.begin(), dom.end(), [&](size_t g) { return std::binary_search(pref.begin(), pref.end(), g); });
        }
    }

    for (size_t t = 0; t < nt; ++t) {
        enc.allowed_global_idx[t] = std::move(domain[t]);
        enc.assign_lit[t].reserve(enc.allowed_global_idx[t].size());
        for (size_t k = 0; k < enc.allowed_global_idx[t].size(); ++k) {
            (void)k;
            const int v = solver.declare_one_more_variable();
            enc.assign_lit[t].push_back(v);
        }
    }
}

// Step 4: Exactly-one constraint per target.
// For each target t:
//   - At-least-one: unit clause (x_{t,g0} v x_{t,g1} v ... v x_{t,gk}).
//   - At-most-one:  sequential (Sinz) encoding to avoid O(domain^2) pairwise clauses.
void topology_sat_encode_exactly_one_per_target(TopologySatSolver& solver, const TopologySatHardEncoding& enc) {
    const size_t nt = enc.assign_lit.size();
    for (size_t t = 0; t < nt; ++t) {
        const auto& lits = enc.assign_lit[t];
        TT_ASSERT(lits.size() == enc.allowed_global_idx[t].size());
        for (int lit : lits) {
            solver.add(lit);
        }
        solver.add(0);
        topology_sat_add_at_most_one_sequential(solver, lits);
    }
}

// Step 5: Injectivity -- each global node may be used by at most one target.
// Collect all assign literals that reference each global, then add AMO over that set.
void topology_sat_encode_injectivity(
    TopologySatSolver& solver, const TopologySatGraphView& graph_data, const TopologySatHardEncoding& enc) {
    const size_t nt = enc.assign_lit.size();
    const size_t ng = graph_data.n_global;

    std::vector<std::vector<int>> lits_per_global(ng);
    for (size_t t = 0; t < nt; ++t) {
        for (size_t k = 0; k < enc.assign_lit[t].size(); ++k) {
            const size_t g = enc.allowed_global_idx[t][k];
            lits_per_global[g].push_back(enc.assign_lit[t][k]);
        }
    }
    for (size_t g = 0; g < ng; ++g) {
        topology_sat_add_at_most_one_sequential(solver, lits_per_global[g]);
    }
}

// Step 5b: Bijection completeness. When |targets| == |globals| an injective mapping is necessarily surjective, so
// every global must be used by exactly one target. The at-least-one-per-global clauses (the dual of injectivity)
// are logically redundant given exactly-one-per-target + injectivity, but they give the SAT solver the
// permutation/pigeonhole propagation it otherwise lacks -- which is what makes otherwise-intractable bijection
// instances (e.g. a logical ring embedded into a sparse physical graph, i.e. a Hamiltonian-cycle search) converge.
// Returns false (trivial UNSAT) if some global has no candidate target: no bijection can then exist.
bool topology_sat_encode_bijection_completeness(
    TopologySatSolver& solver, const TopologySatGraphView& graph_data, TopologySatHardEncoding& enc) {
    if (graph_data.n_target != graph_data.n_global) {
        return true;
    }
    const size_t nt = enc.assign_lit.size();
    const size_t ng = graph_data.n_global;
    std::vector<std::vector<int>> lits_per_global(ng);
    for (size_t t = 0; t < nt; ++t) {
        for (size_t k = 0; k < enc.assign_lit[t].size(); ++k) {
            lits_per_global[enc.allowed_global_idx[t][k]].push_back(enc.assign_lit[t][k]);
        }
    }
    for (size_t g = 0; g < ng; ++g) {
        if (lits_per_global[g].empty()) {
            enc.trivial_unsat = true;
            enc.trivial_reason = fmt::format(
                "Topology SAT: global node {} has no candidate target, so no bijection exists (n_target == n_global == "
                "{})",
                g,
                ng);
            return false;
        }
        for (int lit : lits_per_global[g]) {
            solver.add(lit);
        }
        solver.add(0);
    }
    return true;
}

// Step 6: Adjacency preservation via support encoding.
//
// WHY support encoding (not pairwise clauses):
//   For each directed arc (t1 -> g_a) and each adjacent target t2, we emit ONE clause:
//       not x_{t1,g_a}  v  x_{t2,g_{b1}}  v  x_{t2,g_{b2}}  v  ...
//   where g_{bi} ranges over all globals in domain[t2] adjacent (and channel-compatible
//   in STRICT mode) to g_a.  This is O(edges x domain_size) clauses, vs. the naive
//   O(edges x domain_size^2) pairwise incompatibility clauses.  When a candidate has NO
//   compatible partner the clause degenerates to the unit clause not x_{t1,g_a}, giving
//   implicit arc-consistency filtering inside the solver.
void topology_sat_encode_adjacency_support(
    TopologySatSolver& solver,
    const TopologySatGraphView& graph_data,
    const TopologySatHardEncoding& enc,
    ConnectionValidationMode validation_mode) {
    const size_t nt = enc.assign_lit.size();

    for (size_t t1 = 0; t1 < nt; ++t1) {
        for (size_t t2 : graph_data.target_adj_idx[t1]) {
            if (t2 <= t1) {
                continue;
            }
            const auto& gidx1 = enc.allowed_global_idx[t1];
            const auto& lit1 = enc.assign_lit[t1];
            const auto& gidx2 = enc.allowed_global_idx[t2];
            const auto& lit2 = enc.assign_lit[t2];
            size_t required_channels = 1;
            if (t1 < graph_data.target_conn_count.size()) {
                const auto& tc = graph_data.target_conn_count[t1];
                const auto itc = tc.find(t2);
                if (itc != tc.end()) {
                    required_channels = itc->second;
                }
            }
            auto is_compatible = [&](size_t ga, size_t gb) -> bool {
                if (ga == gb) {
                    return false;
                }
                if (!are_globals_adjacent(graph_data, ga, gb)) {
                    return false;
                }
                if (validation_mode == ConnectionValidationMode::STRICT) {
                    size_t actual_channels = 0;
                    if (ga < graph_data.global_conn_count.size()) {
                        const auto& gc = graph_data.global_conn_count[ga];
                        const auto itg = gc.find(gb);
                        if (itg != gc.end()) {
                            actual_channels = itg->second;
                        }
                    }
                    if (actual_channels < required_channels) {
                        return false;
                    }
                }
                return true;
            };
            // Forward direction: if t1 is assigned g_a, t2 must map to some compatible g_b.
            for (size_t i1 = 0; i1 < gidx1.size(); ++i1) {
                solver.add(-lit1[i1]);
                for (size_t i2 = 0; i2 < gidx2.size(); ++i2) {
                    if (is_compatible(gidx1[i1], gidx2[i2])) {
                        solver.add(lit2[i2]);
                    }
                }
                solver.add(0);
            }
            // Backward direction: if t2 is assigned g_b, t1 must map to some compatible g_a.
            for (size_t i2 = 0; i2 < gidx2.size(); ++i2) {
                solver.add(-lit2[i2]);
                for (size_t i1 = 0; i1 < gidx1.size(); ++i1) {
                    if (is_compatible(gidx1[i1], gidx2[i2])) {
                        solver.add(lit1[i1]);
                    }
                }
                solver.add(0);
            }
        }
    }
}

// Step 7: Same-rank group constraints.
// Targets in the same group (target_to_group[t] == tg, tg != SIZE_MAX) must all map to
// globals that share the same global_to_same_rank_group label.  Pairs with different
// labels get a binary incompatibility clause not x_{t1,g1} v not x_{t2,g2}.
void topology_sat_encode_same_rank_groups(
    TopologySatSolver& solver,
    [[maybe_unused]] const TopologySatGraphView& graph_data,
    const TopologySatConstraintView& constraint_data,
    const TopologySatHardEncoding& enc) {
    const size_t nt = enc.assign_lit.size();
    const auto& target_to_group = constraint_data.target_to_group;
    const auto& global_rank = constraint_data.global_to_same_rank_group;
    if (target_to_group.empty() || global_rank.empty()) {
        return;
    }
    for (size_t t1 = 0; t1 < nt; ++t1) {
        if (t1 >= target_to_group.size()) {
            continue;
        }
        const size_t tg = target_to_group[t1];
        if (tg == SIZE_MAX) {
            continue;
        }
        for (size_t t2 = t1 + 1; t2 < nt; ++t2) {
            if (t2 >= target_to_group.size() || target_to_group[t2] != tg) {
                continue;
            }
            const auto& gidx1 = enc.allowed_global_idx[t1];
            const auto& lit1 = enc.assign_lit[t1];
            const auto& gidx2 = enc.allowed_global_idx[t2];
            const auto& lit2 = enc.assign_lit[t2];
            for (size_t i1 = 0; i1 < gidx1.size(); ++i1) {
                const size_t glob1 = gidx1[i1];
                if (glob1 >= global_rank.size()) {
                    continue;
                }
                const int L1 = global_rank[glob1];
                for (size_t i2 = 0; i2 < gidx2.size(); ++i2) {
                    const size_t glob2 = gidx2[i2];
                    if (glob2 >= global_rank.size()) {
                        continue;
                    }
                    const int L2 = global_rank[glob2];
                    if (L1 != L2) {
                        solver.add(-lit1[i1]);
                        solver.add(-lit2[i2]);
                        solver.add(0);
                    }
                }
            }
        }
    }
}

// Step 8: Cardinality constraints -- at-least-k over specified (target, global) pairs.
// Each fulfilled pair contributes its weight (default 1). Encode by repeating the assign
// literal `weight` times into the existing at-least-k helper (same encoding as unweighted
// when every weight is 1). Do not unique the expanded slots: that would drop the weight.
bool topology_sat_encode_cardinality_constraints(
    TopologySatSolver& solver, const TopologySatConstraintView& constraint_data, TopologySatHardEncoding& enc) {
    static constexpr size_t kMaxCardinalityCombClauses = 500000;

    for (const auto& card_entry : constraint_data.cardinality_constraints) {
        const size_t min_count = card_entry.min_count;
        std::vector<int> slots;
        std::set<int> distinct_lits;
        for (const auto& [ti, gi] : card_entry.pairs) {
            if (ti >= enc.allowed_global_idx.size()) {
                continue;
            }
            const auto& globs = enc.allowed_global_idx[ti];
            const auto& lits_row = enc.assign_lit[ti];
            int assign_lit = 0;
            bool found = false;
            for (size_t kk = 0; kk < globs.size(); ++kk) {
                if (globs[kk] == gi) {
                    assign_lit = lits_row[kk];
                    found = true;
                    break;
                }
            }
            if (!found) {
                continue;
            }
            distinct_lits.insert(assign_lit);
            slots.push_back(assign_lit);
        }
        static constexpr size_t kMaxCardinalityLiterals = 4096;
        if (distinct_lits.size() > kMaxCardinalityLiterals) {
            enc.trivial_unsat = true;
            enc.trivial_reason = fmt::format(
                "topology_sat: cardinality has {} distinct feasible pair literals (cap {}); narrow the pair set or "
                "raise the cap",
                distinct_lits.size(),
                kMaxCardinalityLiterals);
            return false;
        }
        if (slots.size() < min_count) {
            enc.trivial_unsat = true;
            enc.trivial_reason = fmt::format(
                "topology_sat: cardinality needs {} listings but only {} are feasible in the current domains",
                min_count,
                slots.size());
            return false;
        }
        std::string card_reason;
        if (!topology_sat_add_at_least_k_literals(solver, slots, min_count, kMaxCardinalityCombClauses, &card_reason)) {
            enc.trivial_unsat = true;
            enc.trivial_reason =
                card_reason.empty() ? std::string("topology_sat: cardinality encoding failed") : std::move(card_reason);
            return false;
        }
    }

    return true;
}

// Top-level hard constraint orchestrator.  Calls the eight sub-functions above in order:
//   1. Build initial domains  (degree + constraint filtering)
//   2. Apply AC-3 arc consistency
//   3. Create assignment variables
//   4. Exactly-one per target (ALO + AMO)
//   5. Injectivity (AMO over globals)
//   6. Adjacency support clauses
//   7. Same-rank group incompatibility clauses
//   8. Cardinality at-least-k constraints
bool topology_sat_encode_hard_constraints(
    TopologySatSolver& solver,
    const TopologySatGraphView& graph_data,
    const TopologySatConstraintView& constraint_data,
    TopologySatHardEncoding& enc,
    ConnectionValidationMode validation_mode) {
    enc = TopologySatHardEncoding{};
    const size_t nt = graph_data.n_target;

    if (nt == 0) {
        return true;
    }

    enc.allowed_global_idx.resize(nt);
    enc.assign_lit.resize(nt);

    // 1. Initial domain: constraint + degree filtering.
    std::vector<std::vector<size_t>> domain;
    if (!topology_sat_build_initial_domains(graph_data, constraint_data, enc, domain)) {
        return false;
    }

    // 2. Arc consistency (AC-3).
    if (!topology_sat_apply_arc_consistency(graph_data, constraint_data, validation_mode, enc, domain)) {
        return false;
    }

    // 3. Create assignment variables (preferred globals listed first in each row).
    topology_sat_create_assignment_variables(solver, constraint_data, enc, domain);

    // 4. Exactly one global choice per target.
    topology_sat_encode_exactly_one_per_target(solver, enc);

    // 5. Injective: each global node used by at most one target.
    topology_sat_encode_injectivity(solver, graph_data, enc);

    // 5b. Bijection completeness (only binds when n_target == n_global): every global must be used. Strengthens
    // propagation for permutation-shaped instances and detects globals with no candidate target as trivial UNSAT.
    if (!topology_sat_encode_bijection_completeness(solver, graph_data, enc)) {
        return false;
    }

    // 6. Adjacency preservation via support encoding.
    topology_sat_encode_adjacency_support(solver, graph_data, enc, validation_mode);

    // 7. Same-rank target groups.
    topology_sat_encode_same_rank_groups(solver, graph_data, constraint_data, enc);

    // 8. Cardinality: at least min_count of the listed (target, global) assignment literals must be true.
    return topology_sat_encode_cardinality_constraints(solver, constraint_data, enc);
}

// ── Soft / Objective Encoding ─────────────────────────────────────────────────

// topology_sat_append_preferred_hit_indicators
//
// For each target t that has at least one preferred global reachable in its domain,
// introduce a Tseitin indicator variable p_t and add the bidirectional equivalence:
//
//     p_t  <=>  (x_{t,g_{p1}} v x_{t,g_{p2}} v ... v x_{t,g_{pk}})
//
// where g_{p1..pk} are the globals in domain[t] intersect preferred_globals[t].
//
// Tseitin encoding: two clause groups enforce the equivalence without introducing
// exponential blowup:
//   Forward  (p -> OR):  not p  v  x_{t,g_{p1}}  v  ...  v  x_{t,g_{pk}}
//   Backward (x -> p):  for each pi:  not x_{t,g_{pi}}  v  p
//
// The resulting p_t literals are collected into pref_hit_literals_out and later
// fed into topology_sat_add_at_least_k_literals to force the solver toward the
// maximum simultaneously achievable preferred-hit count.
void topology_sat_append_preferred_hit_indicators(
    TopologySatSolver& solver,
    const TopologySatHardEncoding& enc,
    const TopologySatConstraintView& constraint_data,
    std::vector<int>& pref_hit_literals_out) {
    pref_hit_literals_out.clear();
    const size_t nt = enc.assign_lit.size();
    for (size_t t = 0; t < nt; ++t) {
        if (t >= constraint_data.preferred_global_indices.size()) {
            continue;
        }
        const auto& preferred_globals = constraint_data.preferred_global_indices[t];
        if (preferred_globals.empty()) {
            continue;
        }

        // Collect assign literals for this target that correspond to preferred globals.
        const auto& globs = enc.allowed_global_idx[t];
        const auto& row_lits = enc.assign_lit[t];
        std::vector<int> row_pref_lits;
        row_pref_lits.reserve(globs.size());
        for (size_t k = 0; k < globs.size(); ++k) {
            if (std::binary_search(preferred_globals.begin(), preferred_globals.end(), globs[k])) {
                row_pref_lits.push_back(row_lits[k]);
            }
        }
        if (row_pref_lits.empty()) {
            continue;
        }

        // Introduce indicator p and encode p <=> OR(row_pref_lits) via two clause groups.
        const int p = solver.declare_one_more_variable();

        // Forward: not p v x_{t,g_{p1}} v ... v x_{t,g_{pk}}
        solver.add(-p);
        for (int lit : row_pref_lits) {
            solver.add(lit);
        }
        solver.add(0);

        // Backward: for each pi, not x_{t,g_{pi}} v p
        for (int lit : row_pref_lits) {
            solver.add(p);
            solver.add(-lit);
            solver.add(0);
        }

        pref_hit_literals_out.push_back(p);
    }
}

// indicator <=> OR_p (a_p & b_p)  (Tseitin on pairwise AND of two positive assign literals).
inline bool topology_sat_define_indicator_as_or_of_pairwise_and(
    TopologySatSolver& solver, int indicator, const std::vector<std::pair<int, int>>& pair_lits) {
    if (pair_lits.empty()) {
        solver.add(-indicator);
        solver.add(0);
        return true;
    }
    std::vector<int> y_vars;
    y_vars.reserve(pair_lits.size());
    for (const auto& ab : pair_lits) {
        const int a = ab.first;
        const int b = ab.second;
        const int y = solver.declare_one_more_variable();
        y_vars.push_back(y);
        solver.add(-y);
        solver.add(a);
        solver.add(0);
        solver.add(-y);
        solver.add(b);
        solver.add(0);
        solver.add(-a);
        solver.add(-b);
        solver.add(y);
        solver.add(0);
    }
    solver.add(-indicator);
    for (int y : y_vars) {
        solver.add(y);
    }
    solver.add(0);
    for (int y : y_vars) {
        solver.add(-y);
        solver.add(indicator);
        solver.add(0);
    }
    return true;
}

// Upper bound on how many relaxed-channel threshold literals would be created (one per (edge, k) level). Cheap
// O(edges) count so we can skip building the auxiliary channel CNF when the k-descent pass would be too expensive.
size_t topology_sat_relaxed_channel_threshold_literal_count_upper_bound(const TopologySatGraphView& graph_data) {
    static constexpr size_t kMaxKPerEdge = 24;
    size_t cnt = 0;
    const size_t nt = graph_data.n_target;
    for (size_t t1 = 0; t1 < nt; ++t1) {
        for (size_t t2 : graph_data.target_adj_idx[t1]) {
            if (t2 <= t1) {
                continue;
            }
            size_t required = 1;
            if (t1 < graph_data.target_conn_count.size()) {
                const auto& tc = graph_data.target_conn_count[t1];
                const auto itc = tc.find(t2);
                if (itc != tc.end()) {
                    required = std::max(required, itc->second);
                }
            }
            if (t2 < graph_data.target_conn_count.size()) {
                const auto& tc2 = graph_data.target_conn_count[t2];
                const auto itc2 = tc2.find(t1);
                if (itc2 != tc2.end()) {
                    required = std::max(required, itc2->second);
                }
            }
            cnt += std::min(required, kMaxKPerEdge);
        }
    }
    return cnt;
}

// RELAXED: for each undirected target edge (t1,t2) and each k in 1..min(R,kMaxK), add literal I_{e,k} true iff the
// chosen globals for t1,t2 use an adjacent host edge with parallel link count >= k. Maximizing sum_{e,k} I_{e,k}
// equals maximizing sum_e min(R_e, actual_e) (same objective shape DFS uses for channel_match_score ordering).
bool topology_sat_append_relaxed_channel_threshold_literals(
    TopologySatSolver& solver,
    const TopologySatHardEncoding& enc,
    const TopologySatGraphView& graph_data,
    std::vector<int>& channel_threshold_literals_out,
    std::string* fail_reason) {
    channel_threshold_literals_out.clear();
    static constexpr size_t kMaxKPerEdge = 24;
    static constexpr size_t kMaxPairsPerIndicator = 512;
    static constexpr size_t kMaxTotalIndicators = 2048;

    const size_t nt = enc.assign_lit.size();

    for (size_t t1 = 0; t1 < nt; ++t1) {
        for (size_t t2 : graph_data.target_adj_idx[t1]) {
            if (t2 <= t1) {
                continue;
            }
            size_t required = 1;
            if (t1 < graph_data.target_conn_count.size()) {
                const auto& tc = graph_data.target_conn_count[t1];
                const auto itc = tc.find(t2);
                if (itc != tc.end()) {
                    required = std::max(required, itc->second);
                }
            }
            if (t2 < graph_data.target_conn_count.size()) {
                const auto& tc2 = graph_data.target_conn_count[t2];
                const auto itc2 = tc2.find(t1);
                if (itc2 != tc2.end()) {
                    required = std::max(required, itc2->second);
                }
            }
            const auto& gidx1 = enc.allowed_global_idx[t1];
            const auto& lit1 = enc.assign_lit[t1];
            const auto& gidx2 = enc.allowed_global_idx[t2];
            const auto& lit2 = enc.assign_lit[t2];
            const size_t k_hi = std::min(required, kMaxKPerEdge);
            for (size_t k = 1; k <= k_hi; ++k) {
                std::vector<std::pair<int, int>> pair_lits;
                pair_lits.reserve(std::min(gidx1.size() * gidx2.size(), kMaxPairsPerIndicator + 1));
                for (size_t i1 = 0; i1 < gidx1.size(); ++i1) {
                    const size_t glob1 = gidx1[i1];
                    for (size_t i2 = 0; i2 < gidx2.size(); ++i2) {
                        const size_t glob2 = gidx2[i2];
                        if (glob1 == glob2) {
                            continue;
                        }
                        if (!are_globals_adjacent(graph_data, glob1, glob2)) {
                            continue;
                        }
                        size_t actual_channels = 0;
                        if (glob1 < graph_data.global_conn_count.size()) {
                            const auto& gc = graph_data.global_conn_count[glob1];
                            const auto itg = gc.find(glob2);
                            if (itg != gc.end()) {
                                actual_channels = itg->second;
                            }
                        }
                        if (actual_channels < k) {
                            continue;
                        }
                        pair_lits.emplace_back(lit1[i1], lit2[i2]);
                        if (pair_lits.size() > kMaxPairsPerIndicator) {
                            if (fail_reason != nullptr) {
                                *fail_reason = fmt::format(
                                    "topology_sat: relaxed channel indicator for edge ({},{}) level {} exceeds {} "
                                    "feasible assign pairs",
                                    t1,
                                    t2,
                                    k,
                                    kMaxPairsPerIndicator);
                            }
                            return false;
                        }
                    }
                }
                const int ind = solver.declare_one_more_variable();
                if (!topology_sat_define_indicator_as_or_of_pairwise_and(solver, ind, pair_lits)) {
                    return false;
                }
                channel_threshold_literals_out.push_back(ind);
                if (channel_threshold_literals_out.size() > kMaxTotalIndicators) {
                    if (fail_reason != nullptr) {
                        *fail_reason = fmt::format(
                            "topology_sat: relaxed channel threshold literals exceeded {}", kMaxTotalIndicators);
                    }
                    return false;
                }
            }
        }
    }
    return true;
}

bool topology_sat_decode_hard_solution(
    TopologySatSolver& solver, const TopologySatHardEncoding& enc, std::vector<int>& mapping_out) {
    if (enc.trivial_unsat) {
        return false;
    }
    const size_t nt = enc.allowed_global_idx.size();
    mapping_out.assign(nt, static_cast<int>(-1));
    if (nt == 0) {
        return true;
    }
    for (size_t t = 0; t < nt; ++t) {
        const auto& lits = enc.assign_lit[t];
        const auto& globs = enc.allowed_global_idx[t];
        int picked = -1;
        for (size_t k = 0; k < lits.size(); ++k) {
            if (solver.val(lits[k]) > 0) {
                if (picked >= 0) {
                    return false;
                }
                picked = static_cast<int>(globs[k]);
            }
        }
        if (picked < 0) {
            return false;
        }
        mapping_out[t] = picked;
    }
    return true;
}

// Value-symmetry-breaking hint for equal-size (bijection) instances. Embedding a logical graph into an equal-size
// physical graph (e.g. a ring -> a Hamiltonian cycle) has large value symmetry -- any automorphism of the
// physical graph maps one solution to another -- which makes generic CDCL re-derive the same conflicts under each
// symmetric image and thrash. Fixing one target to one candidate collapses that symmetry. We return the literal
// to *assume* (not assert): assumptions are retracted after each solve(), so the caller re-solves without it if it
// proves the instance UNSAT. That makes this sound for any instance with no graph-shape detection -- the only
// precondition is a bijection, where this symmetry (and the resulting hardness) actually arises. Returns 0 when no
// hint applies.
int topology_sat_symmetry_assumption_lit(const TopologySatGraphView& graph_data, const TopologySatHardEncoding& enc) {
    if (graph_data.n_target != graph_data.n_global) {
        return 0;
    }
    if (enc.assign_lit.empty() || enc.assign_lit[0].empty()) {
        return 0;
    }
    return enc.assign_lit[0][0];
}

// ── SatSearchBackend (session + CaDiCaL live only here) ───────────────────────

bool SatSearchBackend::start(
    const TopologySatGraphView& graph_data,
    const TopologySatConstraintView& constraint_data,
    ConnectionValidationMode validation_mode,
    bool unique_shapes,
    const std::vector<std::vector<int>>& initial_forbidden_shape_keys,
    std::string* error_out) {
    impl_.reset();
    if (constraint_data.max_same_rank_groups_used > 0 &&
        !topology_sat_max_groups_cap_capacity_feasible(
            constraint_data, graph_data.n_target, constraint_data.max_same_rank_groups_used)) {
        return false;
    }
    impl_ = std::make_unique<Impl>();
    auto& s = *impl_;
    s.solver.configure_for_blocking_clause_enumeration();
    s.unique_shapes = unique_shapes;
    s.enc = {};
    if (!topology_sat_encode_hard_constraints(s.solver, graph_data, constraint_data, s.enc, validation_mode)) {
        if (error_out != nullptr) {
            *error_out = s.enc.trivial_reason;
        }
        impl_.reset();
        return false;
    }

    // HARD host-group cap: at-most-k occupancy in CNF. Infeasible caps fail the session; the mapper restarts
    // without the cap. Do not encode a guarded/optional cap here.
    if (constraint_data.max_same_rank_groups_used > 0) {
        size_t num_host_groups = 0;
        size_t max_group_capacity = 0;
        size_t min_group_capacity = SIZE_MAX;
        for (const auto& grp : constraint_data.same_rank_groups) {
            if (!grp.empty()) {
                ++num_host_groups;
                max_group_capacity = std::max(max_group_capacity, grp.size());
                min_group_capacity = std::min(min_group_capacity, grp.size());
            }
        }
        if (num_host_groups >= 1 && max_group_capacity > 0) {
            const size_t K = constraint_data.max_same_rank_groups_used;
            const bool uniform_capacity = (min_group_capacity == max_group_capacity);
            const bool full_packing = uniform_capacity && (graph_data.n_target == K * max_group_capacity);
            if (!topology_sat_max_groups_cap_capacity_feasible(constraint_data, graph_data.n_target, K) ||
                !topology_sat_encode_at_most_k_groups(s.solver, constraint_data, s.enc, K, full_packing)) {
                impl_.reset();
                return false;
            }
            s.cap_active = true;
        }
    }

    // SOFT occupancy packing: only when there is no HARD cap. Encode at-most-k_floor as optional CNF
    // (activation extra_lit on asserting clauses). First solve assumes the lit; later stages drop it so
    // an infeasible packing never fails the session.
    if (constraint_data.max_same_rank_groups_used == 0 && constraint_data.minimize_same_rank_groups_used) {
        size_t num_host_groups = 0;
        size_t max_group_capacity = 0;
        size_t min_group_capacity = SIZE_MAX;
        for (const auto& grp : constraint_data.same_rank_groups) {
            if (!grp.empty()) {
                ++num_host_groups;
                max_group_capacity = std::max(max_group_capacity, grp.size());
                min_group_capacity = std::min(min_group_capacity, grp.size());
            }
        }
        if (num_host_groups >= 1 && max_group_capacity > 0) {
            const size_t k_floor = (graph_data.n_target + max_group_capacity - 1) / max_group_capacity;
            const bool uniform_capacity = (min_group_capacity == max_group_capacity);
            const bool full_packing = uniform_capacity && (graph_data.n_target == k_floor * max_group_capacity);
            if (k_floor < num_host_groups &&
                topology_sat_max_groups_cap_capacity_feasible(constraint_data, graph_data.n_target, k_floor)) {
                const int minimize_lit = s.solver.declare_one_more_variable();
                if (topology_sat_encode_at_most_k_groups(
                        s.solver, constraint_data, s.enc, k_floor, full_packing, /*extra_lit=*/-minimize_lit)) {
                    s.minimize_lit = minimize_lit;
                } else {
                    s.solver.add(-minimize_lit);
                    s.solver.add(0);
                }
            }
        }
    }

    // Preferred at-least-k (when preferred mappings exist) and RELAXED channel-threshold literals.
    // The preferred objective is a ranking of solutions, so it is skipped for unique_shapes enumeration: that
    // mode asks for every distinct image set (e.g. PGD placement candidates), and ranking would only reorder
    // them while making the first placements depend on the preference rather than on the topology.
    std::vector<int> pref_hit_literals;
    if (!unique_shapes) {
        topology_sat_append_preferred_hit_indicators(s.solver, s.enc, constraint_data, pref_hit_literals);
    }
    if (!pref_hit_literals.empty()) {
        static constexpr size_t kExactPreferredLbMaxTargets = 10;
        static constexpr size_t kMidPreferredLbMaxTargets = 20;
        static constexpr size_t kPreferredLbDfsBudgetSmall = 80'000'000;
        static constexpr size_t kPreferredLbDfsBudgetMid = 400'000;
        const size_t nt = graph_data.n_target;
        size_t k_lb = 0;
        if (nt <= kExactPreferredLbMaxTargets) {
            k_lb = topology_sat_preferred_exact_lower_bound(
                graph_data, constraint_data, s.enc, kPreferredLbDfsBudgetSmall);
        } else if (nt <= kMidPreferredLbMaxTargets) {
            k_lb =
                topology_sat_preferred_exact_lower_bound(graph_data, constraint_data, s.enc, kPreferredLbDfsBudgetMid);
        } else {
            k_lb = topology_sat_preferred_greedy_lower_bound(graph_data, constraint_data, s.enc);
            if (k_lb == 0) {
                k_lb = topology_sat_preferred_exact_lower_bound(graph_data, constraint_data, s.enc, 600'000);
            }
        }
        if (k_lb > 0) {
            const size_t k_use = std::min(k_lb, pref_hit_literals.size());
            static constexpr size_t kPrefCardinalityCombClauses = 500000;
            std::string card_reason;
            const int preferred_lit = s.solver.declare_one_more_variable();
            const bool encoded = topology_sat_add_at_least_k_literals(
                s.solver,
                pref_hit_literals,
                k_use,
                kPrefCardinalityCombClauses,
                &card_reason,
                /*extra_lit=*/-preferred_lit);
            if (encoded) {
                s.preferred_lit = preferred_lit;
            } else {
                s.solver.add(-preferred_lit);
                s.solver.add(0);
            }
        }
    }
    if (validation_mode == ConnectionValidationMode::RELAXED) {
        static constexpr size_t kMaxRelaxedChannelLiteralsSingleSolve = 256;
        const size_t ch_mc_ub = topology_sat_relaxed_channel_threshold_literal_count_upper_bound(graph_data);
        if (ch_mc_ub <= kMaxRelaxedChannelLiteralsSingleSolve) {
            std::vector<int> ch_lits;
            std::string ch_reason;
            (void)topology_sat_append_relaxed_channel_threshold_literals(
                s.solver, s.enc, graph_data, ch_lits, &ch_reason);
        }
    }

    s.symmetry_lit = topology_sat_symmetry_assumption_lit(graph_data, s.enc);
    const int min_lit = s.minimize_lit;
    const int pref = s.preferred_lit;
    s.stages.clear();
    if (min_lit != 0 && pref != 0) {
        s.stages = {{min_lit, pref}, {min_lit}, {pref}, {}};
    } else if (min_lit != 0) {
        s.stages = {{min_lit}, {}};
    } else if (pref != 0) {
        s.stages = {{pref}, {}};
    } else {
        s.stages = {{}};
    }
    for (const auto& shape_key : initial_forbidden_shape_keys) {
        std::vector<int> forbid_clause;
        topology_sat_build_shape_blocking_clause(s.enc, shape_key, forbid_clause);
        topology_sat_add_shape_clause_or_unsat(s.solver, s.enc, forbid_clause);
    }
    return true;
}

bool SatSearchBackend::block(const std::vector<int>& mapping) {
    if (impl_ == nullptr) {
        return false;
    }
    auto& s = *impl_;
    return topology_sat_add_blocking_clause_for_mapping(s.solver, s.enc, mapping, s.unique_shapes);
}

bool SatSearchBackend::next(std::vector<int>& mapping_out) {
    if (impl_ == nullptr) {
        return false;
    }
    auto& s = *impl_;
    auto solve_and_decode = [&]() -> bool {
        for (; s.stage < s.stages.size(); ++s.stage) {
            const auto& optional_lits = s.stages[s.stage];
            auto solve_once = [&](bool with_symmetry_hint) -> bool {
                if (with_symmetry_hint) {
                    s.solver.assume(s.symmetry_lit);
                }
                for (int lit : optional_lits) {
                    s.solver.assume(lit);
                }
                ++s.solve_calls;
                bool limited = s.cap_active;
                if (!limited && s.minimize_lit != 0) {
                    for (int lit : optional_lits) {
                        if (lit == s.minimize_lit) {
                            limited = true;
                            break;
                        }
                    }
                }
                const int status = limited ? s.solver.solve_limited(kHostCapConflictBudget) : s.solver.solve();
                return status == TopologySatSolver::kSat;
            };
            if ((s.symmetry_lit != 0 && solve_once(/*with_symmetry_hint=*/true)) ||
                solve_once(/*with_symmetry_hint=*/false)) {
                return topology_sat_decode_hard_solution(s.solver, s.enc, mapping_out);
            }
        }
        return false;
    };
    if (!solve_and_decode()) {
        return false;
    }
    (void)block(mapping_out);
    return true;
}

size_t SatSearchBackend::solve_calls() const noexcept { return impl_ == nullptr ? 0 : impl_->solve_calls; }

}  // namespace tt::tt_fabric::detail
