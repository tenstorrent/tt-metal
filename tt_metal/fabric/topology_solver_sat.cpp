// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <limits>
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

// Full definition of the opaque session type forward-declared in topology_solver.hpp.
struct TopologySatSession {
    TopologySatSolver solver;
    // Minimal-host priming (see TOPOLOGY_OCCUPANCY_SOLVE_README §7): when the session carries an occupancy objective,
    // create_and_encode primes the solver (warm descent + full-packing lock) and makes the achieved cap PERMANENT.
    // The primed model is returned by the FIRST solve_and_decode; subsequent calls are bounded (warm + capped).
    std::vector<int> primed_first_mapping;
    bool has_primed_mapping = false;
    int enum_loop_budget = 0;  // >0 => bound each solve_and_decode with solve_limited (occupancy objective present)
};

// ── Adjacency and Edge Helpers ────────────────────────────────────────────────
namespace {

// ── Phase profiling (opt-in via TT_TOPO_SAT_PROFILE=1) ────────────────────────
// Emits per-phase wall-clock timings for the SAT encode/solve pipeline so we can attribute where a slow ring solve
// actually spends its time (domain build, AC-3, adjacency support, symmetry break, the solve itself, ...). Off by
// default (single env lookup, cached) so it costs nothing in production.
inline bool topology_sat_profile_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("TT_TOPO_SAT_PROFILE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

class TopologySatScopedTimer {
public:
    explicit TopologySatScopedTimer(std::string label) :
        label_(std::move(label)), start_(std::chrono::steady_clock::now()) {}
    ~TopologySatScopedTimer() {
        if (!topology_sat_profile_enabled()) {
            return;
        }
        const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count();
        log_info(tt::LogFabric, "[topo-sat-profile] {} : {:.1f} ms", label_, ms);
    }

private:
    std::string label_;
    std::chrono::steady_clock::time_point start_;
};

// Manual (non-RAII) elapsed helper for phases that don't map cleanly to a scope.
inline double topology_sat_elapsed_ms(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

// Read a non-negative integer tuning knob from the environment, or return `fallback` if unset/invalid. Cached per
// variable name is not needed here (called at most a few times per solve), but the lookup is trivially cheap.
inline long topology_sat_env_long(const char* name, long fallback) {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return fallback;
    }
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(v, &end, 10);
    if (end == v || errno != 0 || parsed < 0) {
        return fallback;
    }
    return parsed;
}

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

// Exclude one complete assignment (or its image-set shape when unique_shapes) — same logic as topology_sat_search_n.
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
// c[i][j] represents "at least j+1 of lits[0..i] are true".
//   - force (default true): assert c[m-1][k-1] ("at least k of lits") as a hard unit clause.
//   - out_last_row != nullptr: also fill it with the final row c[m-1][0..cols-1] (c[m-1][j] == ">= (j+1) of lits"),
//     so one counter encoding exposes every threshold as an assumable literal (used by the soft minimize descent).
inline void topology_sat_add_at_least_k_counter(
    TopologySatSolver& solver,
    const std::vector<int>& lits,
    size_t k,
    bool force = true,
    std::vector<int>* out_last_row = nullptr) {
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
    if (out_last_row != nullptr) {
        *out_last_row = c[m - 1];
    }
    if (force) {
        solver.add(c[m - 1][k - 1]);
        solver.add(0);
    }
}

// At-least-k on independent literals.  Uses the small combinatorial encoding when affordable (O(C(m,m-k+1))
// clauses), otherwise falls back to the sequential counter encoding (O(m*k) clauses + aux vars).
inline bool topology_sat_add_at_least_k_literals(
    TopologySatSolver& solver,
    const std::vector<int>& lits,
    size_t k,
    size_t max_combination_clauses,
    std::string* trivial_reason) {
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
            solver.add(lit);
            solver.add(0);
        }
        return true;
    }
    const size_t clause_width = m - k + 1;
    if (topology_sat_combinations_exceed_limit(m, clause_width, max_combination_clauses)) {
        topology_sat_add_at_least_k_counter(solver, lits, k);
        return true;
    }
    topology_sat_emit_combinations_indices(m, clause_width, [&](const std::vector<size_t>& comb) {
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

// ── Same-rank-group occupancy: minimal-host-count objective ───────────────────
//
// The inter-mesh minimal-host objective is expressed as a cardinality constraint over per-host-group OCCUPANCY:
// "at most k of the same-rank global groups (host partitions) are occupied", where the solver freely chooses WHICH
// k (any combination) -- so it never pins to a specific, possibly-unroutable cover. Two flavours share the same
// occupancy indicators: a HARD cap (topology_sat_encode_at_most_k_groups) and a SOFT minimize
// (topology_sat_solve_minimize_groups). Both are driven purely by MappingConstraints (max_/minimize_
// same_rank_groups_used) -- callers set the groups + target; nothing here is solver-specific policy.

// Build one "occupied" indicator per non-empty host group: occ_g <=> (some target maps into a global of group g).
// When all_or_nothing is true, additionally force occ_g => every (reachable) global of g is used. This is valid
// ONLY when a minimal-host packing fills each used host completely (target count is a multiple of a uniform group
// capacity); it eliminates partially-used hosts, which massively prunes the at-most-k search (the fast path for the
// hard cap). Returns the occupancy indicators (one per non-empty group).
inline void topology_sat_build_group_occupancy(
    TopologySatSolver& solver,
    const TopologySatConstraintView& constraint_data,
    const TopologySatHardEncoding& enc,
    bool all_or_nothing,
    std::vector<int>& occ_out,
    std::vector<std::vector<int>>* used_per_group_out = nullptr) {
    occ_out.clear();
    if (used_per_group_out != nullptr) {
        used_per_group_out->clear();
    }
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
        if (used_per_group_out != nullptr) {
            used_per_group_out->push_back(std::move(used_m));
        }
    }
}

// Add the all-or-nothing tightening (occ_g => every reachable mesh of group g is used) to an occupancy encoding that
// was built WITHOUT it. Lets a warm incremental solver that already descended under partial packing be tightened to
// full packing in place -- reusing all learned clauses + phase saving -- for a final minimal-host "lock" solve.
inline void topology_sat_add_all_or_nothing_tightening(
    TopologySatSolver& solver, const std::vector<int>& occ, const std::vector<std::vector<int>>& used_per_group) {
    const size_t n = std::min(occ.size(), used_per_group.size());
    for (size_t i = 0; i < n; ++i) {
        for (int um : used_per_group[i]) {
            solver.add(-occ[i]);
            solver.add(um);
            solver.add(0);
        }
    }
}

// Capacity feasibility: can k same-rank global groups hold n_target placements at all? Each group contributes as
// many slots as it has globals; the k LARGEST groups must sum to >= n_target (generalizes
// ceil(n_target / max_group_size) to non-uniform group sizes).
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

// HARD: at most k_hosts same-rank global groups occupied. Returns true if encoded (or non-binding); false only if
// the underlying cardinality is trivially impossible.
//
// `full_packing` == true means a used host must be completely filled and n_target == k_hosts * capacity. In that
// case the all-or-nothing clauses ALONE force the count: with an injective placement of exactly k_hosts*capacity
// meshes into all-or-nothing hosts of that capacity, exactly k_hosts hosts end up occupied -- so we skip the
// cardinality counter entirely. This is the fast path: only local per-host clauses (strong unit propagation), no
// sequential-counter aux variables (whose propagation is weak and was the bottleneck).
inline bool topology_sat_encode_at_most_k_groups(
    TopologySatSolver& solver,
    const TopologySatConstraintView& constraint_data,
    const TopologySatHardEncoding& enc,
    size_t k_hosts,
    bool full_packing) {
    std::vector<int> occ;
    topology_sat_build_group_occupancy(solver, constraint_data, enc, /*all_or_nothing=*/full_packing, occ);
    if (full_packing) {
        return true;  // all-or-nothing already forces exactly k_hosts occupied; no counter needed
    }
    const size_t num_present = occ.size();
    if (num_present == 0 || k_hosts >= num_present) {
        return true;  // not binding
    }
    // General case: explicit "at most k occupied" == "at least (num_present - k) of the negated occupancy literals".
    std::vector<int> neg;
    neg.reserve(num_present);
    for (int o : occ) {
        neg.push_back(-o);
    }
    static constexpr size_t kGroupBudgetCombClauses = 500000;
    std::string reason;
    return topology_sat_add_at_least_k_literals(solver, neg, num_present - k_hosts, kGroupBudgetCombClauses, &reason);
}

// SOFT: minimize the number of occupied groups, best-effort. Takes one warm feasible solve, then descends an
// assumable "at most (current-1)" budget under a per-step conflict cap, keeping the best (fewest-group) model.
// Never turns a feasible instance UNSAT (step 1 is unconstrained). Writes the best mapping to best_mapping_out and
// its group count to best_k_out; returns true on any feasible model. `k_floor` stops the descent once reached
// (e.g. the capacity lower bound) so we don't waste solves probing below the achievable minimum.
inline bool topology_sat_solve_minimize_groups(
    TopologySatSolver& solver,
    const TopologySatHardEncoding& enc,
    const TopologySatConstraintView& constraint_data,
    int conflict_cap,
    size_t k_floor,
    std::vector<int>& best_mapping_out,
    size_t& best_k_out,
    size_t hard_cap_k = 0,
    int hard_conflict_cap = 0,
    bool* hard_cap_met_out = nullptr,
    bool make_cap_permanent = false) {
    // make_cap_permanent: after settling on best_k, assert "<= best_k occupied" as a PERMANENT unit clause (not a
    // one-shot assumption) so the SAME solver can be reused for blocking-clause enumeration / incremental .next with
    // every subsequent solve() automatically bounded to best_k. Used by topology_sat_search_n and the session; the
    // single solve leaves it false (it never re-solves after this).
    best_mapping_out.clear();
    best_k_out = 0;
    if (hard_cap_met_out != nullptr) {
        *hard_cap_met_out = false;
    }
    std::vector<int> occ;
    std::vector<std::vector<int>> used_per_group;
    topology_sat_build_group_occupancy(
        solver, constraint_data, enc, /*all_or_nothing=*/false, occ, &used_per_group);
    const size_t num_present = occ.size();

    if (topology_sat_profile_enabled()) {
        std::map<size_t, int> size_hist;
        for (const auto& upg : used_per_group) {
            ++size_hist[upg.size()];
        }
        std::string hist;
        for (const auto& [sz, cnt] : size_hist) {
            hist += fmt::format("{}x{} ", cnt, sz);
        }
        log_info(
            tt::LogFabric,
            "[topo-sat-profile]   minimize.group_reachable_mesh_sizes : num_groups={} hist(count x reachable)={}",
            num_present,
            hist);
    }

    if (num_present < 2) {  // nothing to minimize; just find any feasible model
        if (solver.solve() != TopologySatSolver::kSat) {
            return false;
        }
        return topology_sat_decode_hard_solution(solver, enc, best_mapping_out);
    }

    std::vector<int> neg;
    neg.reserve(num_present);
    for (int o : occ) {
        neg.push_back(-o);
    }
    // One shared counter: geq_unoccupied[j] == ">= (j+1) groups UNoccupied" == "<= (num_present-(j+1)) occupied".
    std::vector<int> geq_unoccupied;
    topology_sat_add_at_least_k_counter(solver, neg, num_present - 1, /*force=*/false, &geq_unoccupied);

    auto count_occupied = [&]() {
        size_t c = 0;
        for (int o : occ) {
            if (solver.val(o) == o) {
                ++c;
            }
        }
        return c;
    };
    auto atmost_lit = [&](size_t k) -> int {  // assume => "<= k occupied"
        if (k >= num_present) {
            return 0;
        }
        const size_t need = num_present - k;  // groups that must be unoccupied
        return (need >= 1 && need - 1 < geq_unoccupied.size()) ? geq_unoccupied[need - 1] : 0;
    };

    const bool profile = topology_sat_profile_enabled();
    auto t_warm = std::chrono::steady_clock::now();
    if (solver.solve() != TopologySatSolver::kSat) {  // step 1: warm feasible model
        if (profile) {
            log_info(
                tt::LogFabric,
                "[topo-sat-profile]   minimize.warm_solve : {:.1f} ms (UNSAT/unknown, num_present={})",
                topology_sat_elapsed_ms(t_warm),
                num_present);
        }
        return false;
    }
    best_k_out = count_occupied();
    topology_sat_decode_hard_solution(solver, enc, best_mapping_out);
    if (profile) {
        log_info(
            tt::LogFabric,
            "[topo-sat-profile]   minimize.warm_solve : {:.1f} ms (SAT, occupied={}, num_present={})",
            topology_sat_elapsed_ms(t_warm),
            best_k_out,
            num_present);
    }

    const size_t floor = std::max<size_t>(k_floor, 1);
    size_t iter = 0;
    while (best_k_out > floor) {
        const size_t target_k = best_k_out - 1;
        const int bound = atmost_lit(target_k);
        if (bound == 0) {
            break;
        }
        solver.assume(bound);
        auto t_iter = std::chrono::steady_clock::now();
        const int st = (conflict_cap > 0) ? solver.solve_limited(conflict_cap) : solver.solve();
        ++iter;
        if (st == TopologySatSolver::kSat) {
            best_k_out = count_occupied();  // may drop by more than one
            topology_sat_decode_hard_solution(solver, enc, best_mapping_out);
            if (profile) {
                log_info(
                    tt::LogFabric,
                    "[topo-sat-profile]   minimize.descent[{}] target<={} : {:.1f} ms (SAT, now occupied={})",
                    iter,
                    target_k,
                    topology_sat_elapsed_ms(t_iter),
                    best_k_out);
            }
        } else {
            if (profile) {
                log_info(
                    tt::LogFabric,
                    "[topo-sat-profile]   minimize.descent[{}] : {:.1f} ms (status={} -> stop, floor={})",
                    iter,
                    topology_sat_elapsed_ms(t_iter),
                    st,
                    floor);
            }
            break;  // kUnsat: optimal reached.  kUnknown: too hard -> keep best proven.
        }
    }

    // Optional final HARD-CAP LOCK. If a hard cap K was requested and the partial-packing descent did not already
    // reach it, make one more attempt on THIS warm solver: tighten to full packing (all-or-nothing) and assume
    // "<= K occupied". Reusing every learned clause + saved phase from the descent is the strongest warm start we
    // can give the cap; the all-or-nothing clauses give strong unit propagation that the partial descent lacks,
    // so this can crack the exact minimal-host packing where the partial descent stalls just above it. Sound: on
    // UNSAT/unknown we keep the best descent model, so this never regresses a feasible result.
    if (hard_cap_k > 0 && !best_mapping_out.empty() && best_k_out > hard_cap_k && hard_cap_k < num_present) {
        topology_sat_add_all_or_nothing_tightening(solver, occ, used_per_group);
        const int bound = atmost_lit(hard_cap_k);
        auto t_lock = std::chrono::steady_clock::now();
        if (bound != 0) {
            solver.assume(bound);
        }
        const int st = (hard_conflict_cap > 0) ? solver.solve_limited(hard_conflict_cap) : solver.solve();
        if (st == TopologySatSolver::kSat) {
            best_k_out = count_occupied();
            topology_sat_decode_hard_solution(solver, enc, best_mapping_out);
            if (hard_cap_met_out != nullptr) {
                *hard_cap_met_out = (best_k_out <= hard_cap_k);
            }
        }
        if (profile) {
            log_info(
                tt::LogFabric,
                "[topo-sat-profile]   minimize.hardlock target<={} (full-packing) : {:.1f} ms (status={}, occupied={})",
                hard_cap_k,
                topology_sat_elapsed_ms(t_lock),
                st,
                best_k_out);
        }
    }

    // Permanently cap the solver at the achieved occupancy so it can be reused for enumeration / incremental .next.
    // (One-shot assume() is cleared after each solve(); a unit clause is not.) Only when a real bound exists.
    if (make_cap_permanent && !best_mapping_out.empty() && best_k_out < num_present) {
        const int bound = atmost_lit(best_k_out);
        if (bound != 0) {
            solver.add(bound);
            solver.add(0);
            if (profile) {
                log_info(
                    tt::LogFabric,
                    "[topo-sat-profile]   minimize.permanent_cap : asserted <= {} occupied (unit clause)",
                    best_k_out);
            }
        }
    }
    return !best_mapping_out.empty();
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
// For each entry in constraint_data.cardinality_constraints, collect the assign literals
// corresponding to feasible pairs in the current domains and encode at-least-k using
// either the combinatorial or sequential counter encoding (whichever is cheaper).
bool topology_sat_encode_cardinality_constraints(
    TopologySatSolver& solver, const TopologySatConstraintView& constraint_data, TopologySatHardEncoding& enc) {
    static constexpr size_t kMaxCardinalityCombClauses = 500000;

    for (const auto& card_entry : constraint_data.cardinality_constraints) {
        const auto& pair_set = card_entry.pairs;
        const size_t min_count = card_entry.min_count;
        std::set<int> distinct_lits;
        for (const auto& [ti, gi] : pair_set) {
            if (ti >= enc.allowed_global_idx.size()) {
                continue;
            }
            const auto& globs = enc.allowed_global_idx[ti];
            const auto& lits_row = enc.assign_lit[ti];
            for (size_t kk = 0; kk < globs.size(); ++kk) {
                if (globs[kk] == gi) {
                    distinct_lits.insert(lits_row[kk]);
                    break;
                }
            }
        }
        std::vector<int> lits(distinct_lits.begin(), distinct_lits.end());
        static constexpr size_t kMaxCardinalityLiterals = 4096;
        if (lits.size() > kMaxCardinalityLiterals) {
            enc.trivial_unsat = true;
            enc.trivial_reason = fmt::format(
                "topology_sat: cardinality has {} distinct feasible pair literals (cap {}); narrow the pair set or "
                "raise the cap",
                lits.size(),
                kMaxCardinalityLiterals);
            return false;
        }
        if (lits.size() < min_count) {
            enc.trivial_unsat = true;
            enc.trivial_reason = fmt::format(
                "topology_sat: cardinality needs {} satisfied literals but only {} (target,global) pairs are "
                "feasible in the current domains",
                min_count,
                lits.size());
            return false;
        }
        std::string card_reason;
        if (!topology_sat_add_at_least_k_literals(solver, lits, min_count, kMaxCardinalityCombClauses, &card_reason)) {
            enc.trivial_unsat = true;
            enc.trivial_reason =
                card_reason.empty() ? std::string("topology_sat: cardinality encoding failed") : std::move(card_reason);
            return false;
        }
    }

    return true;
}

// ── Ring / snake structural detection + symmetry breaking ─────────────────────
//
// Embedding a 1-D logical target (a ring or a path/"snake") into a larger physical graph is a
// Hamiltonian-cycle-like search whose cost explodes under the target's own automorphisms: a ring of N nodes has
// 2N automorphisms (N rotations x 2 reflections), a path has 2 (identity + reversal). CDCL re-derives equivalent
// conflicts under each symmetric image and thrashes. We detect these shapes structurally on the target graph
// (cheap, O(V+E), independent of global size) and break the symmetry:
//   * Rotation (rings only): fix the anchor node's image as an *assumption* (retracted on UNSAT -> sound for any
//     instance, including rings smaller than the global graph). See topology_sat_symmetry_assumption_lit.
//   * Reflection (rings and snakes): assert img(neighbor_a) < img(neighbor_b) as a *hard* lex clause -- sound for a
//     *detected* ring/snake because the mirror (reversed-traversal) solution always exists, so exactly one of the
//     twin pair satisfies the strict inequality. Removes the clockwise/CCW twin (extra 2x). See
//     topology_sat_encode_ring_reflection_break.
struct TopologySatRingSnakeInfo {
    bool is_ring = false;
    bool is_snake = false;
    // Ring: anchor is any node (we use 0); neighbor_a/neighbor_b are the anchor's two ring neighbors, swapped by the
    // reflection. Snake: anchor is one endpoint; neighbor_a/neighbor_b are the two path endpoints, swapped by the
    // reflection (the reversal of the path).
    size_t anchor = 0;
    size_t neighbor_a = 0;
    size_t neighbor_b = 0;
    bool valid() const { return is_ring || is_snake; }
};

// Detect whether the logical target graph is a single ring (cycle) or snake (simple path). O(V+E).
//   Ring:  connected, every node has exactly 2 distinct neighbors (=> edges == nodes), n >= 3.
//   Snake: connected, exactly two degree-1 endpoints, all others degree 2 (=> edges == nodes-1), n >= 2.
inline TopologySatRingSnakeInfo topology_sat_detect_ring_or_snake(const TopologySatGraphView& graph_data) {
    TopologySatRingSnakeInfo info;
    const size_t n = graph_data.n_target;
    if (n < 2) {
        return info;
    }

    // Distinct-neighbor sets (target_adj_idx may list a neighbor once per connection).
    std::vector<std::set<size_t>> nbr(n);
    for (size_t u = 0; u < n; ++u) {
        for (size_t v : graph_data.target_adj_idx[u]) {
            if (v < n && v != u) {
                nbr[u].insert(v);
            }
        }
    }

    // Connectivity via iterative DFS from node 0.
    size_t visited_count = 0;
    std::vector<bool> visited(n, false);
    std::vector<size_t> stack{0};
    visited[0] = true;
    while (!stack.empty()) {
        const size_t u = stack.back();
        stack.pop_back();
        ++visited_count;
        for (size_t v : nbr[u]) {
            if (!visited[v]) {
                visited[v] = true;
                stack.push_back(v);
            }
        }
    }
    if (visited_count != n) {
        return info;  // disconnected -> not a single ring/snake
    }

    size_t deg1 = 0;
    size_t deg2 = 0;
    std::vector<size_t> endpoints;
    for (size_t u = 0; u < n; ++u) {
        const size_t d = nbr[u].size();
        if (d == 1) {
            ++deg1;
            endpoints.push_back(u);
        } else if (d == 2) {
            ++deg2;
        } else {
            return info;  // a branch or isolated node -> neither ring nor snake
        }
    }

    if (deg1 == 0 && deg2 == n && n >= 3) {
        info.is_ring = true;
        info.anchor = 0;
        auto it = nbr[0].begin();
        info.neighbor_a = *it;
        info.neighbor_b = *std::next(it);
    } else if (deg1 == 2 && (deg1 + deg2) == n) {
        info.is_snake = true;
        info.anchor = endpoints[0];
        info.neighbor_a = endpoints[0];
        info.neighbor_b = endpoints[1];
    }
    return info;
}

// Reflection symmetry break: forbid every combination where img(neighbor_a) >= img(neighbor_b), i.e. assert the
// strict lexicographic order img(neighbor_a) < img(neighbor_b) on the two reflection-swapped targets. Sound only for
// a detected ring/snake (the mirror solution is guaranteed to exist). Uses the global *node index* as the order key.
void topology_sat_encode_ring_reflection_break(
    TopologySatSolver& solver, const TopologySatGraphView& graph_data, const TopologySatHardEncoding& enc) {
    const auto info = topology_sat_detect_ring_or_snake(graph_data);
    if (!info.valid()) {
        return;
    }
    const size_t ta = info.neighbor_a;
    const size_t tb = info.neighbor_b;
    if (ta >= enc.assign_lit.size() || tb >= enc.assign_lit.size() || ta == tb) {
        return;
    }
    const auto& globs_a = enc.allowed_global_idx[ta];
    const auto& lits_a = enc.assign_lit[ta];
    const auto& globs_b = enc.allowed_global_idx[tb];
    const auto& lits_b = enc.assign_lit[tb];
    // Forbid (a maps to ga) & (b maps to gb) whenever ga >= gb, enforcing the strict order img(a) < img(b).
    // Domain rows may be reordered (preferred globals first), so compare global indices directly rather than by
    // position. O(|A||B|) pairs, which is small (bounded by n_global^2) and one-time per encode.
    for (size_t i = 0; i < globs_a.size(); ++i) {
        const size_t ga = globs_a[i];
        for (size_t j = 0; j < globs_b.size(); ++j) {
            if (globs_b[j] <= ga) {  // violates ga < gb -> forbid this pair
                solver.add(-lits_a[i]);
                solver.add(-lits_b[j]);
                solver.add(0);
            }
        }
    }
}

// Top-level hard constraint orchestrator.  Calls the sub-functions below in order:
//   1. Build initial domains  (degree + constraint filtering)
//   2. Apply AC-3 arc consistency
//   3. Create assignment variables
//   4. Exactly-one per target (ALO + AMO)
//   5. Injectivity (AMO over globals)
//   6. Adjacency support clauses
//   6b. Ring/snake reflection symmetry break (hard, only when a ring/snake target is detected)
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

    const bool profile = topology_sat_profile_enabled();
    auto phase_start = std::chrono::steady_clock::now();
    size_t prev_clauses = solver.num_clauses();
    size_t prev_vars = solver.num_variables();
    auto mark = [&](const char* name) {
        if (profile) {
            const size_t dc = solver.num_clauses() - prev_clauses;
            const size_t dv = solver.num_variables() - prev_vars;
            log_info(
                tt::LogFabric,
                "[topo-sat-profile]   encode.{} : {:.1f} ms (+{} clauses, +{} vars)",
                name,
                topology_sat_elapsed_ms(phase_start),
                dc,
                dv);
        }
        prev_clauses = solver.num_clauses();
        prev_vars = solver.num_variables();
        phase_start = std::chrono::steady_clock::now();
    };

    // 1. Initial domain: constraint + degree filtering.
    std::vector<std::vector<size_t>> domain;
    if (!topology_sat_build_initial_domains(graph_data, constraint_data, enc, domain)) {
        return false;
    }
    mark("1_initial_domains");

    // 2. Arc consistency (AC-3).
    if (!topology_sat_apply_arc_consistency(graph_data, constraint_data, validation_mode, enc, domain)) {
        return false;
    }
    mark("2_arc_consistency");

    // 3. Create assignment variables (preferred globals listed first in each row).
    topology_sat_create_assignment_variables(solver, constraint_data, enc, domain);
    mark("3_create_vars");

    // 4. Exactly one global choice per target.
    topology_sat_encode_exactly_one_per_target(solver, enc);
    mark("4_exactly_one");

    // 5. Injective: each global node used by at most one target.
    topology_sat_encode_injectivity(solver, graph_data, enc);
    mark("5_injectivity");

    // 5b. Bijection completeness (only binds when n_target == n_global): every global must be used. Strengthens
    // propagation for permutation-shaped instances and detects globals with no candidate target as trivial UNSAT.
    if (!topology_sat_encode_bijection_completeness(solver, graph_data, enc)) {
        return false;
    }
    mark("5b_bijection");

    // 6. Adjacency preservation via support encoding.
    topology_sat_encode_adjacency_support(solver, graph_data, enc, validation_mode);
    mark("6_adjacency_support");

    // 6b. Ring/snake reflection symmetry break (hard; no-op unless the target graph is a detected ring/snake).
    topology_sat_encode_ring_reflection_break(solver, graph_data, enc);
    mark("6b_reflection_break");

    // 7. Same-rank target groups.
    topology_sat_encode_same_rank_groups(solver, graph_data, constraint_data, enc);
    mark("7_same_rank_groups");

    // 8. Cardinality: at least min_count of the listed (target, global) assignment literals must be true.
    const bool card_ok = topology_sat_encode_cardinality_constraints(solver, constraint_data, enc);
    mark("8_cardinality");
    return card_ok;
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

// Rotation symmetry-breaking hint for ring targets. Embedding an N-node ring has N rotational images (plus the 2
// reflections handled by topology_sat_encode_ring_reflection_break); CDCL re-derives the same conflicts under each
// rotation and thrashes. Fixing the ring anchor node to one candidate global collapses the rotational orbit. We
// return the literal to *assume* (not assert): assumptions are retracted after each solve(), so the caller
// re-solves without it if it proves the instance UNSAT. That makes this sound for any ring instance -- including
// rings smaller than the global graph (a subset embedding), which the old bijection-only gate skipped. Returns 0
// when the target is not a detected ring (paths have no rotational symmetry, so the reflection break alone applies).
int topology_sat_symmetry_assumption_lit(const TopologySatGraphView& graph_data, const TopologySatHardEncoding& enc) {
    const auto info = topology_sat_detect_ring_or_snake(graph_data);
    if (!info.is_ring) {
        return 0;
    }
    if (info.anchor >= enc.assign_lit.size() || enc.assign_lit[info.anchor].empty()) {
        return 0;
    }
    return enc.assign_lit[info.anchor][0];
}

bool topology_sat_search(
    const TopologySatGraphView& graph_data,
    const TopologySatConstraintView& constraint_data,
    ConnectionValidationMode validation_mode,
    [[maybe_unused]] bool quiet_mode,
    TopologySearchState& state) {
    state = TopologySearchState{};
    state.mapping.assign(graph_data.n_target, -1);
    state.used.assign(graph_data.n_global, false);

    if (graph_data.n_global < graph_data.n_target) {
        state.error_message = fmt::format(
            "Cannot map target graph to global graph: target graph is larger with {} nodes, but global graph only has "
            "{} nodes",
            graph_data.n_target,
            graph_data.n_global);
        if (quiet_mode) {
            log_debug(tt::LogFabric, "{}", state.error_message);
        } else {
            log_error(tt::LogFabric, "{}", state.error_message);
        }
        return false;
    }

    if (graph_data.n_target == 0) {
        return true;
    }

    // Solve with a value-symmetry-breaking assumption (collapses the symmetric models that make a bijection
    // embedding -- a Hamiltonian-cycle search -- thrash). The assumption is only a hint: if it makes the instance
    // UNSAT we re-solve without it, so this never turns a solvable instance UNSAT regardless of graph shape.
    auto solve_with_symmetry_break = [&](TopologySatSolver& solver, const TopologySatHardEncoding& enc) -> int {
        const int assumption = topology_sat_symmetry_assumption_lit(graph_data, enc);
        if (assumption != 0) {
            solver.assume(assumption);
            const int status = solver.solve();
            if (status == TopologySatSolver::kSat) {
                return status;
            }
        }
        return solver.solve();
    };

    // Same rotation-anchor hint, for the conflict-bounded occupancy path (hard host-group cap). If the anchor makes
    // the bounded solve UNSAT/unknown, retry once without it so the hint never turns a solvable instance UNSAT.
    auto solve_limited_with_symmetry_break =
        [&](TopologySatSolver& solver, const TopologySatHardEncoding& enc, int budget) -> int {
        const int assumption = topology_sat_symmetry_assumption_lit(graph_data, enc);
        if (assumption != 0) {
            solver.assume(assumption);
            const int status = solver.solve_limited(budget);
            if (status == TopologySatSolver::kSat) {
                return status;
            }
        }
        return solver.solve_limited(budget);
    };

    auto finalize_success = [&](TopologySatSolver& solver, const TopologySatHardEncoding& enc) -> bool {
        if (!topology_sat_decode_hard_solution(solver, enc, state.mapping)) {
            state.error_message = "Topology SAT: decode failed (model inconsistent with encoding)";
            if (quiet_mode) {
                log_debug(tt::LogFabric, "{}", state.error_message);
            } else {
                log_error(tt::LogFabric, "{}", state.error_message);
            }
            return false;
        }
        std::fill(state.used.begin(), state.used.end(), false);
        for (size_t t = 0; t < state.mapping.size(); ++t) {
            const int gi = state.mapping[t];
            if (gi >= 0 && static_cast<size_t>(gi) < state.used.size()) {
                state.used[static_cast<size_t>(gi)] = true;
            }
        }
        return true;
    };

    auto solve_hard_only = [&](TopologySatSolver& solver, TopologySatHardEncoding& enc) -> bool {
        if (!topology_sat_encode_hard_constraints(solver, graph_data, constraint_data, enc, validation_mode)) {
            state.error_message = enc.trivial_reason.empty()
                                      ? std::string("Topology SAT: encoding failed (trivial UNSAT)")
                                      : enc.trivial_reason;
            if (quiet_mode) {
                log_debug(tt::LogFabric, "{}", state.error_message);
            } else {
                log_error(tt::LogFabric, "{}", state.error_message);
            }
            return false;
        }
        const int status = solve_with_symmetry_break(solver, enc);
        if (status != TopologySatSolver::kSat) {
            state.error_message = fmt::format(
                "Failed to find mapping (SAT): target graph with {} nodes cannot be embedded in global graph with {} "
                "nodes under hard constraints",
                graph_data.n_target,
                graph_data.n_global);
            if (quiet_mode) {
                log_debug(tt::LogFabric, "{}", state.error_message);
            } else {
                log_error(tt::LogFabric, "{}", state.error_message);
            }
            return false;
        }
        return finalize_success(solver, enc);
    };

    // Minimal-host objective, expressed via same-rank-group OCCUPANCY (driven entirely by MappingConstraints):
    //   - max_same_rank_groups_used  > 0 -> HARD "at most K host groups occupied" (solver picks which K).
    //   - minimize_same_rank_groups_used -> SOFT best-effort minimize (used as the fallback if the hard cap can't
    //     be met, or on its own).
    // Try the hard cap first (with the full-packing tightening for speed); if it is unsatisfiable / too hard within
    // the bounded budget, back down to the soft minimize. Both let the solver choose any host combination, so we
    // never pin to a specific (possibly-unroutable) cover.
    if (constraint_data.max_same_rank_groups_used > 0 || constraint_data.minimize_same_rank_groups_used) {
        // Two-budget strategy for the occupancy objective:
        //  - kGroupObjectiveConflictBudget (TT_TOPO_SAT_CONFLICT_BUDGET, default 1M): the cold hard cap and the final
        //    full-packing LOCK -- the lock is what actually reaches K (its all-or-nothing propagation cracks packings
        //    the partial descent cannot), and empirically needs ~250-270k conflicts, so 1M is a safe margin.
        //  - kGroupDescentConflictBudget (TT_TOPO_SAT_DESCENT_BUDGET, default 20k): the partial-packing DESCENT steps.
        //    Kept small on purpose: the descent is only a cheap warm-up for the lock. The deep descent steps (e.g.
        //    target<=22) are combinatorially hard and, with a large budget, one step can grind for minutes (measured
        //    318s at 1M) before the lock ever runs. A small budget makes those steps bail in <1s (status=unknown ->
        //    keep best), handing a warm solver to the lock -- restoring the ~90s end-to-end path.
        const int kGroupObjectiveConflictBudget =
            static_cast<int>(topology_sat_env_long("TT_TOPO_SAT_CONFLICT_BUDGET", 1'000'000));
        const int kGroupDescentConflictBudget =
            static_cast<int>(topology_sat_env_long("TT_TOPO_SAT_DESCENT_BUDGET", 20'000));
        // Hard-cap full-packing LOCK budget. Default 0 == UNBOUNDED: when a strict hard cap is set we run the lock
        // (all-or-nothing, warm from the descent) until it either proves the exact k_min packing SAT or proves it
        // UNSAT -- never returning an over-cap best-effort result within a budget. The descent stays small
        // (TT_TOPO_SAT_DESCENT_BUDGET) so we still hand the lock a warm solver quickly; only the final strict lock
        // is unbounded. Overridable via TT_TOPO_SAT_LOCK_BUDGET (>0 to re-bound for A/B or CI safety).
        const int kGroupLockConflictBudget =
            static_cast<int>(topology_sat_env_long("TT_TOPO_SAT_LOCK_BUDGET", 0));
        const bool profile = topology_sat_profile_enabled();
        if (profile) {
            const auto info = topology_sat_detect_ring_or_snake(graph_data);
            log_info(
                tt::LogFabric,
                "[topo-sat-profile] occupancy path: n_target={} n_global={} ring={} snake={} max_k={} minimize={}",
                graph_data.n_target,
                graph_data.n_global,
                info.is_ring,
                info.is_snake,
                constraint_data.max_same_rank_groups_used,
                constraint_data.minimize_same_rank_groups_used);
        }
        size_t max_cap = 0;
        for (const auto& g : constraint_data.same_rank_groups) {
            max_cap = std::max(max_cap, g.size());
        }
        // Capacity lower bound on host groups: floor for the soft descent (never probe below the achievable min).
        const size_t k_floor = (max_cap > 0) ? (graph_data.n_target + max_cap - 1) / max_cap : 1;

        // Cold one-shot hard cap: only when there is NO soft minimize to piggy-back on. When BOTH are set (the
        // common minimal-host case), we instead route the cap through the soft descent below, which warm-starts the
        // full-packing lock from an incremental solver full of learned clauses -- far more likely to actually hit K
        // than a cold solve, which otherwise just burns the whole conflict budget before falling back anyway.
        if (constraint_data.max_same_rank_groups_used > 0 && !constraint_data.minimize_same_rank_groups_used) {
            const size_t K = constraint_data.max_same_rank_groups_used;
            if (!topology_sat_max_groups_cap_capacity_feasible(constraint_data, graph_data.n_target, K)) {
                if (!quiet_mode) {
                    log_warning(
                        tt::LogFabric,
                        "Topology SAT: hard host-group cap k={} is infeasible for {} target(s) given same-rank "
                        "group capacities; skipping hard cap and falling back to soft minimize if enabled",
                        K,
                        graph_data.n_target);
                }
            } else {
                // Full-packing tightening is valid exactly when a used host must be completely filled: uniform capacity
                // and target count an exact multiple, i.e. K * max_cap == n_target.
                const bool full_packing = (max_cap > 0 && graph_data.n_target == K * max_cap);
                TopologySatSolver solver;
                TopologySatHardEncoding enc;
                auto t_enc = std::chrono::steady_clock::now();
                const bool enc_ok =
                    topology_sat_encode_hard_constraints(solver, graph_data, constraint_data, enc, validation_mode);
                if (profile) {
                    log_info(
                        tt::LogFabric,
                        "[topo-sat-profile] hard-cap: encode_hard_constraints total : {:.1f} ms (ok={}); CNF so far "
                        "{} vars, {} clauses, {} literals",
                        topology_sat_elapsed_ms(t_enc),
                        enc_ok,
                        solver.num_variables(),
                        solver.num_clauses(),
                        solver.num_literals());
                }
                bool amk_ok = false;
                if (enc_ok) {
                    // Warm-start: solve the base (ring-only) encoding first — cheap, since a plain ring embedding
                    // is easy — then pin that feasible assignment as sticky CaDiCaL phase hints. When the harder
                    // all-or-nothing host-packing clauses are added below, CDCL branches toward a *real* ring
                    // embedding first and only has to "repair" it toward host-alignment, instead of rediscovering a
                    // Hamiltonian ring from scratch under the packing coupling. Sound: phases are only branching
                    // hints, never constraints. Gated by TT_TOPO_SAT_HARDCAP_WARMSTART (default on) for A/B testing.
                    const bool warmstart = topology_sat_env_long("TT_TOPO_SAT_HARDCAP_WARMSTART", 1) != 0;
                    if (warmstart) {
                        auto t_warm = std::chrono::steady_clock::now();
                        const int warm_status = solve_with_symmetry_break(solver, enc);
                        int hints = 0;
                        if (warm_status == TopologySatSolver::kSat) {
                            for (size_t t = 0; t < enc.assign_lit.size(); ++t) {
                                for (size_t i = 0; i < enc.assign_lit[t].size(); ++i) {
                                    const int lit = enc.assign_lit[t][i];
                                    solver.phase(solver.val(lit) > 0 ? lit : -lit);
                                    ++hints;
                                }
                            }
                        }
                        if (profile) {
                            log_info(
                                tt::LogFabric,
                                "[topo-sat-profile] hard-cap: warm soft-solve : {:.1f} ms (status={}, phase_hints={})",
                                topology_sat_elapsed_ms(t_warm),
                                warm_status,
                                hints);
                        }
                    }
                    auto t_amk = std::chrono::steady_clock::now();
                    const size_t pc = solver.num_clauses();
                    const size_t pv = solver.num_variables();
                    amk_ok = topology_sat_encode_at_most_k_groups(solver, constraint_data, enc, K, full_packing);
                    if (profile) {
                        log_info(
                            tt::LogFabric,
                            "[topo-sat-profile] hard-cap: encode_at_most_k_groups (K={}, full_packing={}) : {:.1f} ms "
                            "(+{} clauses, +{} vars); CNF total {} vars, {} clauses, {} literals",
                            K,
                            full_packing,
                            topology_sat_elapsed_ms(t_amk),
                            solver.num_clauses() - pc,
                            solver.num_variables() - pv,
                            solver.num_variables(),
                            solver.num_clauses(),
                            solver.num_literals());
                    }
                }
                if (enc_ok && amk_ok) {
                    auto t_solve = std::chrono::steady_clock::now();
                    const int solve_status =
                        solve_limited_with_symmetry_break(solver, enc, kGroupObjectiveConflictBudget);
                    if (profile) {
                        log_info(
                            tt::LogFabric,
                            "[topo-sat-profile] hard-cap: solve_limited (budget={}) : {:.1f} ms (status={})",
                            kGroupObjectiveConflictBudget,
                            topology_sat_elapsed_ms(t_solve),
                            solve_status);
                    }
                    if (solve_status == TopologySatSolver::kSat && finalize_success(solver, enc)) {
                        if (!quiet_mode) {
                            log_info(tt::LogFabric, "Topology SAT: hard-capped host-group usage at {} group(s)", K);
                        }
                        return true;
                    }
                }
            }
            // Hard cap unsatisfiable / too hard within the budget -> fall back to the soft minimize (if enabled).
        }

        if (constraint_data.minimize_same_rank_groups_used) {
            TopologySatSolver solver;
            TopologySatHardEncoding enc;
            auto t_min_enc = std::chrono::steady_clock::now();
            const bool min_enc_ok =
                topology_sat_encode_hard_constraints(solver, graph_data, constraint_data, enc, validation_mode);
            if (profile) {
                log_info(
                    tt::LogFabric,
                    "[topo-sat-profile] minimize: encode_hard_constraints total : {:.1f} ms (ok={})",
                    topology_sat_elapsed_ms(t_min_enc),
                    min_enc_ok);
            }
            if (min_enc_ok) {
                auto t_min = std::chrono::steady_clock::now();
                std::vector<int> best_mapping;
                size_t best_k = 0;
                // If a hard cap K was also requested, hand it to the descent so it finishes with a warm full-packing
                // lock at K (reusing the descent's learned clauses) instead of a separate cold solve.
                const size_t hard_cap_k = constraint_data.max_same_rank_groups_used;
                bool hard_cap_met = false;
                // Small descent budget (cheap warm-up that bails fast from the hard deep steps) + large lock budget
                // (the full-packing lock that actually reaches K). This split is what keeps the end-to-end solve ~90s;
                // a single large budget lets a deep descent step grind for minutes before the lock runs.
                const bool min_ok = topology_sat_solve_minimize_groups(
                    solver,
                    enc,
                    constraint_data,
                    kGroupDescentConflictBudget,
                    k_floor,
                    best_mapping,
                    best_k,
                    hard_cap_k,
                    kGroupLockConflictBudget,  // strict: unbounded by default (run lock until k_min proven / UNSAT)
                    &hard_cap_met);
                if (profile) {
                    log_info(
                        tt::LogFabric,
                        "[topo-sat-profile] minimize: solve_minimize_groups : {:.1f} ms (ok={}, best_k={}, "
                        "hard_cap_k={}, hard_cap_met={})",
                        topology_sat_elapsed_ms(t_min),
                        min_ok,
                        best_k,
                        hard_cap_k,
                        hard_cap_met);
                }
                if (min_ok && !best_mapping.empty()) {
                    state.mapping = best_mapping;
                    std::fill(state.used.begin(), state.used.end(), false);
                    for (int gi : state.mapping) {
                        if (gi >= 0 && static_cast<size_t>(gi) < state.used.size()) {
                            state.used[static_cast<size_t>(gi)] = true;
                        }
                    }
                    if (!quiet_mode) {
                        if (hard_cap_k > 0 && hard_cap_met) {
                            log_info(
                                tt::LogFabric,
                                "Topology SAT: hard-capped host-group usage at {} group(s) (via warm full-packing lock)",
                                best_k);
                        } else {
                            log_info(
                                tt::LogFabric,
                                "Topology SAT: minimized host-group usage to {} group(s) (capacity lower bound {})",
                                best_k,
                                k_floor);
                        }
                    }
                    return true;
                }
            }
        }
        // Objective could not produce a model -> fall through to the normal preferred/plain solve for error paths.
    }

    bool has_preferred = false;
    for (size_t t = 0; t < graph_data.n_target && !has_preferred; ++t) {
        if (t < constraint_data.preferred_global_indices.size() &&
            !constraint_data.preferred_global_indices[t].empty()) {
            has_preferred = true;
        }
    }
    if (!has_preferred) {
        TopologySatSolver solver;
        TopologySatHardEncoding enc;
        return solve_hard_only(solver, enc);
    }

    TopologySatSolver solver;
    TopologySatHardEncoding enc;
    if (!topology_sat_encode_hard_constraints(solver, graph_data, constraint_data, enc, validation_mode)) {
        state.error_message = enc.trivial_reason.empty() ? std::string("Topology SAT: encoding failed (trivial UNSAT)")
                                                         : enc.trivial_reason;
        if (quiet_mode) {
            log_debug(tt::LogFabric, "{}", state.error_message);
        } else {
            log_error(tt::LogFabric, "{}", state.error_message);
        }
        return false;
    }
    std::vector<int> pref_hit_literals;
    topology_sat_append_preferred_hit_indicators(solver, enc, constraint_data, pref_hit_literals);

    if (!pref_hit_literals.empty()) {
        static constexpr size_t kExactPreferredLbMaxTargets = 10;
        static constexpr size_t kMidPreferredLbMaxTargets = 20;
        static constexpr size_t kPreferredLbDfsBudgetSmall = 80'000'000;
        static constexpr size_t kPreferredLbDfsBudgetMid = 400'000;
        const size_t nt = graph_data.n_target;
        size_t k_lb = 0;
        if (nt <= kExactPreferredLbMaxTargets) {
            k_lb =
                topology_sat_preferred_exact_lower_bound(graph_data, constraint_data, enc, kPreferredLbDfsBudgetSmall);
        } else if (nt <= kMidPreferredLbMaxTargets) {
            k_lb = topology_sat_preferred_exact_lower_bound(graph_data, constraint_data, enc, kPreferredLbDfsBudgetMid);
        } else {
            k_lb = topology_sat_preferred_greedy_lower_bound(graph_data, constraint_data, enc);
            if (k_lb == 0) {
                k_lb = topology_sat_preferred_exact_lower_bound(graph_data, constraint_data, enc, 600'000);
            }
        }
        if (k_lb > 0) {
            const size_t k_use = std::min(k_lb, pref_hit_literals.size());
            static constexpr size_t kPrefCardinalityCombClauses = 500000;
            std::string card_reason;
            if (!topology_sat_add_at_least_k_literals(
                    solver, pref_hit_literals, k_use, kPrefCardinalityCombClauses, &card_reason)) {
                if (!quiet_mode && !card_reason.empty()) {
                    log_debug(tt::LogFabric, "Topology SAT: preferred at-least-k skipped: {}", card_reason);
                }
            }
        }
    }

    if (validation_mode == ConnectionValidationMode::RELAXED) {
        static constexpr size_t kMaxRelaxedChannelLiteralsSingleSolve = 256;
        const size_t ch_mc_ub = topology_sat_relaxed_channel_threshold_literal_count_upper_bound(graph_data);
        if (ch_mc_ub <= kMaxRelaxedChannelLiteralsSingleSolve) {
            std::vector<int> ch_lits;
            std::string ch_reason;
            if (!topology_sat_append_relaxed_channel_threshold_literals(solver, enc, graph_data, ch_lits, &ch_reason)) {
                if (!ch_reason.empty() && !quiet_mode) {
                    log_debug(tt::LogFabric, "Topology SAT: relaxed channel threshold literals skipped: {}", ch_reason);
                }
            }
        } else if (!quiet_mode) {
            log_debug(
                tt::LogFabric,
                "Topology SAT: relaxed channel literals skipped for preferred pass (upper_bound {} > {})",
                ch_mc_ub,
                kMaxRelaxedChannelLiteralsSingleSolve);
        }
    }
    const int status = solve_with_symmetry_break(solver, enc);
    if (status != TopologySatSolver::kSat) {
        state.error_message = fmt::format(
            "Failed to find mapping (SAT): target graph with {} nodes cannot be embedded in global graph with {} "
            "nodes under hard constraints",
            graph_data.n_target,
            graph_data.n_global);
        if (quiet_mode) {
            log_debug(tt::LogFabric, "{}", state.error_message);
        } else {
            log_error(tt::LogFabric, "{}", state.error_message);
        }
        return false;
    }
    return finalize_success(solver, enc);
}

// ── topology_sat_search_n — enumerate up to max_solutions with blocking clauses ─────────────────

bool topology_sat_search_n(
    const TopologySatGraphView& graph_data,
    const TopologySatConstraintView& constraint_data,
    ConnectionValidationMode validation_mode,
    size_t max_solutions,
    std::vector<std::vector<int>>& all_mappings_out,
    bool quiet_mode,
    bool unique_shapes,
    const std::vector<std::vector<int>>& initial_forbidden_shape_keys,
    TopologySearchState& state) {
    state = TopologySearchState{};
    state.mapping.assign(graph_data.n_target, -1);
    state.used.assign(graph_data.n_global, false);
    all_mappings_out.clear();

    if (max_solutions == 0) {
        return false;
    }

    if (graph_data.n_target == 0) {
        all_mappings_out.push_back({});
        return true;
    }

    if (graph_data.n_global < graph_data.n_target) {
        return false;
    }

    // One CaDiCaL::Solver for the whole enumeration: encode once, then add blocking clauses and solve() in a loop.
    // (No full re-encode / new solver per model.)
    TopologySatSolver solver;
    solver.configure_for_blocking_clause_enumeration();
    TopologySatHardEncoding enc;
    if (!topology_sat_encode_hard_constraints(solver, graph_data, constraint_data, enc, validation_mode)) {
        return false;
    }

    for (const auto& shape_key : initial_forbidden_shape_keys) {
        std::vector<int> forbid_clause;
        topology_sat_build_shape_blocking_clause(enc, shape_key, forbid_clause);
        topology_sat_add_shape_clause_or_unsat(solver, enc, forbid_clause);
    }

    // Minimal-host occupancy objective (same strategy as the single solve): PRIME the solver with the warm descent
    // + full-packing lock and make the achieved cap PERMANENT (unit clause), so every enumerated solution occupies
    // the minimal host count -- not just the first. The primed model is the first solution; the loop below finds
    // the rest (warm, permanently capped, only a blocking clause added). See TOPOLOGY_OCCUPANCY_SOLVE_README §6.
    // Per-solve budget for the blocking-clause enumeration loop. DEFAULT 0 = UNBOUNDED: for --all-solutions we must
    // NOT give up on a conflict budget -- a bounded solve that hits its cap returns kUnknown, which is
    // indistinguishable from "no more solutions" and would silently truncate the enumeration (reporting fewer
    // solutions than exist and a false "exhaustive"). Unbounded means each solve runs to a definite kSat (another
    // solution) or kUnsat (genuinely exhausted). Set TT_TOPO_SAT_ENUM_BUDGET>0 only if you explicitly want a
    // best-effort truncated enumeration. (Dedicated var so it doesn't perturb the single-solve objective budget.)
    const int kEnumLoopConflictBudget = static_cast<int>(topology_sat_env_long("TT_TOPO_SAT_ENUM_BUDGET", 0));
    if (constraint_data.max_same_rank_groups_used > 0 || constraint_data.minimize_same_rank_groups_used) {
        const int kDescentBudget = static_cast<int>(topology_sat_env_long("TT_TOPO_SAT_DESCENT_BUDGET", 20'000));
        const int kLockBudget = static_cast<int>(topology_sat_env_long("TT_TOPO_SAT_LOCK_BUDGET", 0));
        size_t max_cap = 0;
        for (const auto& g : constraint_data.same_rank_groups) {
            max_cap = std::max(max_cap, g.size());
        }
        const size_t k_floor = (max_cap > 0) ? (graph_data.n_target + max_cap - 1) / max_cap : 1;
        const size_t hard_cap_k = constraint_data.max_same_rank_groups_used;
        std::vector<int> first_mapping;
        size_t best_k = 0;
        bool hard_cap_met = false;
        const bool primed = topology_sat_solve_minimize_groups(
            solver,
            enc,
            constraint_data,
            kDescentBudget,
            k_floor,
            first_mapping,
            best_k,
            hard_cap_k,
            kLockBudget,
            &hard_cap_met,
            /*make_cap_permanent=*/true);
        if (primed && !first_mapping.empty()) {
            all_mappings_out.push_back(first_mapping);
            if (!quiet_mode) {
                log_info(
                    tt::LogFabric,
                    "topology_sat_search_n: primed minimal-host enumeration at {} occupied host group(s) "
                    "(hard_cap_k={}, met={})",
                    best_k,
                    hard_cap_k,
                    hard_cap_met);
            }
            if (all_mappings_out.size() >= max_solutions ||
                !topology_sat_add_blocking_clause_for_mapping(solver, enc, all_mappings_out.back(), unique_shapes)) {
                // Reached the cap with the first solution, or can't block it -> done.
                if (!all_mappings_out.empty()) {
                    state.mapping = all_mappings_out.back();
                    std::fill(state.used.begin(), state.used.end(), false);
                    for (int gi : state.mapping) {
                        if (gi >= 0 && static_cast<size_t>(gi) < state.used.size()) {
                            state.used[static_cast<size_t>(gi)] = true;
                        }
                    }
                }
                return !all_mappings_out.empty();
            }
        }
        // If priming failed, fall through to the plain enumeration loop below (best-effort, uncapped).
    }

    using enum_clock = std::chrono::steady_clock;
    constexpr auto kEnumProgressLogInterval = std::chrono::seconds(5);
    // Eligible for an immediate first progress line, then at most once per kEnumProgressLogInterval.
    auto last_enum_progress_log = enum_clock::now() - kEnumProgressLogInterval;

    while (all_mappings_out.size() < max_solutions) {
        // Default (kEnumLoopConflictBudget==0): UNBOUNDED solve -- never give up on a budget, so we only stop on a
        // real kUnsat (genuine exhaustion), never on a kUnknown that would silently truncate. If a budget is set,
        // fall back to solve_limited (best-effort; kUnknown then stops the loop).
        const int status = (kEnumLoopConflictBudget > 0) ? solver.solve_limited(kEnumLoopConflictBudget)
                                                         : solver.solve();
        if (status != TopologySatSolver::kSat) {
            break;
        }

        std::vector<int> current_mapping;
        if (!topology_sat_decode_hard_solution(solver, enc, current_mapping)) {
            break;
        }
        all_mappings_out.push_back(std::move(current_mapping));

        // Progress: emit in non-quiet mode, and ALSO whenever profiling is on (so a long quiet enumeration -- e.g.
        // map_multi_mesh_to_physical_n runs quiet -- still gives a live per-solution count under TT_TOPO_SAT_PROFILE).
        if (!quiet_mode || topology_sat_profile_enabled()) {
            const auto now = enum_clock::now();
            const bool reached_cap = all_mappings_out.size() >= max_solutions;
            if (reached_cap || now - last_enum_progress_log >= kEnumProgressLogInterval) {
                log_info(
                    tt::LogFabric,
                    "topology_sat_search_n: found {} solution(s) so far (max={})",
                    all_mappings_out.size(),
                    max_solutions);
                last_enum_progress_log = now;
            }
        }

        if (!topology_sat_add_blocking_clause_for_mapping(solver, enc, all_mappings_out.back(), unique_shapes)) {
            break;
        }
    }

    if (!all_mappings_out.empty()) {
        state.mapping = all_mappings_out.back();
        std::fill(state.used.begin(), state.used.end(), false);
        for (int gi : state.mapping) {
            if (gi >= 0 && static_cast<size_t>(gi) < state.used.size()) {
                state.used[static_cast<size_t>(gi)] = true;
            }
        }
    }

    return !all_mappings_out.empty();
}

// ── Session bridge functions (public API — declared in topology_solver.hpp) ────

void topology_sat_session_destroy(TopologySatSession* p) noexcept { delete p; }

std::unique_ptr<TopologySatSession, TopologySatSessionDeleter> topology_sat_session_create_and_encode(
    const TopologySatGraphView& graph_data,
    const TopologySatConstraintView& constraint_data,
    TopologySatHardEncoding& enc,
    ConnectionValidationMode validation_mode) {
    auto session = std::unique_ptr<TopologySatSession, TopologySatSessionDeleter>(new TopologySatSession{});
    session->solver.configure_for_blocking_clause_enumeration();
    enc = {};
    if (!topology_sat_encode_hard_constraints(session->solver, graph_data, constraint_data, enc, validation_mode)) {
        return nullptr;
    }
    // Same minimal-host strategy as topology_sat_search_n: PRIME the solver with the warm descent + full-packing
    // lock, make the achieved cap PERMANENT, and stash the primed model so every incremental (.next) solution
    // occupies the minimal host count. See TOPOLOGY_OCCUPANCY_SOLVE_README §7.
    if (constraint_data.max_same_rank_groups_used > 0 || constraint_data.minimize_same_rank_groups_used) {
        const int kDescentBudget = static_cast<int>(topology_sat_env_long("TT_TOPO_SAT_DESCENT_BUDGET", 20'000));
        const int kLockBudget = static_cast<int>(topology_sat_env_long("TT_TOPO_SAT_LOCK_BUDGET", 0));
        // 0 = UNBOUNDED (default): never give up on a budget during .next enumeration -- see the rationale in
        // topology_sat_search_n. A kUnknown budget give-up would silently truncate the incremental enumeration.
        session->enum_loop_budget = static_cast<int>(topology_sat_env_long("TT_TOPO_SAT_ENUM_BUDGET", 0));
        size_t max_cap = 0;
        for (const auto& g : constraint_data.same_rank_groups) {
            max_cap = std::max(max_cap, g.size());
        }
        const size_t k_floor = (max_cap > 0) ? (graph_data.n_target + max_cap - 1) / max_cap : 1;
        const size_t hard_cap_k = constraint_data.max_same_rank_groups_used;
        std::vector<int> first_mapping;
        size_t best_k = 0;
        bool hard_cap_met = false;
        const bool primed = topology_sat_solve_minimize_groups(
            session->solver,
            enc,
            constraint_data,
            kDescentBudget,
            k_floor,
            first_mapping,
            best_k,
            hard_cap_k,
            kLockBudget,
            &hard_cap_met,
            /*make_cap_permanent=*/true);
        if (primed && !first_mapping.empty()) {
            session->primed_first_mapping = std::move(first_mapping);
            session->has_primed_mapping = true;
            log_info(
                tt::LogFabric,
                "Topology SAT enumeration session: primed minimal-host enumeration at {} occupied host group(s) "
                "(hard_cap_k={}, met={})",
                best_k,
                hard_cap_k,
                hard_cap_met);
        }
        // If priming failed, the session falls back to plain (unbounded, uncapped) enumeration.
    }
    return session;
}

bool topology_sat_session_add_blocking_clause(
    TopologySatSession* session,
    TopologySatHardEncoding& enc,
    const std::vector<int>& raw_mapping,
    bool unique_shapes) {
    return topology_sat_add_blocking_clause_for_mapping(session->solver, enc, raw_mapping, unique_shapes);
}

bool topology_sat_session_solve_and_decode(
    TopologySatSession* session, const TopologySatHardEncoding& enc, std::vector<int>& raw_out) {
    // First call after a minimal-host prime returns the primed model directly (already decoded); no extra solve.
    if (session->has_primed_mapping) {
        session->has_primed_mapping = false;
        raw_out = session->primed_first_mapping;
        return true;
    }
    // With an occupancy objective the solver is permanently capped -- bound each solve so enumeration terminates
    // (a distinct minimal-host packing can still be hard); on unknown/unsat we report no further solution.
    const int status = (session->enum_loop_budget > 0) ? session->solver.solve_limited(session->enum_loop_budget)
                                                        : session->solver.solve();
    if (status != TopologySatSolver::kSat) {
        return false;
    }
    return topology_sat_decode_hard_solution(session->solver, enc, raw_out);
}

}  // namespace tt::tt_fabric::detail
