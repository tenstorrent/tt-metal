// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <atomic>
#include <functional>
#include <mutex>
#include <thread>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
// Clause-sharing portfolio ENUMERATION state (TT_TOPO_SAT_SHARE=1 + TT_TOPO_SAT_PORTFOLIO=N). Defined below the
// portfolio drivers; forward-declared here so the session can own one.
struct TopologySatShareEnumState;

struct TopologySatSession {
    TopologySatSolver solver;
    // Minimal-host priming (see TOPOLOGY_OCCUPANCY_SOLVE_README §7): when the session carries an occupancy objective,
    // create_and_encode primes the solver (warm descent + full-packing lock) and makes the achieved cap PERMANENT.
    // The primed model is returned by the FIRST solve_and_decode; subsequent calls are bounded (warm + capped).
    std::vector<int> primed_first_mapping;
    bool has_primed_mapping = false;
    int enum_loop_budget = 0;  // >0 => bound each solve_and_decode with solve_limited (occupancy objective present)
    // Non-null => this session enumerates through the clause-sharing PORTFOLIO (N persistent cooperating workers)
    // instead of the single incremental solver above (which then only serves as the encode/var-numbering reference).
    std::unique_ptr<TopologySatShareEnumState> share_enum;
};

// ── Adjacency and Edge Helpers ────────────────────────────────────────────────
namespace {

// ── Phase profiling ───────────────────────────────────────────────────────────
// Emits per-phase wall-clock timings for the SAT encode/solve pipeline so we can attribute where a slow ring solve
// actually spends its time (domain build, AC-3, adjacency support, symmetry break, the solve itself, ...). Always
// collected: the lines go to DEBUG, so they cost nothing unless debug logging is on, and quiet_mode suppresses them
// entirely (quiet callers such as auto-discovery probes must stay silent even at debug level).
class TopologySatScopedTimer {
public:
    TopologySatScopedTimer(std::string label, bool quiet_mode) :
        label_(std::move(label)), start_(std::chrono::steady_clock::now()), quiet_(quiet_mode) {}
    ~TopologySatScopedTimer() {
        if (quiet_) {
            return;
        }
        const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count();
        log_debug(tt::LogFabric, "[topo-sat-profile] {} : {:.1f} ms", label_, ms);
    }

private:
    std::string label_;
    std::chrono::steady_clock::time_point start_;
    bool quiet_;
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

// ── Host-Usage Budget (minimize distinct same-rank global groups used) ────────
//
// Adds a hard "at most k_hosts distinct same-rank global groups (host partitions) are used" constraint.
// For every host group p that has at least one assignment literal, introduce an indicator h_p and add
// (¬x_{t,g} v h_p) for each assign literal whose global g belongs to group p — so using any global in p
// forces h_p true. Bounding the number of true h_p to k_hosts is encoded as "at least (P - k_hosts) of the
// h_p are false" via the existing at-least-k machinery over the negated indicator literals.
//
// Returns true if the budget was encoded (or is non-binding); false only if the at-least-k encoding reports
// the bound is trivially impossible (caller then tries a larger budget).
bool topology_sat_encode_host_group_budget(
    TopologySatSolver& solver,
    const TopologySatConstraintView& constraint_data,
    const TopologySatHardEncoding& enc,
    size_t k_hosts) {
    const auto& global_to_host = constraint_data.global_to_same_rank_group;
    const size_t num_groups = constraint_data.same_rank_groups.size();
    if (global_to_host.empty() || num_groups == 0) {
        return true;
    }

    // Collect assignment literals per host group (group labels are dense ids in [0, num_groups)).
    std::vector<std::vector<int>> host_assign_lits(num_groups);
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
            host_assign_lits[static_cast<size_t>(label)].push_back(lits[k]);
        }
    }

    // One "host used" indicator per non-empty group, with backward implication (used global => host used).
    std::vector<int> neg_host_lits;
    neg_host_lits.reserve(num_groups);
    for (size_t p = 0; p < num_groups; ++p) {
        if (host_assign_lits[p].empty()) {
            continue;
        }
        const int h = solver.declare_one_more_variable();
        for (int a : host_assign_lits[p]) {
            solver.add(-a);
            solver.add(h);
            solver.add(0);
        }
        neg_host_lits.push_back(-h);
    }

    const size_t num_present = neg_host_lits.size();
    if (num_present == 0 || k_hosts >= num_present) {
        return true;  // budget is not binding
    }

    static constexpr size_t kHostBudgetCombClauses = 500000;
    std::string reason;
    return topology_sat_add_at_least_k_literals(
        solver, neg_host_lits, num_present - k_hosts, kHostBudgetCombClauses, &reason);
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

// ── EXPERIMENT: minimal-host solve mode (TT_TOPO_SAT_MIN_MODE) ────────────────────────────────
//   0 baseline    : warm solve + soft descent + hard-cap lock (original).
//   1 skipdescent : warm solve + hard-cap lock only (skip the one-at-a-time descent).
//   2 greedy      : construct a host-packing greedily (no SAT search), verify by unit propagation.
enum class TopoMinMode { Baseline = 0, SkipDescent = 1, Greedy = 2, HardCapOnly = 3, AtMostK = 4 };
inline TopoMinMode topology_sat_min_mode() {
    switch (topology_sat_env_long("TT_TOPO_SAT_MIN_MODE", 0)) {
        case 1: return TopoMinMode::SkipDescent;
        case 2: return TopoMinMode::Greedy;
        case 3: return TopoMinMode::HardCapOnly;
        case 4: return TopoMinMode::AtMostK;
        default: return TopoMinMode::Baseline;
    }
}

// GREEDY minimal-host fill. Walk target adjacency (BFS), placing each target on an unused, adjacency-consistent
// global, preferring globals in an already-opened host group (fill a host before opening a new one) and compacting
// toward low group ids. Then VERIFY the constructed assignment by assuming it and unit-propagating: an invalid or
// dead-ended greedy simply returns false (never a wrong mapping). No SAT *search* — eliminates the descent/lock.
inline bool topology_sat_greedy_minhost_fill(
    TopologySatSolver& solver,
    const TopologySatGraphView& graph_data,
    const TopologySatHardEncoding& enc,
    const TopologySatConstraintView& constraint_data,
    const std::vector<int>& occ,
    std::vector<int>& best_mapping_out,
    size_t& best_k_out,
    bool quiet_mode) {
    const size_t nt = graph_data.n_target;
    const size_t ng = graph_data.n_global;
    best_mapping_out.clear();
    best_k_out = 0;
    if (nt == 0) {
        return false;
    }
    std::vector<size_t> chosen_slot(nt, SIZE_MAX);
    std::vector<char> global_used(ng, 0);
    std::vector<int> group_open(constraint_data.same_rank_groups.size(), 0);
    auto g_adj = [&](size_t a, size_t b) {
        const auto& adj = graph_data.global_adj_idx[a];
        return std::binary_search(adj.begin(), adj.end(), b);
    };
    auto group_of = [&](size_t g) -> int {
        return (g < constraint_data.global_to_same_rank_group.size()) ? constraint_data.global_to_same_rank_group[g]
                                                                      : -1;
    };
    // DFS (linear walk) order over targets: on a ring/line this yields a contiguous path 0,1,2,... (NOT the
    // both-ends BFS order 0,1,N-1,2,...), so consecutive stages fill one host before the walk moves on.
    std::vector<size_t> order;
    order.reserve(nt);
    {
        std::vector<char> seen(nt, 0);
        for (size_t s = 0; s < nt; ++s) {
            if (seen[s]) {
                continue;
            }
            std::vector<size_t> stack{s};
            while (!stack.empty()) {
                const size_t u = stack.back();
                stack.pop_back();
                if (seen[u]) {
                    continue;
                }
                seen[u] = 1;
                order.push_back(u);
                for (size_t v : graph_data.target_adj_idx[u]) {
                    if (!seen[v]) {
                        stack.push_back(v);
                    }
                }
            }
        }
    }
    // Capacity lower bound: don't open more than k_min host groups (forces packing to the minimum).
    size_t max_cap = 0;
    for (const auto& grp : constraint_data.same_rank_groups) {
        max_cap = std::max(max_cap, grp.size());
    }
    const size_t k_min = (max_cap > 0) ? (nt + max_cap - 1) / max_cap : constraint_data.same_rank_groups.size();
    size_t opened = 0;              // distinct host groups currently opened
    long visit_budget = 3'000'000;  // backtracking cap -> fail (never hang) if no clean packing is found fast
    // Backtracking DFS in BFS (adjacency) order, guided by the host-fill heuristic, pruned to <= k_min groups.
    std::function<bool(size_t)> place = [&](size_t oi) -> bool {
        if (--visit_budget < 0) {
            return false;
        }
        if (oi == order.size()) {
            return true;
        }
        const size_t t = order[oi];
        const auto& globs = enc.allowed_global_idx[t];
        std::vector<std::pair<long, size_t>> cands;  // (score, slot k)
        for (size_t k = 0; k < globs.size(); ++k) {
            const size_t g = globs[k];
            if (global_used[g]) {
                continue;
            }
            bool adj_ok = true;
            for (size_t tn : graph_data.target_adj_idx[t]) {
                if (chosen_slot[tn] == SIZE_MAX) {
                    continue;
                }
                if (!g_adj(g, enc.allowed_global_idx[tn][chosen_slot[tn]])) {
                    adj_ok = false;
                    break;
                }
            }
            if (!adj_ok) {
                continue;
            }
            const int grp = group_of(g);
            const bool opens_new = (grp < 0) || (group_open[grp] == 0);
            if (opens_new && opened >= k_min) {
                continue;  // k_min pruning: refuse to exceed the minimal host count
            }
            long score = 0;
            if (grp >= 0 && group_open[grp] > 0) {
                score += 1000000;  // fill an already-opened host first
            }
            if (grp >= 0) {
                score -= grp;  // compact toward low group ids
            }
            cands.emplace_back(score, k);
        }
        std::sort(cands.begin(), cands.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        for (const auto& [sc, k] : cands) {
            (void)sc;
            const size_t g = globs[k];
            const int grp = group_of(g);
            const bool opens_new = (grp < 0) || (group_open[grp] == 0);
            chosen_slot[t] = k;
            global_used[g] = 1;
            if (grp >= 0) {
                ++group_open[grp];
            }
            if (opens_new) {
                ++opened;
            }
            if (place(oi + 1)) {
                return true;
            }
            if (opens_new) {
                --opened;
            }
            if (grp >= 0) {
                --group_open[grp];
            }
            global_used[g] = 0;
            chosen_slot[t] = SIZE_MAX;
        }
        return false;
    };
    if (!place(0)) {
        if (!quiet_mode) {
            log_debug(
                tt::LogFabric,
                "[topo-sat-profile]   greedy.fill : no <= k_min={} packing found (dead-end/budget) -> greedy FAILED",
                k_min);
        }
        return false;
    }
    // Verify by assumption + unit propagation. Correct-by-check: a bad construction returns UNSAT here.
    for (size_t t = 0; t < nt; ++t) {
        solver.assume(enc.assign_lit[t][chosen_slot[t]]);
    }
    if (solver.solve() != TopologySatSolver::kSat) {
        if (!quiet_mode) {
            log_debug(tt::LogFabric, "[topo-sat-profile]   greedy.verify : constructed assignment UNSAT -> greedy FAILED");
        }
        return false;
    }
    if (!topology_sat_decode_hard_solution(solver, enc, best_mapping_out)) {
        return false;
    }
    size_t c = 0;
    for (int o : occ) {
        if (solver.val(o) == o) {
            ++c;
        }
    }
    best_k_out = c;
    return !best_mapping_out.empty();
}

// GOAL-1 base-embedding warm-start (TT_TOPO_SAT_BASE_WARMHINT=1). Construct a ring/line embedding by an
// adjacency DFS (first-fit, NO host constraint — Goal 1 ignores minimization) and phase-hint it so CaDiCaL
// branches toward it. Best-effort and always sound: phases are branching preferences only; partial or
// imperfect hints never change correctness. Returns the number of phase hints applied.
inline int topology_sat_apply_base_warmhint(
    TopologySatSolver& solver, const TopologySatGraphView& graph_data, const TopologySatHardEncoding& enc) {
    const size_t nt = graph_data.n_target;
    const size_t ng = graph_data.n_global;
    if (nt == 0) {
        return 0;
    }
    std::vector<size_t> chosen(nt, SIZE_MAX);
    std::vector<char> used(ng, 0);
    auto g_adj = [&](size_t a, size_t b) {
        const auto& adj = graph_data.global_adj_idx[a];
        return std::binary_search(adj.begin(), adj.end(), b);
    };
    std::vector<size_t> order;
    order.reserve(nt);
    {
        std::vector<char> seen(nt, 0);
        for (size_t s = 0; s < nt; ++s) {
            if (seen[s]) {
                continue;
            }
            std::vector<size_t> st{s};
            while (!st.empty()) {
                const size_t u = st.back();
                st.pop_back();
                if (seen[u]) {
                    continue;
                }
                seen[u] = 1;
                order.push_back(u);
                for (size_t v : graph_data.target_adj_idx[u]) {
                    if (!seen[v]) {
                        st.push_back(v);
                    }
                }
            }
        }
    }
    for (size_t t : order) {
        const auto& globs = enc.allowed_global_idx[t];
        long pick = -1;
        for (size_t k = 0; k < globs.size(); ++k) {
            const size_t g = globs[k];
            if (used[g]) {
                continue;
            }
            bool ok = true;
            for (size_t tn : graph_data.target_adj_idx[t]) {
                if (chosen[tn] == SIZE_MAX) {
                    continue;
                }
                if (!g_adj(g, enc.allowed_global_idx[tn][chosen[tn]])) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                pick = static_cast<long>(k);
                break;
            }
        }
        if (pick < 0) {
            continue;  // dead-end for this target: leave unhinted (partial hint is still sound)
        }
        chosen[t] = static_cast<size_t>(pick);
        used[globs[pick]] = 1;
    }
    int hints = 0;
    for (size_t t = 0; t < nt; ++t) {
        if (chosen[t] == SIZE_MAX) {
            continue;
        }
        solver.phase(enc.assign_lit[t][chosen[t]]);
        ++hints;
    }
    return hints;
}

// SOFT: minimize the number of occupied groups, best-effort. Takes one warm feasible solve, then descends an
// assumable "at most (current-1)" budget under a per-step conflict cap, keeping the best (fewest-group) model.
// Never turns a feasible instance UNSAT (step 1 is unconstrained). Writes the best mapping to best_mapping_out and
// its group count to best_k_out; returns true on any feasible model. `k_floor` stops the descent once reached
// (e.g. the capacity lower bound) so we don't waste solves probing below the achievable minimum.
inline bool topology_sat_solve_minimize_groups(
    const TopologySatGraphView& graph_data,
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
    bool make_cap_permanent = false,
    bool quiet_mode = false) {
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

    // EXPERIMENT dispatch: greedy-only host-fill (no SAT search). Returns the greedy result directly; on greedy
    // failure returns false (greedy-only has no fallback -- the caller/harness records it as a failure).
    const TopoMinMode min_mode = topology_sat_min_mode();
    if (min_mode == TopoMinMode::Greedy) {
        const auto t_greedy = std::chrono::steady_clock::now();
        const bool ok =
            topology_sat_greedy_minhost_fill(solver, graph_data, enc, constraint_data, occ, best_mapping_out, best_k_out, quiet_mode);
        if (!quiet_mode) {
            log_debug(
                tt::LogFabric,
                "[topo-sat-profile]   greedy.minhost_fill : {:.1f} ms (ok={}, occupied={}, num_present={})",
                topology_sat_elapsed_ms(t_greedy),
                ok,
                best_k_out,
                num_present);
        }
        return ok;
    }

    if (!quiet_mode) {
        std::map<size_t, int> size_hist;
        for (const auto& upg : used_per_group) {
            ++size_hist[upg.size()];
        }
        std::string hist;
        for (const auto& [sz, cnt] : size_hist) {
            hist += fmt::format("{}x{} ", cnt, sz);
        }
        log_debug(
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

    const bool profile = !quiet_mode;

    // HYBRID (TT_TOPO_SAT_GIMSATUL=1): delegate each heavy SAT solve to the external gimsatul binary while our
    // CaDiCaL keeps driving descent/decode. `assumption` (a counter/cap literal, 0 = none) is baked into gimsatul's
    // CNF as a temporary hard unit (gimsatul has no assume()); on kSat our val() reflects gimsatul's model so
    // count_occupied()/decode work unchanged. Returns 0 (unknown, e.g. binary missing) -> native fallback below.
    const bool use_gimsatul = topology_sat_env_long("TT_TOPO_SAT_GIMSATUL", 0) != 0;
    const int gimsatul_threads = static_cast<int>(topology_sat_env_long("TT_TOPO_SAT_GIMSATUL_THREADS", 32));
    // TT_TOPO_SAT_GIM_FIRST=1: gimsatul does only the FIRST (warm feasible) solve; the descent then runs on native
    // incremental CaDiCaL, warm-started from gimsatul's model via phase hints. gim_active is flipped off after the
    // warm solve. (Without it, every solve is delegated to gimsatul -- re-export + cold re-solve per descent step.)
    const bool gim_first_only = topology_sat_env_long("TT_TOPO_SAT_GIM_FIRST", 0) != 0;
    // Same size gate as the other experiment hooks: only the big inter-mesh solve is worth a subprocess round-trip
    // (and only it participates in the distilled pool's formula family).
    bool gim_active = use_gimsatul && solver.num_variables() > 5000;
    // `cap_k` = the host cap the `assumption` literal enforces for this solve (0 = uncapped); with TT_TOPO_SAT_POOL=1
    // it bounds which distilled pool entries the gimsatul dump may inject (see gimsatul_solve).
    auto delegated_solve = [&](int assumption, int native_budget, size_t cap_k) -> int {
        if (gim_active) {
            std::vector<int> units;
            if (assumption != 0) {
                units.push_back(assumption);
            }
            const int st = solver.gimsatul_solve(gimsatul_threads, units, assumption != 0 ? cap_k : 0);
            if (st != 0) {
                return st;  // gimsatul returned a verdict (kSat/kUnsat)
            }
            // st == 0 (no binary / parse fail) -> fall through to the native solve
        }
        if (assumption != 0) {
            solver.assume(assumption);
        }
        return (native_budget > 0) ? solver.solve_limited(native_budget) : solver.solve();
    };

    // EXPERIMENT mode 3 (HardCapOnly): skip the warm feasible solve AND the descent -- go straight to a single
    // cold all-or-nothing hard-cap solve at hard_cap_k. Tests whether the warm-start (mode 1) actually matters.
    if (min_mode == TopoMinMode::HardCapOnly && hard_cap_k > 0 && hard_cap_k < num_present) {
        topology_sat_add_all_or_nothing_tightening(solver, occ, used_per_group);
        const int bound = atmost_lit(hard_cap_k);
        // EXPERIMENT hook: dump the HOST-CAP-INCLUSIVE CNF. write_dimacs ignores assumptions, so bake the atmost
        // bound in as a HARD unit clause first -- otherwise the dumped CNF would silently drop the host cap and be
        // trivially SAT. This lets an external one-shot solver (gimsatul) attempt the full with-host-min problem.
        if (const char* dp = std::getenv("TT_TOPO_SAT_DUMP_DIMACS");
            dp != nullptr && dp[0] != '\0' && solver.num_variables() > 5000) {
            if (bound != 0) {
                solver.add(bound);
                solver.add(0);
                solver.note_permanent_cap(hard_cap_k);
            }
            const bool okw = solver.write_dimacs(dp);
            log_info(
                tt::LogFabric,
                "[topo-sat] TT_TOPO_SAT_DUMP_DIMACS hardcap(occupied<={}) -> {} ({}), {} vars, {} clauses (wrapper count)",
                hard_cap_k,
                dp,
                okw ? "ok" : "FAILED",
                solver.num_variables(),
                solver.num_clauses());
            return false;
        }
        const auto t_hc = std::chrono::steady_clock::now();
        const int st = delegated_solve(bound, 0, hard_cap_k);
        bool ok = false;
        if (st == TopologySatSolver::kSat) {
            best_k_out = count_occupied();
            topology_sat_decode_hard_solution(solver, enc, best_mapping_out);
            if (hard_cap_met_out != nullptr) {
                *hard_cap_met_out = (best_k_out <= hard_cap_k);
            }
            ok = !best_mapping_out.empty();
            // Enumeration support (search_n / session prime): make the hard cap PERMANENT so every subsequent
            // solve/enumeration step -- native OR a GIM_EVERY gimsatul dump of the clause tape -- stays capped.
            // (The delegated/assumed bound above is one-shot; without this unit the post-prime enumeration would
            // silently run uncapped in mode 3.)
            if (ok && make_cap_permanent && bound != 0) {
                solver.add(bound);
                solver.add(0);
                solver.note_permanent_cap(hard_cap_k);
                if (profile) {
                    log_debug(
                        tt::LogFabric,
                        "[topo-sat-profile]   hardcap_only.permanent_cap : asserted <= {} occupied (unit clause)",
                        hard_cap_k);
                }
            }
        }
        if (profile) {
            log_debug(
                tt::LogFabric,
                "[topo-sat-profile]   hardcap_only target<={} : {:.1f} ms (status={}, occupied={}, ok={})",
                hard_cap_k,
                topology_sat_elapsed_ms(t_hc),
                st,
                best_k_out,
                ok);
        }
        return ok;
    }

    // EXPERIMENT mode 4 (AtMostK): assert "<= k_min occupied" as a HARD counter unit (no all-or-nothing
    // tightening, works for non-exact fills), then do ONE plain solve. No warm, no descent.
    if (min_mode == TopoMinMode::AtMostK) {
        const size_t target = (hard_cap_k > 0) ? hard_cap_k : std::max<size_t>(k_floor, 1);
        const int bound = atmost_lit(target);
        if (bound != 0) {
            solver.add(bound);
            solver.add(0);
            solver.note_permanent_cap(target);
        }
        const auto t_am = std::chrono::steady_clock::now();
        const int st = solver.solve();
        bool ok = false;
        if (st == TopologySatSolver::kSat) {
            best_k_out = count_occupied();
            topology_sat_decode_hard_solution(solver, enc, best_mapping_out);
            if (hard_cap_met_out != nullptr) {
                *hard_cap_met_out = (best_k_out <= target);
            }
            ok = !best_mapping_out.empty();
        }
        if (profile) {
            log_debug(
                tt::LogFabric,
                "[topo-sat-profile]   atmostk target<={} : {:.1f} ms (status={}, occupied={}, ok={})",
                target,
                topology_sat_elapsed_ms(t_am),
                st,
                best_k_out,
                ok);
        }
        return ok;
    }

    auto t_warm = std::chrono::steady_clock::now();
    if (delegated_solve(0, 0, 0) != TopologySatSolver::kSat) {  // step 1: warm feasible model
        if (profile) {
            log_debug(
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
        log_debug(
            tt::LogFabric,
            "[topo-sat-profile]   minimize.warm_solve : {:.1f} ms (SAT, occupied={}, num_present={})",
            topology_sat_elapsed_ms(t_warm),
            best_k_out,
            num_present);
    }
    // HYBRID gimsatul-first: gimsatul did the heavy warm feasible solve; hand its model to CaDiCaL as phase hints
    // and run the rest of the descent/lock on native incremental CaDiCaL (learned-clause reuse across steps).
    if (gim_active && gim_first_only) {
        solver.phase_hint_from_last_gimsatul_model();
        gim_active = false;
    }

    const size_t floor = std::max<size_t>(k_floor, 1);
    // TT_TOPO_SAT_SKIP_DESCENT: skip the one-host-at-a-time soft descent and rely on the direct k_min hard-cap
    // lock below (all-or-nothing packing at the target). Experiment: for dense packings (e.g. 2x4 64-stage,
    // 24->16 descent) the incremental descent dominates while a single capped solve at k_min lands directly.
    const bool skip_descent =
        (min_mode == TopoMinMode::SkipDescent) || (topology_sat_env_long("TT_TOPO_SAT_SKIP_DESCENT", 0) != 0);
    size_t iter = 0;
    while (!skip_descent && best_k_out > floor) {
        const size_t target_k = best_k_out - 1;
        const int bound = atmost_lit(target_k);
        if (bound == 0) {
            break;
        }
        auto t_iter = std::chrono::steady_clock::now();
        const int st = delegated_solve(bound, conflict_cap, target_k);
        ++iter;
        if (st == TopologySatSolver::kSat) {
            best_k_out = count_occupied();  // may drop by more than one
            topology_sat_decode_hard_solution(solver, enc, best_mapping_out);
            if (profile) {
                log_debug(
                    tt::LogFabric,
                    "[topo-sat-profile]   minimize.descent[{}] target<={} : {:.1f} ms (SAT, now occupied={})",
                    iter,
                    target_k,
                    topology_sat_elapsed_ms(t_iter),
                    best_k_out);
            }
        } else {
            if (profile) {
                log_debug(
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
        const int st = delegated_solve(bound, hard_conflict_cap, hard_cap_k);
        if (st == TopologySatSolver::kSat) {
            best_k_out = count_occupied();
            topology_sat_decode_hard_solution(solver, enc, best_mapping_out);
            if (hard_cap_met_out != nullptr) {
                *hard_cap_met_out = (best_k_out <= hard_cap_k);
            }
        }
        if (profile) {
            log_debug(
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
            solver.note_permanent_cap(best_k_out);
            if (profile) {
                log_debug(
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
    ConnectionValidationMode validation_mode,
    bool quiet_mode) {
    enc = TopologySatHardEncoding{};
    const size_t nt = graph_data.n_target;

    if (nt == 0) {
        return true;
    }

    enc.allowed_global_idx.resize(nt);
    enc.assign_lit.resize(nt);

    const bool profile = !quiet_mode;
    auto phase_start = std::chrono::steady_clock::now();
    size_t prev_clauses = solver.num_clauses();
    size_t prev_vars = solver.num_variables();
    auto mark = [&](const char* name) {
        if (profile) {
            const size_t dc = solver.num_clauses() - prev_clauses;
            const size_t dv = solver.num_variables() - prev_vars;
            log_debug(
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

// GOAL-1 / Design B: parallel seed portfolio (TT_TOPO_SAT_PORTFOLIO=N). Spawn N worker threads on rank 0, each
// with its OWN CaDiCaL solver + independent encoding + a distinct seed (optionally fastsat). The first worker to
// reach SAT sets `done`, which every worker's terminator polls and aborts on. The winning mapping is returned.
// SAT solve time is highly seed-sensitive, so wall ~= fastest-of-N (bounded by free cores). Encode is serialized
// (cheap, ~0.1s) to avoid any shared-state races; the parallel win is entirely in the solve.
inline bool topology_sat_run_portfolio(
    const TopologySatGraphView& graph_data,
    const TopologySatConstraintView& constraint_data,
    ConnectionValidationMode validation_mode,
    int n_workers,
    long base_seed,
    bool fastsat,
    bool quiet_mode,
    std::vector<int>& mapping_out) {
    std::atomic<bool> done{false};
    std::mutex mtx;  // guards encode serialization + the winner claim
    std::vector<int> best;
    int winner = -1;
    double win_ms = 0.0;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(n_workers));
    for (int k = 0; k < n_workers; ++k) {
        workers.emplace_back([&, k]() {
            TopologySatSolver solver;
            solver.set_cancel_flag(&done);
            (void)solver.set_option("seed", static_cast<int>((base_seed >= 0 ? base_seed : 0) + k));
            if (fastsat) {
                (void)solver.set_option("target", 2);
                (void)solver.set_option("phase", 1);
            }
            TopologySatHardEncoding enc;
            {
                std::lock_guard<std::mutex> lk(mtx);  // serialize encode (cheap; sidesteps any shared-state race)
                if (done.load(std::memory_order_relaxed)) {
                    return;
                }
                if (!topology_sat_encode_hard_constraints(
                        solver, graph_data, constraint_data, enc, validation_mode, /*quiet_mode=*/true)) {
                    return;
                }
            }
            const int assumption = topology_sat_symmetry_assumption_lit(graph_data, enc);
            int status;
            if (assumption != 0) {
                solver.assume(assumption);
                status = solver.solve();
                if (status != TopologySatSolver::kSat && !done.load(std::memory_order_relaxed)) {
                    status = solver.solve();  // retry without the symmetry assumption (it is one-shot)
                }
            } else {
                status = solver.solve();
            }
            if (status != TopologySatSolver::kSat) {
                return;  // UNSAT/unknown or cancelled
            }
            std::vector<int> m;
            if (!topology_sat_decode_hard_solution(solver, enc, m)) {
                return;
            }
            std::lock_guard<std::mutex> lk(mtx);
            if (!done.exchange(true)) {
                best = std::move(m);
                winner = k;
                win_ms = topology_sat_elapsed_ms(t0);
            }
        });
    }
    for (auto& w : workers) {
        w.join();
    }
    if (winner < 0) {
        return false;
    }
    mapping_out = std::move(best);
    if (!quiet_mode) {
        log_info(
            tt::LogFabric,
            "[topo-sat-profile] portfolio: {}-way, winner=seed_offset {} in {:.1f} ms (fastsat={})",
            n_workers,
            winner,
            win_ms,
            fastsat);
    }
    return true;
}

// Clause-sharing portfolio. Like topology_sat_run_portfolio, but the N seed workers COOPERATE: each connects a Learner
// that publishes its short learned clauses to a shared pool, and between conflict-budget windows imports peers' clauses
// via add(). This reproduces gimsatul-style clause sharing while every worker stays a full incremental CaDiCaL (BVE and
// warmth intact) -- CaDiCaL's ExternalPropagator import path was rejected because it freezes observed vars and disables
// elimination. Sharing is sound + model-preserving (every shared clause is entailed by the common base formula). Like
// the race, the first worker to a model cancels the rest and its mapping is returned.
inline bool topology_sat_run_sharing_portfolio(
    const TopologySatGraphView& graph_data,
    const TopologySatConstraintView& constraint_data,
    ConnectionValidationMode validation_mode,
    int n_workers,
    long base_seed,
    bool fastsat,
    int share_budget,
    int share_max_size,
    bool quiet_mode,
    std::vector<int>& mapping_out) {
    std::atomic<bool> done{false};
    std::mutex mtx;
    ClauseSharingPool pool;
    std::atomic<long> shared_total{0};
    std::vector<int> best;
    int winner = -1;
    double win_ms = 0.0;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(n_workers));
    for (int k = 0; k < n_workers; ++k) {
        workers.emplace_back([&, k]() {
            TopologySatSolver solver;
            solver.set_cancel_flag(&done);
            (void)solver.set_option("seed", static_cast<int>((base_seed >= 0 ? base_seed : 0) + k));
            if (fastsat) {
                (void)solver.set_option("target", 2);
                (void)solver.set_option("phase", 1);
            }
            TopologySatHardEncoding enc;
            {
                std::lock_guard<std::mutex> lk(mtx);
                if (done.load(std::memory_order_relaxed)) {
                    return;
                }
                if (!topology_sat_encode_hard_constraints(
                        solver, graph_data, constraint_data, enc, validation_mode, /*quiet_mode=*/true)) {
                    return;
                }
            }
            // Connect export AFTER encode so only learned (not input) clauses are published.
            solver.enable_clause_export(&pool, k, share_max_size);
            const int assumption = topology_sat_symmetry_assumption_lit(graph_data, enc);

            // Budgeted incremental loop: solve a window, import peers' clauses, repeat until SAT/UNSAT/cancel.
            bool first = true;
            std::size_t cursor = 0;
            std::vector<std::vector<int>> imported;
            for (;;) {
                if (done.load(std::memory_order_relaxed)) {
                    return;
                }
                if (first && assumption != 0) {
                    solver.assume(assumption);  // one-shot hint (retracted after the solve)
                }
                const int status = solver.solve_limited(share_budget);
                if (status == TopologySatSolver::kSat) {
                    std::vector<int> m;
                    if (!topology_sat_decode_hard_solution(solver, enc, m)) {
                        return;
                    }
                    std::lock_guard<std::mutex> lk(mtx);
                    if (!done.exchange(true)) {
                        best = std::move(m);
                        winner = k;
                        win_ms = topology_sat_elapsed_ms(t0);
                    }
                    return;
                }
                if (status == TopologySatSolver::kUnsat) {
                    if (first && assumption != 0) {
                        first = false;  // the hint may have forced UNSAT -- retry the window without it
                        continue;
                    }
                    return;  // genuine UNSAT for this worker
                }
                // status == 0: conflict budget exhausted (or cancelled) -> import peers' clauses and continue.
                first = false;
                imported.clear();
                pool.drain(k, cursor, imported);
                for (const auto& cl : imported) {
                    for (const int lit : cl) {
                        solver.add(lit);
                    }
                    solver.add(0);
                }
                shared_total.fetch_add(static_cast<long>(imported.size()), std::memory_order_relaxed);
            }
        });
    }
    for (auto& w : workers) {
        w.join();
    }
    if (winner < 0) {
        return false;
    }
    mapping_out = std::move(best);
    if (!quiet_mode) {
        log_info(
            tt::LogFabric,
            "[topo-sat-profile] sharing-portfolio: {}-way, winner=seed_offset {} in {:.1f} ms (budget={}, max_size={}, "
            "pool={}, imports={})",
            n_workers,
            winner,
            win_ms,
            share_budget,
            share_max_size,
            pool.size(),
            shared_total.load(std::memory_order_relaxed));
    }
    return true;
}

// ── Clause-sharing portfolio ENUMERATION (TT_TOPO_SAT_SHARE=1 + TT_TOPO_SAT_PORTFOLIO=N, occupancy cap) ─────────
// Extends the sharing portfolio from the single solve to MULTI-SOLUTION enumeration. N persistent CaDiCaL workers
// each hold their own deterministic encode of the SAME base formula (deterministic encode => identical variable
// numbering => blocking clauses reconstruct exactly), the hard host cap as a PERMANENT unit (mode-3 semantics:
// all-or-nothing tightening + occupancy counter + cap unit), and a Learner exporting short learned clauses to one
// shared pool. Each solution is a cooperative ROUND: workers race budgeted solve windows, importing peers' clauses
// between windows; the first model cancels the rest. The winning mapping is then blocked in EVERY worker before the
// next round starts (so all workers always agree on the found-solution set and dedup stays global), and workers
// PERSIST across rounds -- learned clauses, imports and saved phases carry over. Soundness: worker formulas are
// add-only supersets of one common base, every shared clause is learned from (hence entailed by) that family, and a
// genuine kUnsat from any worker proves global exhaustion (all workers' formulas are equisatisfiable: they differ
// only by entailed clauses).
struct TopologySatShareEnumState {
    int n_workers = 0;
    int share_budget = 2000;
    int share_max_size = 8;
    size_t hard_cap_k = 0;
    bool exhausted = false;
    size_t rounds = 0;
    std::vector<std::unique_ptr<TopologySatSolver>> solvers;
    std::vector<TopologySatHardEncoding> encs;
    ClauseSharingPool pool;
    std::vector<std::size_t> cursors;      // per-worker drain cursor into `pool`
    std::atomic<bool> round_done{false};   // per-round cancel flag (reset at each round start)
    std::atomic<long> imported_total{0};
};

inline bool topology_sat_share_enum_enabled() {
    return topology_sat_env_long("TT_TOPO_SAT_SHARE", 0) != 0 &&
           topology_sat_env_long("TT_TOPO_SAT_PORTFOLIO", 0) > 1;
}

// EXPERIMENT (rebase adaptation): the old branch's mapper set MappingConstraints::max_same_rank_groups_used to the
// capacity lower bound k_min (ceil(n_target / max host-group capacity)); the rebased base removed that hard-cap API
// (its default single-solve path is an upward host-budget walk instead). The experiment paths reconstruct the same
// hard cap locally, and only when an experiment knob (TT_TOPO_SAT_MIN_MODE / TT_TOPO_SAT_SHARE+PORTFOLIO) is set --
// with no experiment env every path is the base's default.
inline bool topology_sat_min_mode_env_set() {
    const char* v = std::getenv("TT_TOPO_SAT_MIN_MODE");
    return v != nullptr && v[0] != '\0';
}
inline bool topology_sat_experiment_minhost_enabled(const TopologySatConstraintView& constraint_data) {
    return constraint_data.minimize_same_rank_groups_used &&
           (topology_sat_min_mode_env_set() || topology_sat_share_enum_enabled());
}
inline size_t topology_sat_experiment_hard_cap_k(
    const TopologySatGraphView& graph_data, const TopologySatConstraintView& constraint_data) {
    if (!topology_sat_experiment_minhost_enabled(constraint_data)) {
        return 0;
    }
    size_t max_cap = 0;
    for (const auto& g : constraint_data.same_rank_groups) {
        max_cap = std::max(max_cap, g.size());
    }
    return (max_cap > 0) ? (graph_data.n_target + max_cap - 1) / max_cap : 0;
}

// Build the N-worker state. Returns nullptr when the formula is too small to be the inter-mesh solve (<= 5000 vars,
// same gate as the other experiment hooks -- the caller then falls back to the normal single-solver path) or when a
// worker fails to encode.
inline std::unique_ptr<TopologySatShareEnumState> topology_sat_share_enum_init(
    const TopologySatGraphView& graph_data,
    const TopologySatConstraintView& constraint_data,
    ConnectionValidationMode validation_mode,
    size_t hard_cap_k,
    bool quiet_mode) {
    (void)quiet_mode;  // path markers below are deliberately always-on (experiment evidence)
    auto st = std::make_unique<TopologySatShareEnumState>();
    st->n_workers = static_cast<int>(topology_sat_env_long("TT_TOPO_SAT_PORTFOLIO", 0));
    st->share_budget = static_cast<int>(topology_sat_env_long("TT_TOPO_SAT_SHARE_BUDGET", 2000));
    st->share_max_size = static_cast<int>(topology_sat_env_long("TT_TOPO_SAT_SHARE_MAX_SIZE", 8));
    st->hard_cap_k = hard_cap_k;
    const long base_seed = topology_sat_env_long("TT_TOPO_SAT_SEED", 0);
    const bool fastsat = topology_sat_env_long("TT_TOPO_SAT_FASTSAT", 0) != 0;
    st->cursors.assign(static_cast<size_t>(st->n_workers), 0);
    st->encs.resize(static_cast<size_t>(st->n_workers));
    for (int k = 0; k < st->n_workers; ++k) {
        auto solver = std::make_unique<TopologySatSolver>();
        solver->configure_for_blocking_clause_enumeration();
        (void)solver->set_option("seed", static_cast<int>((base_seed >= 0 ? base_seed : 0) + k));
        if (fastsat) {
            (void)solver->set_option("target", 2);
            (void)solver->set_option("phase", 1);
        }
        if (!topology_sat_encode_hard_constraints(
                *solver, graph_data, constraint_data, st->encs[static_cast<size_t>(k)], validation_mode,
                /*quiet_mode=*/true)) {
            return nullptr;
        }
        if (k == 0 && solver->num_variables() <= 5000) {
            return nullptr;  // small (intra-mesh per-stage) solve: not worth N threads; use the normal path
        }
        // Permanent host cap, mode-3 (HardCapOnly) semantics: all-or-nothing tightening + shared unoccupancy
        // counter + "at most hard_cap_k occupied" as a unit clause. Permanent (not assumed) so it binds every round.
        if (hard_cap_k > 0) {
            std::vector<int> occ;
            std::vector<std::vector<int>> upg;
            topology_sat_build_group_occupancy(
                *solver, constraint_data, st->encs[static_cast<size_t>(k)], /*all_or_nothing=*/false, occ, &upg);
            if (hard_cap_k < occ.size()) {
                topology_sat_add_all_or_nothing_tightening(*solver, occ, upg);
                std::vector<int> neg;
                neg.reserve(occ.size());
                for (int o : occ) {
                    neg.push_back(-o);
                }
                std::vector<int> geq_unoccupied;
                topology_sat_add_at_least_k_counter(*solver, neg, occ.size() - 1, /*force=*/false, &geq_unoccupied);
                const size_t need = occ.size() - hard_cap_k;
                const int bound =
                    (need >= 1 && need - 1 < geq_unoccupied.size()) ? geq_unoccupied[need - 1] : 0;
                if (bound != 0) {
                    solver->add(bound);
                    solver->add(0);
                    solver->note_permanent_cap(hard_cap_k);
                }
            }
        }
        // Connect export AFTER encode so only learned (not input) clauses are published.
        solver->enable_clause_export(&st->pool, k, st->share_max_size);
        st->solvers.push_back(std::move(solver));
    }
    // Always-on path marker (survives quiet_mode) so experiments can PROVE the sharing-portfolio path ran.
    log_info(
        tt::LogFabric,
        "[topo-sat] SHARE-PORTFOLIO enumeration: {} workers (budget={}, max_size={}, hard_cap_k={}, seed_base={}, "
        "{} vars/worker)",
        st->n_workers,
        st->share_budget,
        st->share_max_size,
        hard_cap_k,
        base_seed,
        st->solvers.empty() ? 0 : st->solvers[0]->num_variables());
    return st;
}

// One cooperative round: race the persistent workers for the next model. Returns kSat (mapping_out filled from the
// winner), kUnsat (genuinely exhausted; sticky via st.exhausted) or 0 (no verdict -- should not normally happen).
inline int topology_sat_share_enum_solve(
    TopologySatShareEnumState& st, std::vector<int>& mapping_out, bool quiet_mode) {
    (void)quiet_mode;
    if (st.exhausted || st.solvers.empty()) {
        return TopologySatSolver::kUnsat;
    }
    st.round_done.store(false, std::memory_order_relaxed);
    std::mutex mtx;
    int winner = -1;
    bool unsat = false;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> workers;
    workers.reserve(st.solvers.size());
    for (int k = 0; k < st.n_workers; ++k) {
        workers.emplace_back([&, k]() {
            TopologySatSolver& S = *st.solvers[static_cast<size_t>(k)];
            S.set_cancel_flag(&st.round_done);
            std::vector<std::vector<int>> imported;
            for (;;) {
                if (st.round_done.load(std::memory_order_relaxed)) {
                    return;
                }
                const int status = S.solve_limited(st.share_budget);
                if (status == TopologySatSolver::kSat) {
                    std::lock_guard<std::mutex> lk(mtx);
                    if (!st.round_done.exchange(true)) {
                        winner = k;
                    }
                    return;
                }
                if (status == TopologySatSolver::kUnsat) {
                    // Genuine proof (cancellation returns 0, never kUnsat) -> the whole enumeration is exhausted.
                    std::lock_guard<std::mutex> lk(mtx);
                    if (!st.round_done.exchange(true)) {
                        unsat = true;
                    }
                    return;
                }
                // status == 0: window exhausted (or cancelled) -> import peers' clauses and continue.
                imported.clear();
                st.pool.drain(k, st.cursors[static_cast<size_t>(k)], imported);
                for (const auto& cl : imported) {
                    for (const int lit : cl) {
                        S.add(lit);
                    }
                    S.add(0);
                }
                st.imported_total.fetch_add(static_cast<long>(imported.size()), std::memory_order_relaxed);
            }
        });
    }
    for (auto& w : workers) {
        w.join();
    }
    ++st.rounds;
    if (winner >= 0) {
        if (!topology_sat_decode_hard_solution(
                *st.solvers[static_cast<size_t>(winner)], st.encs[static_cast<size_t>(winner)], mapping_out)) {
            return 0;
        }
        log_info(
            tt::LogFabric,
            "[topo-sat] share-enum round {}: winner=worker {} in {:.1f} ms (pool={}, imports={})",
            st.rounds,
            winner,
            topology_sat_elapsed_ms(t0),
            st.pool.size(),
            st.imported_total.load(std::memory_order_relaxed));
        return TopologySatSolver::kSat;
    }
    if (unsat) {
        st.exhausted = true;
        log_info(
            tt::LogFabric,
            "[topo-sat] share-enum round {}: UNSAT in {:.1f} ms -> enumeration exhausted",
            st.rounds,
            topology_sat_elapsed_ms(t0));
        return TopologySatSolver::kUnsat;
    }
    return 0;
}

// Block `mapping` in EVERY worker (before the next round -- the composition invariant for sound sharing + global
// dedup). Deterministic encode => the clause is literally identical across workers.
inline bool topology_sat_share_enum_block(
    TopologySatShareEnumState& st, const std::vector<int>& mapping, bool unique_shapes) {
    bool ok = true;
    for (size_t k = 0; k < st.solvers.size(); ++k) {
        ok = topology_sat_add_blocking_clause_for_mapping(*st.solvers[k], st.encs[k], mapping, unique_shapes) && ok;
    }
    return ok;
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
        // GOAL-1 / Design B: parallel seed portfolio (TT_TOPO_SAT_PORTFOLIO=N > 1). Race N seed workers (each its
        // own solver), take the first to SAT. Returns that mapping directly (bypasses the single-solver path).
        const long portfolio_n = topology_sat_env_long("TT_TOPO_SAT_PORTFOLIO", 0);
        if (portfolio_n > 1) {
            const long base_seed = topology_sat_env_long("TT_TOPO_SAT_SEED", 0);
            const bool fastsat = topology_sat_env_long("TT_TOPO_SAT_FASTSAT", 0) != 0;
            // TT_TOPO_SAT_SHARE=1 -> cooperative clause-sharing portfolio instead of the independent race.
            const bool share = topology_sat_env_long("TT_TOPO_SAT_SHARE", 0) != 0;
            const int share_budget = static_cast<int>(topology_sat_env_long("TT_TOPO_SAT_SHARE_BUDGET", 2000));
            const int share_max_size = static_cast<int>(topology_sat_env_long("TT_TOPO_SAT_SHARE_MAX_SIZE", 8));
            const auto t_pf = std::chrono::steady_clock::now();
            std::vector<int> m;
            const bool ok =
                share ? topology_sat_run_sharing_portfolio(
                            graph_data, constraint_data, validation_mode, static_cast<int>(portfolio_n), base_seed,
                            fastsat, share_budget, share_max_size, quiet_mode, m)
                      : topology_sat_run_portfolio(
                            graph_data, constraint_data, validation_mode, static_cast<int>(portfolio_n), base_seed,
                            fastsat, quiet_mode, m);
            if (ok) {
                state.mapping = std::move(m);
                std::fill(state.used.begin(), state.used.end(), false);
                for (size_t t = 0; t < state.mapping.size(); ++t) {
                    const int gi = state.mapping[t];
                    if (gi >= 0 && static_cast<size_t>(gi) < state.used.size()) {
                        state.used[static_cast<size_t>(gi)] = true;
                    }
                }
                if (!quiet_mode) {
                    log_debug(
                        tt::LogFabric,
                        "[topo-sat-profile] portfolio.total : {:.1f} ms ({}-way)",
                        topology_sat_elapsed_ms(t_pf),
                        portfolio_n);
                }
                return true;
            }
            // portfolio produced no model -> fall through to the normal single-solver path
        }
        // GOAL-1 base-embedding speedups (each env-gated, default off; independently toggleable):
        //   TT_TOPO_SAT_SEED=N     -> CaDiCaL random seed (enables a seed portfolio; SAT time is seed-sensitive)
        //   TT_TOPO_SAT_FASTSAT=1  -> bias CaDiCaL toward finding a model fast
        // Set before encode() so they apply in CONFIGURING state. Rejected options are no-ops.
        const long g1_seed = topology_sat_env_long("TT_TOPO_SAT_SEED", -1);
        if (g1_seed >= 0) {
            (void)solver.set_option("seed", static_cast<int>(g1_seed));
        }
        if (topology_sat_env_long("TT_TOPO_SAT_FASTSAT", 0) != 0) {
            (void)solver.set_option("target", 2);
            (void)solver.set_option("phase", 1);
        }
        if (!topology_sat_encode_hard_constraints(
                solver, graph_data, constraint_data, enc, validation_mode, quiet_mode)) {
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
        // EXPERIMENT hook: dump the encoded CNF to DIMACS (TT_TOPO_SAT_DUMP_DIMACS=path) to feed an external
        // parallel / clause-sharing solver (gimsatul, plingeling, ...). Gated on size so it captures the big
        // inter-mesh solve, not the tiny intra-mesh per-stage ones. Dumps then stops (returns false).
        // Only for the BASE-embedding experiment (NO_MINHOST): otherwise a min-host run's hardcap dump (in
        // solve_minimize_groups) would be clobbered by this base-solve fallback writing the same path.
        if (const char* dp = std::getenv("TT_TOPO_SAT_DUMP_DIMACS");
            dp != nullptr && dp[0] != '\0' && solver.num_variables() > 5000 &&
            std::getenv("TT_TOPO_SAT_NO_MINHOST") != nullptr) {
            const bool ok = solver.write_dimacs(dp);
            log_info(
                tt::LogFabric,
                "[topo-sat-profile] DUMP_DIMACS: wrote CNF to {} ({}), {} vars, {} clauses",
                dp,
                ok ? "ok" : "FAILED",
                solver.num_variables(),
                solver.num_clauses());
            state.error_message = "TT_TOPO_SAT_DUMP_DIMACS: dumped CNF, skipping solve";
            return false;
        }
        // TT_TOPO_SAT_BASE_WARMHINT=1: phase-hint a greedy adjacency-walk embedding before the solve (Goal 1).
        if (topology_sat_env_long("TT_TOPO_SAT_BASE_WARMHINT", 0) != 0) {
            const auto t_wh = std::chrono::steady_clock::now();
            const int h = topology_sat_apply_base_warmhint(solver, graph_data, enc);
            if (!quiet_mode) {
                log_debug(
                    tt::LogFabric,
                    "[topo-sat-profile] base.warmhint : {:.1f} ms ({} phase hints)",
                    topology_sat_elapsed_ms(t_wh),
                    h);
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
    };

    // EXPERIMENT single-solve minimal-host path (TT_TOPO_SAT_MIN_MODE / TT_TOPO_SAT_SHARE+PORTFOLIO): the
    // occupancy-based solve from the pre-rebase branch (warm descent + full-packing hard-cap lock, or the
    // clause-sharing portfolio round), with the capacity lower bound k_min as the hard cap (the old mapper set
    // MappingConstraints::max_same_rank_groups_used to exactly this value; the rebased base removed that API).
    // Active only when an experiment knob is set; on failure it falls through to the base's budget walk below.
    if (topology_sat_experiment_minhost_enabled(constraint_data)) {
        const int kGroupDescentConflictBudget =
            static_cast<int>(topology_sat_env_long("TT_TOPO_SAT_DESCENT_BUDGET", 20'000));
        const int kGroupLockConflictBudget = static_cast<int>(topology_sat_env_long("TT_TOPO_SAT_LOCK_BUDGET", 0));
        const bool profile = !quiet_mode;
        size_t exp_max_cap = 0;
        for (const auto& g : constraint_data.same_rank_groups) {
            exp_max_cap = std::max(exp_max_cap, g.size());
        }
        const size_t k_floor = (exp_max_cap > 0) ? (graph_data.n_target + exp_max_cap - 1) / exp_max_cap : 1;
        const size_t hard_cap_k = topology_sat_experiment_hard_cap_k(graph_data, constraint_data);
        TopologySatSolver solver;
        TopologySatHardEncoding enc;
        auto t_min_enc = std::chrono::steady_clock::now();
        const bool min_enc_ok =
            topology_sat_encode_hard_constraints(solver, graph_data, constraint_data, enc, validation_mode, quiet_mode);
        if (profile) {
            log_debug(
                tt::LogFabric,
                "[topo-sat-profile] minimize: encode_hard_constraints total : {:.1f} ms (ok={})",
                topology_sat_elapsed_ms(t_min_enc),
                min_enc_ok);
        }
        // Mode D SINGLE solve (SHARE=1 + PORTFOLIO=N with a hard host cap): route the capped min-host solve through
        // one cooperative share-enum round so the portfolio composes with the occupancy objective too.
        if (min_enc_ok && topology_sat_share_enum_enabled() && hard_cap_k > 0 && solver.num_variables() > 5000) {
            auto se = topology_sat_share_enum_init(graph_data, constraint_data, validation_mode, hard_cap_k, quiet_mode);
            if (se) {
                std::vector<int> m;
                if (topology_sat_share_enum_solve(*se, m, quiet_mode) == TopologySatSolver::kSat && !m.empty()) {
                    state.mapping = std::move(m);
                    std::fill(state.used.begin(), state.used.end(), false);
                    for (int gi : state.mapping) {
                        if (gi >= 0 && static_cast<size_t>(gi) < state.used.size()) {
                            state.used[static_cast<size_t>(gi)] = true;
                        }
                    }
                    log_info(
                        tt::LogFabric,
                        "Topology SAT: hard-capped host-group usage at {} group(s) (via clause-sharing portfolio)",
                        hard_cap_k);
                    return true;
                }
                // No model from the portfolio round -> fall through to the normal minimize path below.
            }
        }
        if (min_enc_ok) {
            auto t_min = std::chrono::steady_clock::now();
            std::vector<int> best_mapping;
            size_t best_k = 0;
            bool hard_cap_met = false;
            const bool min_ok = topology_sat_solve_minimize_groups(
                graph_data,
                solver,
                enc,
                constraint_data,
                kGroupDescentConflictBudget,
                k_floor,
                best_mapping,
                best_k,
                hard_cap_k,
                kGroupLockConflictBudget,  // strict: unbounded by default (run lock until k_min proven / UNSAT)
                &hard_cap_met,
                /*make_cap_permanent=*/false,
                quiet_mode);
            if (profile) {
                log_debug(
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
        // Experiment objective produced no model -> fall through to the base's budget walk below.
    }

    // Opt-in objective: minimize the number of distinct same-rank global groups (host partitions) the mapping
    // touches. Walk a host-usage budget upward from the capacity-based lower bound (ceil(n_target / max group
    // capacity)) and return the first budget that is satisfiable — that is the minimum number of hosts. This is a
    // complete (not greedy) search, so it finds the true minimum host count when one exists. It is best-effort:
    // if no budget below the total group count is satisfiable we fall through to the normal unconstrained solve,
    // so enabling the objective can never turn a solvable instance UNSAT.
    if (constraint_data.minimize_same_rank_groups_used) {
        size_t num_host_groups = 0;
        size_t max_group_capacity = 0;
        for (const auto& grp : constraint_data.same_rank_groups) {
            if (!grp.empty()) {
                ++num_host_groups;
                max_group_capacity = std::max(max_group_capacity, grp.size());
            }
        }
        if (num_host_groups >= 2 && max_group_capacity > 0) {
            const size_t k_min = (graph_data.n_target + max_group_capacity - 1) / max_group_capacity;
            // Each tight host-budget solve is conflict-capped. Proving the minimum host count for a ring/chain
            // embedded into a strictly larger physical graph (e.g. a 64-mesh decode ring on an 80-mesh / 20-host
            // supercluster) is a Hamiltonian-cycle-with-cardinality search the SAT engine can spin on for minutes;
            // the cap lets an intractable budget be abandoned so the loop (and then the unconstrained fall-through
            // below) still returns a valid mapping quickly. Tractable budgets finish well within the cap and return
            // the identical model they would unbounded, so existing golden mappings are unchanged.
            static constexpr int kHostMinimizeConflictBudget = 300'000;
            for (size_t k = std::max<size_t>(k_min, 1); k < num_host_groups; ++k) {
                TopologySatSolver solver;
                TopologySatHardEncoding enc;
                if (!topology_sat_encode_hard_constraints(solver, graph_data, constraint_data, enc, validation_mode)) {
                    break;  // hard constraints alone are UNSAT; defer to the normal path for error messaging
                }
                if (!topology_sat_encode_host_group_budget(solver, constraint_data, enc, k)) {
                    continue;  // this budget is trivially unencodable; try a larger one
                }
                if (solver.solve_limited(kHostMinimizeConflictBudget) == TopologySatSolver::kSat &&
                    finalize_success(solver, enc)) {
                    if (!quiet_mode) {
                        log_info(
                            tt::LogFabric,
                            "Topology SAT: minimized host-group usage to {} group(s) (capacity lower bound {})",
                            k,
                            k_min);
                    }
                    return true;
                }
            }
            // No binding budget was satisfiable within the conflict cap; fall through to the unconstrained solve.
        }
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
    if (!topology_sat_encode_hard_constraints(
            solver, graph_data, constraint_data, enc, validation_mode, quiet_mode)) {
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

    // Preferred minimization: the lower-bound search (exact DFS / greedy) can dominate on preferred-heavy inputs.
    const auto t_pref_lb = std::chrono::steady_clock::now();
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
    if (!quiet_mode) {
        log_debug(
            tt::LogFabric,
            "[topo-sat-profile] preferred.lower_bound : {:.1f} ms ({} preferred-hit literals)",
            topology_sat_elapsed_ms(t_pref_lb),
            pref_hit_literals.size());
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
    const auto t_pref_solve = std::chrono::steady_clock::now();
    const int status = solve_with_symmetry_break(solver, enc);
    if (!quiet_mode) {
        log_debug(
            tt::LogFabric,
            "[topo-sat-profile] preferred.solve : {:.1f} ms (status={})",
            topology_sat_elapsed_ms(t_pref_solve),
            status);
    }
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

    // EXPERIMENT (Story 3): seed applied to BOTH enumeration modes (incremental + from-scratch) so a seed-variance
    // sweep is fair. Must be set pre-encode (CONFIGURING state). -1 (default) leaves CaDiCaL's default seed.
    const long enum_seed = topology_sat_env_long("TT_TOPO_SAT_SEED", -1);

    // One CaDiCaL::Solver for the whole enumeration: encode once, then add blocking clauses and solve() in a loop.
    // (No full re-encode / new solver per model.)
    TopologySatSolver solver;
    solver.configure_for_blocking_clause_enumeration();
    if (enum_seed >= 0) {
        (void)solver.set_option("seed", static_cast<int>(enum_seed));
    }
    TopologySatHardEncoding enc;
    // Top-level phase attribution for the enumeration path (encode / minimal-host prime / enumerate loop). The scoped
    // timer emits search_n.total on every return; the manual subtotals below split it. DEBUG only, quiet_mode silent.
    TopologySatScopedTimer search_n_total_timer("search_n.total", quiet_mode);
    const auto t_encode = std::chrono::steady_clock::now();
    if (!topology_sat_encode_hard_constraints(
            solver, graph_data, constraint_data, enc, validation_mode, quiet_mode)) {
        return false;
    }
    if (!quiet_mode) {
        log_debug(
            tt::LogFabric,
            "[topo-sat-profile] search_n.encode : {:.1f} ms (CNF {} clauses, {} vars)",
            topology_sat_elapsed_ms(t_encode),
            solver.num_clauses(),
            solver.num_variables());
    }

    for (const auto& shape_key : initial_forbidden_shape_keys) {
        std::vector<int> forbid_clause;
        topology_sat_build_shape_blocking_clause(enc, shape_key, forbid_clause);
        topology_sat_add_shape_clause_or_unsat(solver, enc, forbid_clause);
    }

    // EXPERIMENT (Story 3, TT_TOPO_SAT_ENUM_FROMSCRATCH=1): from-scratch enumeration -- the control for measuring the
    // benefit of incremental state reuse. Instead of ONE persistent solver reused across solutions (the loop below),
    // rebuild a FRESH solver + re-encode + replay ALL prior blocking clauses each solution, then do a mode-3 hardcap
    // solve. No learned-clause / phase / VSIDS carryover. Same result set, different cost. Compares vs the incremental
    // path (default). Uses the same solve_minimize_groups(mode 3) per step -- run with TT_TOPO_SAT_MIN_MODE=3.
    if (topology_sat_env_long("TT_TOPO_SAT_ENUM_FROMSCRATCH", 0) != 0) {
        // Always-on confirmation marker (survives quiet_mode) so experiments can PROVE this path ran.
        log_info(
            tt::LogFabric,
            "[topo-sat] ENUM path = FROM-SCRATCH (fresh solver + replay blocks per solution), seed={}",
            enum_seed);
        size_t fs_max_cap = 0;
        for (const auto& g : constraint_data.same_rank_groups) {
            fs_max_cap = std::max(fs_max_cap, g.size());
        }
        const size_t fs_k_floor = (fs_max_cap > 0) ? (graph_data.n_target + fs_max_cap - 1) / fs_max_cap : 1;
        const size_t fs_hard_cap_k = topology_sat_experiment_hard_cap_k(graph_data, constraint_data);
        const int fs_descent_budget = static_cast<int>(topology_sat_env_long("TT_TOPO_SAT_DESCENT_BUDGET", 20'000));
        const int fs_lock_budget = static_cast<int>(topology_sat_env_long("TT_TOPO_SAT_LOCK_BUDGET", 0));
        const auto t_fs = std::chrono::steady_clock::now();
        while (all_mappings_out.size() < max_solutions) {
            TopologySatSolver fs_solver;
            if (enum_seed >= 0) {
                (void)fs_solver.set_option("seed", static_cast<int>(enum_seed));
            }
            TopologySatHardEncoding fs_enc;
            if (!topology_sat_encode_hard_constraints(
                    fs_solver, graph_data, constraint_data, fs_enc, validation_mode, quiet_mode)) {
                break;
            }
            // Replay the initial forbidden shapes + every solution already found (deterministic encode => same var
            // numbering => blocking clauses reconstruct exactly).
            for (const auto& shape_key : initial_forbidden_shape_keys) {
                std::vector<int> fc;
                topology_sat_build_shape_blocking_clause(fs_enc, shape_key, fc);
                topology_sat_add_shape_clause_or_unsat(fs_solver, fs_enc, fc);
            }
            for (const auto& m : all_mappings_out) {
                topology_sat_add_blocking_clause_for_mapping(fs_solver, fs_enc, m, unique_shapes);
            }
            std::vector<int> fs_mapping;
            size_t fs_best_k = 0;
            bool fs_met = false;
            const bool ok = topology_sat_solve_minimize_groups(
                graph_data, fs_solver, fs_enc, constraint_data, fs_descent_budget, fs_k_floor, fs_mapping, fs_best_k,
                fs_hard_cap_k, fs_lock_budget, &fs_met, /*make_cap_permanent=*/false, quiet_mode);
            if (!ok || fs_mapping.empty()) {
                break;
            }
            all_mappings_out.push_back(std::move(fs_mapping));
            if (!quiet_mode) {
                log_info(
                    tt::LogFabric,
                    "topology_sat_search_n[fromscratch]: found {} solution(s) so far (max={}), {:.1f} ms elapsed",
                    all_mappings_out.size(),
                    max_solutions,
                    topology_sat_elapsed_ms(t_fs));
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

    // Clause-sharing portfolio enumeration (TT_TOPO_SAT_SHARE=1 + PORTFOLIO=N): run the whole multi-solution
    // enumeration through the N-worker cooperative portfolio (see TopologySatShareEnumState). Requires the hard host
    // cap (mode-3 semantics baked permanently into every worker). Falls through to the incremental path when the
    // formula is small (intra-mesh) or the init fails.
    const size_t share_cap_k = topology_sat_experiment_hard_cap_k(graph_data, constraint_data);
    if (topology_sat_share_enum_enabled() && share_cap_k > 0) {
        auto se = topology_sat_share_enum_init(graph_data, constraint_data, validation_mode, share_cap_k, quiet_mode);
        if (se) {
            log_info(
                tt::LogFabric,
                "[topo-sat] ENUM path = SHARE-PORTFOLIO ({} persistent workers, blocking clauses fanned out per "
                "round), seed={}",
                se->n_workers,
                enum_seed);
            for (const auto& shape_key : initial_forbidden_shape_keys) {
                for (size_t k = 0; k < se->solvers.size(); ++k) {
                    std::vector<int> fc;
                    topology_sat_build_shape_blocking_clause(se->encs[k], shape_key, fc);
                    topology_sat_add_shape_clause_or_unsat(*se->solvers[k], se->encs[k], fc);
                }
            }
            while (all_mappings_out.size() < max_solutions) {
                std::vector<int> m;
                if (topology_sat_share_enum_solve(*se, m, quiet_mode) != TopologySatSolver::kSat || m.empty()) {
                    break;
                }
                all_mappings_out.push_back(std::move(m));
                if (!quiet_mode) {
                    log_info(
                        tt::LogFabric,
                        "topology_sat_search_n[share-portfolio]: found {} solution(s) so far (max={})",
                        all_mappings_out.size(),
                        max_solutions);
                }
                if (!topology_sat_share_enum_block(*se, all_mappings_out.back(), unique_shapes)) {
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
    }

    // Drive the solve heartbeat's "progress to first solution" / enumeration counters (see TopologySatSolver).
    solver.set_solution_progress(0, static_cast<std::int64_t>(max_solutions));
    solver.set_progress_phase("enumerate");
    // Always-on confirmation marker (survives quiet_mode) so experiments can PROVE the incremental path ran.
    log_info(
        tt::LogFabric,
        "[topo-sat] ENUM path = INCREMENTAL (one solver reused + blocking clauses), seed={}",
        enum_seed);

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
    if (topology_sat_experiment_minhost_enabled(constraint_data)) {
        const int kDescentBudget = static_cast<int>(topology_sat_env_long("TT_TOPO_SAT_DESCENT_BUDGET", 20'000));
        const int kLockBudget = static_cast<int>(topology_sat_env_long("TT_TOPO_SAT_LOCK_BUDGET", 0));
        size_t max_cap = 0;
        for (const auto& g : constraint_data.same_rank_groups) {
            max_cap = std::max(max_cap, g.size());
        }
        const size_t k_floor = (max_cap > 0) ? (graph_data.n_target + max_cap - 1) / max_cap : 1;
        const size_t hard_cap_k = topology_sat_experiment_hard_cap_k(graph_data, constraint_data);
        std::vector<int> first_mapping;
        size_t best_k = 0;
        bool hard_cap_met = false;
        solver.set_progress_phase("descent");  // minimal-host warm descent -- the long pre-first-solution phase
        const auto t_prime = std::chrono::steady_clock::now();
        const bool primed = topology_sat_solve_minimize_groups(
            graph_data,
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
            /*make_cap_permanent=*/true,
            quiet_mode);
        if (!quiet_mode) {
            log_debug(
                tt::LogFabric,
                "[topo-sat-profile] search_n.prime_minimize : {:.1f} ms (primed={}, best_k={})",
                topology_sat_elapsed_ms(t_prime),
                primed,
                best_k);
        }
        if (primed && !first_mapping.empty()) {
            all_mappings_out.push_back(first_mapping);
            solver.set_progress_phase("enumerate");
            solver.set_solution_progress(
                static_cast<std::int64_t>(all_mappings_out.size()), static_cast<std::int64_t>(max_solutions));
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

    // GIM_EVERY (TT_TOPO_SAT_GIMSATUL=1 + TT_TOPO_SAT_GIM_EVERY=1): delegate EVERY enumeration step to a fresh
    // gimsatul run on the full clause tape (base CNF + permanent cap + all accumulated blocking clauses, + distilled
    // pool clauses when TT_TOPO_SAT_POOL=1). Model parse-back/decoding stay in this driver. gimsatul returning 0
    // (no binary / parse fail) falls back to the native solve.
    const bool gim_every = topology_sat_env_long("TT_TOPO_SAT_GIMSATUL", 0) != 0 &&
                           topology_sat_env_long("TT_TOPO_SAT_GIM_EVERY", 0) != 0 &&
                           solver.num_variables() > 5000;  // big-solve gate (see delegated_solve)
    const int gim_threads = static_cast<int>(topology_sat_env_long("TT_TOPO_SAT_GIMSATUL_THREADS", 32));
    if (gim_every) {
        log_info(tt::LogFabric, "[topo-sat] ENUM steps = GIM_EVERY (fresh gimsatul run per solution)");
    }

    const auto t_enum_loop = enum_clock::now();
    const size_t enum_start_count = all_mappings_out.size();
    while (all_mappings_out.size() < max_solutions) {
        // Default (kEnumLoopConflictBudget==0): UNBOUNDED solve -- never give up on a budget, so we only stop on a
        // real kUnsat (genuine exhaustion), never on a kUnknown that would silently truncate. If a budget is set,
        // fall back to solve_limited (best-effort; kUnknown then stops the loop).
        int status = 0;
        if (gim_every) {
            status = solver.gimsatul_solve(gim_threads, {});
        }
        if (status == 0) {
            status = (kEnumLoopConflictBudget > 0) ? solver.solve_limited(kEnumLoopConflictBudget) : solver.solve();
        }
        if (status != TopologySatSolver::kSat) {
            break;
        }

        std::vector<int> current_mapping;
        if (!topology_sat_decode_hard_solution(solver, enc, current_mapping)) {
            break;
        }
        all_mappings_out.push_back(std::move(current_mapping));

        if (!quiet_mode) {
            const auto now = enum_clock::now();
            const bool reached_cap = all_mappings_out.size() >= max_solutions;
            if (reached_cap || now - last_enum_progress_log >= kEnumProgressLogInterval) {
                log_info(
                    tt::LogFabric,
                    "topology_sat_search_n: found {} / {} solution(s) so far",
                    all_mappings_out.size(),
                    max_solutions);
                last_enum_progress_log = now;
            }
        }

        if (!topology_sat_add_blocking_clause_for_mapping(solver, enc, all_mappings_out.back(), unique_shapes)) {
            break;
        }
    }

    if (!quiet_mode) {
        log_debug(
            tt::LogFabric,
            "[topo-sat-profile] search_n.enumerate_loop : {:.1f} ms ({} via blocking-clause loop, {} total)",
            topology_sat_elapsed_ms(t_enum_loop),
            all_mappings_out.size() - enum_start_count,
            all_mappings_out.size());
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
    ConnectionValidationMode validation_mode,
    bool quiet_mode) {
    auto session = std::unique_ptr<TopologySatSession, TopologySatSessionDeleter>(new TopologySatSession{});
    session->solver.configure_for_blocking_clause_enumeration();
    enc = {};
    if (!topology_sat_encode_hard_constraints(
            session->solver, graph_data, constraint_data, enc, validation_mode, quiet_mode)) {
        return nullptr;
    }
    // Same minimal-host strategy as topology_sat_search_n: PRIME the solver with the warm descent + full-packing
    // lock, make the achieved cap PERMANENT, and stash the primed model so every incremental (.next) solution
    // occupies the minimal host count. See TOPOLOGY_OCCUPANCY_SOLVE_README §7.
    if (topology_sat_experiment_minhost_enabled(constraint_data)) {
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
        const size_t hard_cap_k = topology_sat_experiment_hard_cap_k(graph_data, constraint_data);
        std::vector<int> first_mapping;
        size_t best_k = 0;
        bool hard_cap_met = false;
        // Clause-sharing portfolio enumeration (TT_TOPO_SAT_SHARE=1 + PORTFOLIO=N): the session enumerates through
        // N persistent cooperating workers instead of the single incremental solver. The prime is round 1.
        if (topology_sat_share_enum_enabled() && hard_cap_k > 0) {
            session->share_enum =
                topology_sat_share_enum_init(graph_data, constraint_data, validation_mode, hard_cap_k, quiet_mode);
            if (session->share_enum) {
                if (topology_sat_share_enum_solve(*session->share_enum, first_mapping, quiet_mode) ==
                        TopologySatSolver::kSat &&
                    !first_mapping.empty()) {
                    session->primed_first_mapping = std::move(first_mapping);
                    session->has_primed_mapping = true;
                    log_info(
                        tt::LogFabric,
                        "Topology SAT enumeration session: primed SHARE-PORTFOLIO minimal-host enumeration "
                        "(hard_cap_k={})",
                        hard_cap_k);
                }
                // Prime failed => share_enum is exhausted/stuck; solve_and_decode will report no solutions. Either
                // way the session stays on the portfolio path (mixing paths would lose the workers' blocking state).
                return session;
            }
            // init returned nullptr (small formula / encode failure) -> normal single-solver path below.
        }
        const bool primed = topology_sat_solve_minimize_groups(
            graph_data,
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
            /*make_cap_permanent=*/true,
            quiet_mode);
        if (primed && !first_mapping.empty()) {
            session->primed_first_mapping = std::move(first_mapping);
            session->has_primed_mapping = true;
            if (!quiet_mode) {
                log_info(
                    tt::LogFabric,
                    "Topology SAT enumeration session: primed minimal-host enumeration at {} occupied host group(s) "
                    "(hard_cap_k={}, met={})",
                    best_k,
                    hard_cap_k,
                    hard_cap_met);
            }
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
    // SHARE-PORTFOLIO enumeration: fan the blocking clause out to EVERY persistent worker before the next round
    // (the composition invariant -- all workers always agree on the found-solution set; dedup stays global).
    if (session->share_enum) {
        return topology_sat_share_enum_block(*session->share_enum, raw_mapping, unique_shapes);
    }
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
    // SHARE-PORTFOLIO enumeration: each further solution is one cooperative round over the persistent workers.
    if (session->share_enum) {
        std::vector<int> m;
        if (topology_sat_share_enum_solve(*session->share_enum, m, /*quiet_mode=*/false) !=
                TopologySatSolver::kSat ||
            m.empty()) {
            return false;
        }
        raw_out = std::move(m);
        return true;
    }
    // GIM_EVERY (TT_TOPO_SAT_GIMSATUL=1 + TT_TOPO_SAT_GIM_EVERY=1): delegate EVERY .next step to a fresh gimsatul
    // run over the full clause tape (base CNF + permanent host cap + all accumulated blocking clauses, + distilled
    // pool clauses when TT_TOPO_SAT_POOL=1). Decode stays here; gimsatul 0 (no binary) falls back to native.
    if (topology_sat_env_long("TT_TOPO_SAT_GIMSATUL", 0) != 0 &&
        topology_sat_env_long("TT_TOPO_SAT_GIM_EVERY", 0) != 0 &&
        session->solver.num_variables() > 5000) {  // big-solve gate: intra-mesh sessions stay native
        const int gim_threads = static_cast<int>(topology_sat_env_long("TT_TOPO_SAT_GIMSATUL_THREADS", 32));
        const int gim_status = session->solver.gimsatul_solve(gim_threads, {});
        if (gim_status == TopologySatSolver::kSat) {
            return topology_sat_decode_hard_solution(session->solver, enc, raw_out);
        }
        if (gim_status == TopologySatSolver::kUnsat) {
            return false;
        }
        // 0 -> fall through to the native solve below.
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
