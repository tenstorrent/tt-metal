// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
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
struct TopologySatSession {
    TopologySatSolver solver;
};

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
// c[i][j] represents "at least j+1 of lits[0..i] are true"; assert c[m-1][k-1].
inline void topology_sat_add_at_least_k_counter(TopologySatSolver& solver, const std::vector<int>& lits, size_t k) {
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

    using enum_clock = std::chrono::steady_clock;
    constexpr auto kEnumProgressLogInterval = std::chrono::seconds(5);
    // Eligible for an immediate first progress line, then at most once per kEnumProgressLogInterval.
    auto last_enum_progress_log = enum_clock::now() - kEnumProgressLogInterval;

    while (all_mappings_out.size() < max_solutions) {
        const int status = solver.solve();
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
    if (session->solver.solve() != TopologySatSolver::kSat) {
        return false;
    }
    return topology_sat_decode_hard_solution(session->solver, enc, raw_out);
}

}  // namespace tt::tt_fabric::detail
