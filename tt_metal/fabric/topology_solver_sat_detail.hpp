// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// SAT / Kissat implementation: not part of the public header graph. Include only from TUs that run the SAT
// topology backend (fabric mapper / grouping sources and fabric unit tests). `topology_solver.hpp` / `.tpp` stay
// solver-agnostic; `control_plane.hpp` forward-declares TopologyMapper so llrt does not compile this file.

#pragma once

#ifndef TT_METALIUM_TOPOLOGY_SOLVER_SAT_DETAIL_HPP
#define TT_METALIUM_TOPOLOGY_SOLVER_SAT_DETAIL_HPP

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <set>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_set>
#include <vector>

extern "C" {
#include <kissat.h>
}

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include <tt-metalium/experimental/fabric/topology_solver.hpp>

namespace tt::tt_fabric::detail {

/**
 * Thin IPASIR-style wrapper around Kissat (positive DIMACS literals; 0 ends a clause).
 * Matches the IPASIR-style API expected by topology_sat_* encodings.
 */
struct TopologySatSolver {
    kissat* k = nullptr;
    int next_var = 0;

    TopologySatSolver() : k(kissat_init()) { kissat_set_option(k, "quiet", 1); }
    ~TopologySatSolver() {
        if (k != nullptr) {
            kissat_release(k);
            k = nullptr;
        }
    }
    TopologySatSolver(const TopologySatSolver&) = delete;
    TopologySatSolver& operator=(const TopologySatSolver&) = delete;
    TopologySatSolver(TopologySatSolver&& o) noexcept : k(o.k), next_var(o.next_var) {
        o.k = nullptr;
        o.next_var = 0;
    }
    TopologySatSolver& operator=(TopologySatSolver&& o) noexcept {
        if (this != &o) {
            if (k != nullptr) {
                kissat_release(k);
            }
            k = o.k;
            next_var = o.next_var;
            o.k = nullptr;
            o.next_var = 0;
        }
        return *this;
    }

    int declare_one_more_variable() {
        ++next_var;
        kissat_reserve(k, next_var);
        return next_var;
    }

    void add(int lit) { kissat_add(k, lit); }

    int solve() { return kissat_solve(k); }

    static constexpr int kSat = 10;
    static constexpr int kUnsat = 20;

    // For DIMACS literal `lit`, returns `lit` if satisfied, `-lit` if falsified, else 0 (unknown).
    int val(int lit) const {
        const int a = std::abs(lit);
        const int r = kissat_value(k, a);
        if (r == 0) {
            return 0;
        }
        if (lit > 0) {
            return (r > 0) ? lit : -lit;
        }
        return (r < 0) ? lit : -lit;
    }
};

struct TopologySatHardEncoding {
    bool trivial_unsat = false;
    std::string trivial_reason;
    std::vector<std::vector<size_t>> allowed_global_idx;
    std::vector<std::vector<int>> assign_lit;
};

namespace {

template <typename TargetNode, typename GlobalNode>
bool global_adjacent_idx(const GraphIndexData<TargetNode, GlobalNode>& graph_data, size_t global_i, size_t global_j) {
    if (global_i >= graph_data.n_global || global_j >= graph_data.n_global) {
        return false;
    }
    const auto& adj = graph_data.global_adj_idx[global_i];
    return std::binary_search(adj.begin(), adj.end(), global_j);
}

template <typename TargetNode, typename GlobalNode>
bool topology_sat_edges_ok_if_assign(
    const GraphIndexData<TargetNode, GlobalNode>& graph_data,
    const std::vector<int>& mapping,
    size_t target_idx,
    size_t global_idx) {
    for (size_t tn : graph_data.target_adj_idx[target_idx]) {
        if (mapping[tn] < 0) {
            continue;
        }
        const size_t gg = static_cast<size_t>(mapping[tn]);
        if (!global_adjacent_idx(graph_data, global_idx, gg)) {
            return false;
        }
    }
    return true;
}

template <typename TargetNode, typename GlobalNode>
size_t topology_sat_remaining_preferred_upper_bound(
    const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
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
// becomes an exhaustive search → exact optimum for small instances; otherwise it stops after max_nodes expansions and
// returns the best complete mapping found (still a safe lower bound for at-least-k).
template <typename TargetNode, typename GlobalNode>
size_t topology_sat_max_preferred_lower_bound_with_encoding(
    const GraphIndexData<TargetNode, GlobalNode>& graph_data,
    const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
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
        if (pref_so_far + topology_sat_remaining_preferred_upper_bound(constraint_data, enc, used, ti, nt) <= best) {
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
            if (!topology_sat_edges_ok_if_assign(graph_data, mapping, ti, g)) {
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

template <typename TargetNode, typename GlobalNode>
size_t topology_sat_greedy_pref_lower_bound_with_encoding(
    const GraphIndexData<TargetNode, GlobalNode>& graph_data,
    const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
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
                if (!topology_sat_edges_ok_if_assign(graph_data, mapping, ti, cand.g)) {
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

}  // namespace

inline bool topology_sat_combinations_exceed_limit(size_t n, size_t r, size_t max_combinations) {
    if (r > n) {
        return true;
    }
    if (r == 0 || r == n) {
        return false;
    }
    if (r > n - r) {
        r = n - r;
    }
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
            emit_combination(cur);
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
// O(n²) pairwise binary clauses.  Sequential encoding improves unit-propagation on large domains.
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

// One auxiliary literal per target: true iff the chosen global for that target is in its preferred set (and the
// target has at least one feasible preferred assignment literal).
template <typename TargetNode, typename GlobalNode>
void topology_sat_append_preferred_hit_indicators(
    TopologySatSolver& solver,
    const TopologySatHardEncoding& enc,
    const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
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
        const int p = solver.declare_one_more_variable();
        solver.add(-p);
        for (int lit : row_pref_lits) {
            solver.add(lit);
        }
        solver.add(0);
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
template <typename TargetNode, typename GlobalNode>
size_t topology_sat_relaxed_channel_threshold_literal_count_upper_bound(
    const GraphIndexData<TargetNode, GlobalNode>& graph_data) {
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
template <typename TargetNode, typename GlobalNode>
bool topology_sat_append_relaxed_channel_threshold_literals(
    TopologySatSolver& solver,
    const TopologySatHardEncoding& enc,
    const GraphIndexData<TargetNode, GlobalNode>& graph_data,
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
                        if (!global_adjacent_idx(graph_data, glob1, glob2)) {
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

template <typename TargetNode, typename GlobalNode>
bool topology_sat_encode_hard_constraints(
    TopologySatSolver& solver,
    const GraphIndexData<TargetNode, GlobalNode>& graph_data,
    const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
    TopologySatHardEncoding& enc,
    ConnectionValidationMode validation_mode = ConnectionValidationMode::RELAXED) {
    enc = TopologySatHardEncoding{};
    const size_t nt = graph_data.n_target;
    const size_t ng = graph_data.n_global;

    if (nt == 0) {
        return true;
    }

    enc.allowed_global_idx.resize(nt);
    enc.assign_lit.resize(nt);

    // Initial domain: constraint + degree filtering.
    std::vector<std::vector<size_t>> domain(nt);
    for (size_t t = 0; t < nt; ++t) {
        for (size_t g = 0; g < ng; ++g) {
            if (!constraint_data.is_valid_mapping(t, g)) {
                continue;
            }
            if (graph_data.global_deg[g] < graph_data.target_deg[t]) {
                continue;
            }
            domain[t].push_back(g);
        }
        if (domain[t].empty()) {
            enc.trivial_unsat = true;
            enc.trivial_reason = fmt::format("topology_sat: no allowed global for target_idx {}", t);
            return false;
        }
    }

    // Arc consistency (AC-3): prune domains so every value has at least one compatible support
    // for each neighboring target.  Build membership sets for fast lookup.
    std::vector<std::unordered_set<size_t>> domain_set(nt);
    for (size_t t = 0; t < nt; ++t) {
        domain_set[t].insert(domain[t].begin(), domain[t].end());
    }

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
            if (domain_set[t_neigh].count(g2) == 0) {
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
                for (size_t t2 : graph_data.target_adj_idx[t]) {
                    if (t2 != tn) {
                        next_worklist.emplace_back(t2, t);
                    }
                }
            }
        }
        worklist = std::move(next_worklist);
    }

    // When there are preferred globals for a target, list them first in the assignment matrix order so unit
    // propagation / heuristic order favors them under a single solve (no MaxSAT on preferred-hit count).
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

    // Exactly one global choice per target.
    for (size_t t = 0; t < nt; ++t) {
        const auto& lits = enc.assign_lit[t];
        TT_ASSERT(lits.size() == enc.allowed_global_idx[t].size());
        for (int lit : lits) {
            solver.add(lit);
        }
        solver.add(0);
        topology_sat_add_at_most_one_sequential(solver, lits);
    }

    // Injective: each global node used by at most one target.
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

    // Adjacency preservation via support encoding: for each target edge (t1,t2), for each candidate global g_a
    // assigned to one endpoint, add a clause requiring the other endpoint to map to some adjacent (and, in STRICT
    // mode, channel-compatible) global.  This produces O(edges * domain_size) clauses instead of the
    // O(edges * domain_size²) pairwise incompatibility clauses of the naïve encoding.  When a candidate has no
    // compatible partner the clause becomes a unit clause ¬x_{t,g}, giving implicit arc-consistency filtering.
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
                if (!global_adjacent_idx(graph_data, ga, gb)) {
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
            for (size_t i1 = 0; i1 < gidx1.size(); ++i1) {
                solver.add(-lit1[i1]);
                for (size_t i2 = 0; i2 < gidx2.size(); ++i2) {
                    if (is_compatible(gidx1[i1], gidx2[i2])) {
                        solver.add(lit2[i2]);
                    }
                }
                solver.add(0);
            }
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

    // Same-rank target groups: targets sharing target_to_group must agree on global_to_same_rank_group label.
    const auto& target_to_group = constraint_data.target_to_group;
    const auto& global_rank = constraint_data.global_to_same_rank_group;
    if (!target_to_group.empty() && !global_rank.empty()) {
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

    // Cardinality: at least min_count of the listed (target, global) assignment literals must be true.
    static constexpr size_t kMaxCardinalityCombClauses = 500000;

    for (const auto& card_entry : constraint_data.cardinality_constraints) {
        const auto& pair_set = card_entry.first;
        const size_t min_count = card_entry.second;
        std::set<int> distinct_lits;
        for (const auto& pr : pair_set) {
            const size_t ti = pr.first;
            const size_t gi = pr.second;
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

template <typename TargetNode, typename GlobalNode>
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
template <typename TargetNode, typename GlobalNode>
bool SatSearchEngine<TargetNode, GlobalNode>::search(
    const GraphIndexData<TargetNode, GlobalNode>& graph_data,
    const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
    ConnectionValidationMode validation_mode,
    bool quiet_mode) {
    state_ = TopologySearchState{};
    state_.mapping.assign(graph_data.n_target, -1);
    state_.used.assign(graph_data.n_global, false);
    quiet_mode_ = quiet_mode;

    if (graph_data.n_global < graph_data.n_target) {
        state_.error_message = fmt::format(
            "Cannot map target graph to global graph: target graph is larger with {} nodes, but global graph only has "
            "{} nodes",
            graph_data.n_target,
            graph_data.n_global);
        if (quiet_mode) {
            log_debug(tt::LogFabric, "{}", state_.error_message);
        } else {
            log_error(tt::LogFabric, "{}", state_.error_message);
        }
        return false;
    }

    if (graph_data.n_target == 0) {
        return true;
    }

    auto finalize_success = [&](TopologySatSolver& solver, const TopologySatHardEncoding& enc) -> bool {
        if (!topology_sat_decode_hard_solution<TargetNode, GlobalNode>(solver, enc, state_.mapping)) {
            state_.error_message = "Topology SAT: decode failed (model inconsistent with encoding)";
            if (quiet_mode) {
                log_debug(tt::LogFabric, "{}", state_.error_message);
            } else {
                log_error(tt::LogFabric, "{}", state_.error_message);
            }
            return false;
        }
        std::fill(state_.used.begin(), state_.used.end(), false);
        for (size_t t = 0; t < state_.mapping.size(); ++t) {
            const int gi = state_.mapping[t];
            if (gi >= 0 && static_cast<size_t>(gi) < state_.used.size()) {
                state_.used[static_cast<size_t>(gi)] = true;
            }
        }
        return true;
    };

    auto solve_hard_only = [&](TopologySatSolver& solver, TopologySatHardEncoding& enc) -> bool {
        if (!topology_sat_encode_hard_constraints(solver, graph_data, constraint_data, enc, validation_mode)) {
            state_.error_message = enc.trivial_reason.empty()
                                       ? std::string("Topology SAT: encoding failed (trivial UNSAT)")
                                       : enc.trivial_reason;
            if (quiet_mode) {
                log_debug(tt::LogFabric, "{}", state_.error_message);
            } else {
                log_error(tt::LogFabric, "{}", state_.error_message);
            }
            return false;
        }
        const int status = solver.solve();
        if (status != TopologySatSolver::kSat) {
            state_.error_message = fmt::format(
                "Failed to find mapping (SAT): target graph with {} nodes cannot be embedded in global graph with {} "
                "nodes under hard constraints",
                graph_data.n_target,
                graph_data.n_global);
            if (quiet_mode) {
                log_debug(tt::LogFabric, "{}", state_.error_message);
            } else {
                log_error(tt::LogFabric, "{}", state_.error_message);
            }
            return false;
        }
        return finalize_success(solver, enc);
    };

    // No preferred constraints: one hard CNF + one solve.  RELAXED still uses the same hard encoding (channel-aware
    // support is weaker in AC-3 than STRICT; see topology_sat_encode_hard_constraints).  We intentionally skip
    // iterative relaxed-channel k-descent here: it repeated full SAT for parallel target edges (mapper auto-discovery
    // scale) and dominated wall time without changing feasibility.
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

    // Preferred: one encode + one kissat_solve.  We first compute a lower bound k_lb on the maximum number of
    // preferred targets simultaneously satisfiable, using the same AC-3 domains as the CNF (see enc).  Adding
    // at-least-k on preferred-hit Tseitin literals forces Kissat toward that bound in a single solve (exact when the
    // lower-bound search completes).  Optional relaxed-channel literals are appended when cheap.
    if (has_preferred) {
        TopologySatSolver solver;
        TopologySatHardEncoding enc;
        if (!topology_sat_encode_hard_constraints(solver, graph_data, constraint_data, enc, validation_mode)) {
            state_.error_message = enc.trivial_reason.empty()
                                       ? std::string("Topology SAT: encoding failed (trivial UNSAT)")
                                       : enc.trivial_reason;
            if (quiet_mode) {
                log_debug(tt::LogFabric, "{}", state_.error_message);
            } else {
                log_error(tt::LogFabric, "{}", state_.error_message);
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
                k_lb = topology_sat_max_preferred_lower_bound_with_encoding(
                    graph_data, constraint_data, enc, kPreferredLbDfsBudgetSmall);
            } else if (nt <= kMidPreferredLbMaxTargets) {
                k_lb = topology_sat_max_preferred_lower_bound_with_encoding(
                    graph_data, constraint_data, enc, kPreferredLbDfsBudgetMid);
            } else {
                k_lb = topology_sat_greedy_pref_lower_bound_with_encoding(graph_data, constraint_data, enc);
                if (k_lb == 0) {
                    k_lb =
                        topology_sat_max_preferred_lower_bound_with_encoding(graph_data, constraint_data, enc, 600'000);
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
                if (!topology_sat_append_relaxed_channel_threshold_literals(
                        solver, enc, graph_data, ch_lits, &ch_reason)) {
                    if (!ch_reason.empty() && !quiet_mode) {
                        log_debug(
                            tt::LogFabric, "Topology SAT: relaxed channel threshold literals skipped: {}", ch_reason);
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
        const int status = solver.solve();
        if (status != TopologySatSolver::kSat) {
            state_.error_message = fmt::format(
                "Failed to find mapping (SAT): target graph with {} nodes cannot be embedded in global graph with {} "
                "nodes under hard constraints",
                graph_data.n_target,
                graph_data.n_global);
            if (quiet_mode) {
                log_debug(tt::LogFabric, "{}", state_.error_message);
            } else {
                log_error(tt::LogFabric, "{}", state_.error_message);
            }
            return false;
        }
        return finalize_success(solver, enc);
    }

    state_.error_message = "Topology SAT: internal error (unhandled search branch)";
    return false;
}

}  // namespace tt::tt_fabric::detail
#endif  // TT_METALIUM_TOPOLOGY_SOLVER_SAT_DETAIL_HPP
