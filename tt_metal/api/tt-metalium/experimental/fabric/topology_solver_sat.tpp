// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Included from topology_solver.tpp after the closing brace of namespace tt::tt_fabric::detail.
// This file opens that namespace below so CaDiCaL and other includes are not nested under detail.

#ifndef TT_METALIUM_TOPOLOGY_SOLVER_SAT_TPP
#define TT_METALIUM_TOPOLOGY_SOLVER_SAT_TPP

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

#include <cadical.hpp>
#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_fabric::detail {

namespace {

template <typename TargetNode, typename GlobalNode>
bool global_adjacent_idx(
    const GraphIndexData<TargetNode, GlobalNode>& graph_data,
    size_t global_i,
    size_t global_j) {
    if (global_i >= graph_data.n_global || global_j >= graph_data.n_global) {
        return false;
    }
    const auto& adj = graph_data.global_adj_idx[global_i];
    return std::binary_search(adj.begin(), adj.end(), global_j);
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

// At-least-k on independent literals (standard CNF via combinations of (m-k+1) negated literals per clause).
inline bool topology_sat_add_at_least_k_literals(
    CaDiCaL::Solver& solver,
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
                "topology_sat: cardinality needs at least {} satisfied literals but only {} are listed",
                k,
                m);
        }
        return false;
    }
    if (k == m) {
        // All m literals must be true: m unit clauses (a single OR-of-all clause would only enforce "at least one").
        for (int lit : lits) {
            solver.add(lit);
            solver.add(0);
        }
        return true;
    }
    const size_t clause_width = m - k + 1;
    if (topology_sat_combinations_exceed_limit(m, clause_width, max_combination_clauses)) {
        if (trivial_reason != nullptr) {
            *trivial_reason = fmt::format(
                "topology_sat: cardinality CNF would exceed {} clauses (m={}, k={}); reduce literals or min_count",
                max_combination_clauses,
                m,
                k);
        }
        return false;
    }
    topology_sat_emit_combinations_indices(m, clause_width, [&](const std::vector<size_t>& comb) {
        for (size_t idx : comb) {
            solver.add(lits[idx]);
        }
        solver.add(0);
    });
    return true;
}

// One auxiliary literal per target: true iff the chosen global for that target is in its preferred set (and the
// target has at least one feasible preferred assignment literal).
template <typename TargetNode, typename GlobalNode>
void topology_sat_append_preferred_hit_indicators(
    CaDiCaL::Solver& solver,
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

template <typename TargetNode, typename GlobalNode>
bool topology_sat_encode_hard_constraints(
    CaDiCaL::Solver& solver,
    const GraphIndexData<TargetNode, GlobalNode>& graph_data,
    const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
    TopologySatHardEncoding& enc,
    ConnectionValidationMode validation_mode) {
    enc = TopologySatHardEncoding{};
    const size_t nt = graph_data.n_target;
    const size_t ng = graph_data.n_global;

    if (nt == 0) {
        return true;
    }

    enc.allowed_global_idx.resize(nt);
    enc.assign_lit.resize(nt);

    for (size_t t = 0; t < nt; ++t) {
        auto& allowed = enc.allowed_global_idx[t];
        allowed.clear();
        for (size_t g = 0; g < ng; ++g) {
            if (!constraint_data.is_valid_mapping(t, g)) {
                continue;
            }
            if (graph_data.global_deg[g] < graph_data.target_deg[t]) {
                continue;
            }
            allowed.push_back(g);
        }
        if (allowed.empty()) {
            enc.trivial_unsat = true;
            enc.trivial_reason = fmt::format("topology_sat: no allowed global for target_idx {}", t);
            return false;
        }
        enc.assign_lit[t].reserve(allowed.size());
        for (size_t k = 0; k < allowed.size(); ++k) {
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
        for (size_t i = 0; i < lits.size(); ++i) {
            for (size_t j = i + 1; j < lits.size(); ++j) {
                solver.add(-lits[i]);
                solver.add(-lits[j]);
                solver.add(0);
            }
        }
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
        const auto& Lg = lits_per_global[g];
        for (size_t i = 0; i < Lg.size(); ++i) {
            for (size_t j = i + 1; j < Lg.size(); ++j) {
                solver.add(-Lg[i]);
                solver.add(-Lg[j]);
                solver.add(0);
            }
        }
    }

    // Adjacency preservation: target edge => global edge (subgraph embedding).
    for (size_t t1 = 0; t1 < nt; ++t1) {
        for (size_t t2 : graph_data.target_adj_idx[t1]) {
            if (t2 <= t1) {
                continue;
            }
            const auto& gidx1 = enc.allowed_global_idx[t1];
            const auto& lit1 = enc.assign_lit[t1];
            const auto& gidx2 = enc.allowed_global_idx[t2];
            const auto& lit2 = enc.assign_lit[t2];
            for (size_t i1 = 0; i1 < gidx1.size(); ++i1) {
                const size_t glob1 = gidx1[i1];
                for (size_t i2 = 0; i2 < gidx2.size(); ++i2) {
                    const size_t glob2 = gidx2[i2];
                    if (glob1 == glob2) {
                        continue;
                    }
                    const bool adjacent = global_adjacent_idx(graph_data, glob1, glob2);
                    size_t required_channels = 1;
                    if (t1 < graph_data.target_conn_count.size()) {
                        const auto& tc = graph_data.target_conn_count[t1];
                        const auto itc = tc.find(t2);
                        if (itc != tc.end()) {
                            required_channels = itc->second;
                        }
                    }
                    size_t actual_channels = 0;
                    if (adjacent && glob1 < graph_data.global_conn_count.size()) {
                        const auto& gc = graph_data.global_conn_count[glob1];
                        const auto itg = gc.find(glob2);
                        if (itg != gc.end()) {
                            actual_channels = itg->second;
                        }
                    }
                    const bool strict_channel_ok =
                        validation_mode != ConnectionValidationMode::STRICT || actual_channels >= required_channels;
                    if (!adjacent || !strict_channel_ok) {
                        solver.add(-lit1[i1]);
                        solver.add(-lit2[i2]);
                        solver.add(0);
                    }
                }
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
            enc.trivial_reason = card_reason.empty() ? std::string("topology_sat: cardinality encoding failed")
                                                     : std::move(card_reason);
            return false;
        }
    }

    return true;
}

template <typename TargetNode, typename GlobalNode>
bool topology_sat_decode_hard_solution(
    CaDiCaL::Solver& solver,
    const TopologySatHardEncoding& enc,
    std::vector<int>& mapping_out) {
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

inline bool topology_mapping_use_sat_engine() {
    const char* v = std::getenv("TT_TOPOLOGY_SOLVER_ENGINE");
    if (v == nullptr || v[0] == '\0') {
        return false;
    }
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s == "sat" || s == "1" || s == "true" || s == "yes";
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
            "Cannot map target graph to global graph: target graph is larger with {} nodes, but global graph only has {} nodes",
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

    static constexpr size_t kMaxPrefHitIndicatorsForMaxOpt = 48;
    static constexpr size_t kPrefMaxCardinalityCombClauses = 500000;

    auto finalize_success = [&](CaDiCaL::Solver& solver, const TopologySatHardEncoding& enc) -> bool {
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

    auto solve_hard_only = [&](CaDiCaL::Solver& solver, TopologySatHardEncoding& enc) -> bool {
        if (!topology_sat_encode_hard_constraints(solver, graph_data, constraint_data, enc, validation_mode)) {
            state_.error_message =
                enc.trivial_reason.empty() ? std::string("Topology SAT: encoding failed (trivial UNSAT)") : enc.trivial_reason;
            if (quiet_mode) {
                log_debug(tt::LogFabric, "{}", state_.error_message);
            } else {
                log_error(tt::LogFabric, "{}", state_.error_message);
            }
            return false;
        }
        const int status = solver.solve();
        if (status != CaDiCaL::SATISFIABLE) {
            state_.error_message = fmt::format(
                "Failed to find mapping (SAT): target graph with {} nodes cannot be embedded in global graph with {} nodes under hard constraints",
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

    CaDiCaL::Solver first_solver;
    TopologySatHardEncoding first_enc;
    if (!topology_sat_encode_hard_constraints(first_solver, graph_data, constraint_data, first_enc, validation_mode)) {
        state_.error_message = first_enc.trivial_reason.empty() ? std::string("Topology SAT: encoding failed (trivial UNSAT)")
                                                                : first_enc.trivial_reason;
        if (quiet_mode) {
            log_debug(tt::LogFabric, "{}", state_.error_message);
        } else {
            log_error(tt::LogFabric, "{}", state_.error_message);
        }
        return false;
    }

    std::vector<int> pref_hit_literals;
    topology_sat_append_preferred_hit_indicators(
        first_solver, first_enc, constraint_data, pref_hit_literals);
    const size_t mp = pref_hit_literals.size();

    if (mp == 0 || mp > kMaxPrefHitIndicatorsForMaxOpt) {
        if (mp > kMaxPrefHitIndicatorsForMaxOpt && !quiet_mode) {
            log_debug(
                tt::LogFabric,
                "Topology SAT: {} preferred-hit indicators exceed cap {}; solving once without preferred maximization",
                mp,
                kMaxPrefHitIndicatorsForMaxOpt);
        }
        const int status = first_solver.solve();
        if (status != CaDiCaL::SATISFIABLE) {
            state_.error_message = fmt::format(
                "Failed to find mapping (SAT): target graph with {} nodes cannot be embedded in global graph with {} nodes under hard constraints",
                graph_data.n_target,
                graph_data.n_global);
            if (quiet_mode) {
                log_debug(tt::LogFabric, "{}", state_.error_message);
            } else {
                log_error(tt::LogFabric, "{}", state_.error_message);
            }
            return false;
        }
        return finalize_success(first_solver, first_enc);
    }

    bool cardinality_budget_failed = false;
    for (size_t k = mp; k >= 1; --k) {
        CaDiCaL::Solver solver;
        TopologySatHardEncoding enc;
        if (!topology_sat_encode_hard_constraints(solver, graph_data, constraint_data, enc, validation_mode)) {
            state_.error_message =
                enc.trivial_reason.empty() ? std::string("Topology SAT: encoding failed (trivial UNSAT)") : enc.trivial_reason;
            if (quiet_mode) {
                log_debug(tt::LogFabric, "{}", state_.error_message);
            } else {
                log_error(tt::LogFabric, "{}", state_.error_message);
            }
            return false;
        }
        std::vector<int> pref_lits;
        topology_sat_append_preferred_hit_indicators(solver, enc, constraint_data, pref_lits);
        std::string card_reason;
        if (!topology_sat_add_at_least_k_literals(
                solver, pref_lits, k, kPrefMaxCardinalityCombClauses, &card_reason)) {
            if (!quiet_mode) {
                log_debug(tt::LogFabric, "Topology SAT: preferred maximization cardinality encoding skipped: {}", card_reason);
            }
            cardinality_budget_failed = true;
            break;
        }
        const int status = solver.solve();
        if (status == CaDiCaL::SATISFIABLE) {
            return finalize_success(solver, enc);
        }
        if (k == 1) {
            break;
        }
    }

    if (!cardinality_budget_failed) {
        const int status = first_solver.solve();
        if (status != CaDiCaL::SATISFIABLE) {
            state_.error_message = fmt::format(
                "Failed to find mapping (SAT): target graph with {} nodes cannot be embedded in global graph with {} nodes under hard constraints",
                graph_data.n_target,
                graph_data.n_global);
            if (quiet_mode) {
                log_debug(tt::LogFabric, "{}", state_.error_message);
            } else {
                log_error(tt::LogFabric, "{}", state_.error_message);
            }
            return false;
        }
        return finalize_success(first_solver, first_enc);
    }

    CaDiCaL::Solver fallback_solver;
    TopologySatHardEncoding fallback_enc;
    return solve_hard_only(fallback_solver, fallback_enc);
}

}  // namespace tt::tt_fabric::detail

#endif  // TT_METALIUM_TOPOLOGY_SOLVER_SAT_TPP
