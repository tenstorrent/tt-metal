// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <vector>

#include <tt-metalium/experimental/fabric/topology_solver.hpp>

namespace tt::tt_fabric::detail {

/**
 * Thin IPASIR-style facade over CaDiCaL (`cadical.hpp`). DIMACS wire protocol: positive variable ids, 0 ends a
 * clause; solve() returns kSat / kUnsat / 0 (IPASIR). CaDiCaL is incremental — add() after solve() is supported,
 * which multi-model and blocking-clause enumeration rely on for throughput versus one-shot solvers.
 */
struct TopologySatSolver {
    TopologySatSolver();
    ~TopologySatSolver();

    TopologySatSolver(const TopologySatSolver&) = delete;
    TopologySatSolver& operator=(const TopologySatSolver&) = delete;

    TopologySatSolver(TopologySatSolver&&) noexcept;
    TopologySatSolver& operator=(TopologySatSolver&&) noexcept;

    int declare_one_more_variable();
    void add(int lit);
    // Assume a literal for the next solve() only (retracted afterwards). Lets callers add a symmetry-breaking hint
    // that is sound for any instance: if the assumption makes it UNSAT, re-solve() without it.
    void assume(int lit);
    int solve();
    // Solve capped at `max_conflicts` conflicts. Returns kSat / kUnsat, or 0 (IPASIR "unknown") when the budget
    // is exhausted before a verdict. Lets a caller try an optional/expensive constraint (a tight host-budget
    // minimization) without paying an unbounded proof when it is intractable -- on 0/kUnsat the caller falls back.
    // The limit is cleared afterwards so subsequent solve() calls are unbounded.
    int solve_limited(int max_conflicts);
    int val(int lit) const;

    /**
     * Must be called immediately after construction, before any add() / encoding.
     * Tunes CaDiCaL for AllSAT-style enumeration: repeated solve() after permanent blocking clauses.
     */
    void configure_for_blocking_clause_enumeration();

    static constexpr int kSat = 10;
    static constexpr int kUnsat = 20;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int next_var_ = 0;
};

// Internal SAT function declarations — implemented in topology_solver_sat.cpp.

// At-most-one on listed positive literals: pairwise for small n, else Sinz sequential encoding.
void topology_sat_add_at_most_one(TopologySatSolver& solver, const std::vector<int>& lits);

// Generic occupancy indicators, shared by the inter-mesh host-group cap and the master placement's
// per-host packing. `group_member_lits[g][m]` is the list of literals whose disjunction means member m of
// group g is used. For each non-empty group this appends one occupancy literal `occ` to `occ_out` with
// occ <=> OR(member used); when `all_or_nothing` is set it also forces occ => every reachable member used
// (a used group is FULLY used -- the "fill every host" packing constraint). Asserting clauses are guarded
// by `extra_lit` (0 = unguarded), so a caller can assume(extra_lit) and retract it to make it optional.
void topology_sat_build_occupancy_indicators(
    TopologySatSolver& solver,
    const std::vector<std::vector<std::vector<int>>>& group_member_lits,
    bool all_or_nothing,
    std::vector<int>& occ_out,
    int extra_lit = 0);

bool topology_sat_encode_hard_constraints(
    TopologySatSolver& solver,
    const TopologySatGraphView& graph_data,
    const TopologySatConstraintView& constraint_data,
    TopologySatHardEncoding& enc,
    ConnectionValidationMode validation_mode = ConnectionValidationMode::RELAXED);

bool topology_sat_decode_hard_solution(
    TopologySatSolver& solver, const TopologySatHardEncoding& enc, std::vector<int>& mapping_out);

bool topology_sat_add_blocking_clause_for_mapping(
    TopologySatSolver& solver, TopologySatHardEncoding& enc, const std::vector<int>& raw_mapping, bool unique_shapes);

// indicator <=> OR_p (a_p & b_p) for positive seat/assign literals (Tseitin AND-of-pair OR).
bool topology_sat_define_indicator_as_or_of_pairwise_and(
    TopologySatSolver& solver, int indicator, const std::vector<std::pair<int, int>>& pair_lits);

// At-least-k on listed literals; optional extra_lit guards the constraint (extra_lit => at-least-k).
bool topology_sat_add_at_least_k_literals(
    TopologySatSolver& solver,
    const std::vector<int>& lits,
    std::size_t k,
    std::size_t max_combination_clauses,
    std::string* trivial_reason,
    int extra_lit = 0);

// Template overload: converts GraphIndexData/ConstraintIndexData to views and delegates.
// TODO: drop the views (see TopologySatGraphView in topology_solver.hpp) once SAT can take a
// non-template index base instead.
template <typename TargetNode, typename GlobalNode>
bool topology_sat_encode_hard_constraints(
    TopologySatSolver& solver,
    const GraphIndexData<TargetNode, GlobalNode>& graph_data,
    const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
    TopologySatHardEncoding& enc,
    ConnectionValidationMode validation_mode = ConnectionValidationMode::RELAXED) {
    return topology_sat_encode_hard_constraints(
        solver,
        TopologySatGraphView(graph_data),
        TopologySatConstraintView(constraint_data),
        enc,
        validation_mode);
}

}  // namespace tt::tt_fabric::detail
