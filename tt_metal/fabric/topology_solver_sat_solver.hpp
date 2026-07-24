// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
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

    // CNF size introspection (for profiling which constraints dominate the formula). num_variables is the count of
    // declared SAT variables; num_clauses counts terminated clauses (add(0)); num_literals counts non-zero literals.
    std::size_t num_variables() const { return static_cast<std::size_t>(next_var_ < 0 ? 0 : next_var_); }
    std::size_t num_clauses() const { return num_clauses_; }
    std::size_t num_literals() const { return num_literals_; }
    // Assume a literal for the next solve() only (retracted afterwards). Lets callers add a symmetry-breaking hint
    // that is sound for any instance: if the assumption makes it UNSAT, re-solve() without it.
    void assume(int lit);
    // Force the preferred decision phase of variable |lit| to the sign of lit (sticky across solves until unphase).
    // Used to warm-start a harder incremental solve from an earlier feasible model, so CDCL branches toward that
    // model first and only "repairs" the newly added constraints instead of re-searching from scratch.
    void phase(int lit);
    void unphase(int lit);
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
    std::size_t num_clauses_ = 0;
    std::size_t num_literals_ = 0;
};

// Internal SAT function declarations — implemented in topology_solver_sat.cpp.

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

// Template overload: converts GraphIndexData/ConstraintIndexData to views and delegates.
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
