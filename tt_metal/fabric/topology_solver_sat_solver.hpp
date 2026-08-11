// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <tt-metalium/experimental/fabric/topology_solver.hpp>

namespace tt::tt_fabric::detail {

// Lock-guarded learned-clause exchange for the clause-sharing portfolio. N incremental CaDiCaL workers cooperate by
// sharing short learned clauses: producers publish() clauses (captured via CaDiCaL's Learner callback); each consumer
// keeps its own monotonic cursor and drain()s the clauses it has not yet seen, skipping its own. Soundness: every
// shared clause is one the producer LEARNED from the same base formula, hence entailed by it, so importing it removes
// no model and changes no answer -- it only prunes the consumer's search. Facade-level (no CaDiCaL types) so the
// portfolio driver can own it and hand it to each worker's solver.
struct ClauseSharingPool {
    void publish(int producer_id, const std::vector<int>& lits) {
        std::lock_guard<std::mutex> lk(m_);
        clauses_.push_back(Entry{producer_id, lits});
    }
    // Append every clause after `cursor` not produced by `consumer_id` into `out`; advance `cursor` to the end.
    void drain(int consumer_id, std::size_t& cursor, std::vector<std::vector<int>>& out) {
        std::lock_guard<std::mutex> lk(m_);
        for (std::size_t i = cursor; i < clauses_.size(); ++i) {
            if (clauses_[i].producer_id != consumer_id) {
                out.push_back(clauses_[i].lits);
            }
        }
        cursor = clauses_.size();
    }
    std::size_t size() const {
        std::lock_guard<std::mutex> lk(m_);
        return clauses_.size();
    }

private:
    struct Entry {
        int producer_id;
        std::vector<int> lits;
    };
    mutable std::mutex m_;
    std::vector<Entry> clauses_;
};

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

    // Set a CaDiCaL option (e.g. "seed", "target", "phase"). Returns false if the option/value is rejected. Used by
    // the clause-sharing portfolio to diversify workers (per-worker seed) and bias toward finding a model fast.
    bool set_option(const std::string& name, int value);

    // Point the solver's terminator at a shared cancel flag: once *flag is true, solve() aborts (returns non-SAT).
    // Used by the clause-sharing portfolio so the first worker to hit SAT cancels the rest. nullptr = no cancel.
    void set_cancel_flag(std::atomic<bool>* flag);

    // Progress annotations surfaced in the 15s solve heartbeat. Optional and side-effect-free w.r.t. the search;
    // callers driving a multi-solution enumeration set these so the heartbeat reports "seeking 1st solution" vs
    // "N/target solutions" and a coarse stage tag. Persist across solve() calls.
    void set_progress_phase(std::string_view phase);
    void set_solution_progress(std::int64_t found, std::int64_t target);

    // Clause-sharing portfolio: connect a CaDiCaL Learner that publishes every learned clause of size <= `max_size`
    // into `pool` tagged with `producer_id`. Combined with importing peers' clauses via add() between conflict-budget
    // windows, this reproduces gimsatul-style clause sharing while every worker stays a full incremental CaDiCaL
    // (keeps BVE/warmth/enumeration -- unlike CaDiCaL's ExternalPropagator import path, which would freeze observed
    // vars and disable elimination). Call once, before solve(). `pool` must outlive this solver. No-op if pool==null.
    void enable_clause_export(ClauseSharingPool* pool, int producer_id, int max_size);

    static constexpr int kSat = 10;
    static constexpr int kUnsat = 20;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int next_var_ = 0;
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
