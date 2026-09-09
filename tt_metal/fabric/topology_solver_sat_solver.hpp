// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <atomic>
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

// Distilled clause pool (TT_TOPO_SAT_POOL=1): session-persistent, in-process store of SHORT learned clauses (via the
// clause-sharing Learner hook, size <= TT_TOPO_SAT_POOL_MAX_LITS, default 8) plus root-level FIXED units harvested
// after each native solve. Entries are injected into every gimsatul DIMACS dump (after the base clauses + blocking
// clauses), shortest-first, capped at TT_TOPO_SAT_POOL_MAX_CLAUSES (default 20000) -- baking CaDiCaL's incremental
// knowledge into each cold gimsatul run. Soundness contract: entries must come from the SAME add-only base formula
// family (guaranteed within one generate_rank_bindings process; the >5000-var gate excludes the tiny intra-mesh
// solves). Each entry carries the host-cap level permanently asserted on its producer's formula when it was learned
// (cap_tag; INT_MAX = no permanent cap); a dump only includes entries whose tag is >= (looser than or equal to) the
// run's cap. Hardcap mode (MIN_MODE=3) uses a single cap level, so this is mostly bookkeeping. Dedup by sorted
// literal set (a duplicate keeps the loosest tag). In-memory, per-process; no files.
struct DistilledClausePool {
    static DistilledClausePool& instance();
    static constexpr int kNoCap = std::numeric_limits<int>::max();
    // `lits` must be a clause entailed by (base formula + "occupied <= cap_tag" when cap_tag != kNoCap).
    void publish(const std::vector<int>& lits, int cap_tag);
    // Entries valid for a run enforcing "occupied <= run_cap" (tag >= run_cap) whose literals all fit in max_var,
    // shortest-first, at most max_clauses.
    std::vector<std::vector<int>> collect(int run_cap, std::size_t max_clauses, int max_var) const;
    std::size_t size() const;

private:
    static constexpr std::size_t kMaxStored = 200'000;  // memory safety valve; new entries are dropped beyond this
    mutable std::mutex m_;
    std::map<std::vector<int>, int> entries_;  // sorted-lits -> loosest cap tag seen
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

    // Progress annotations surfaced in the 15s solve heartbeat (see topology_solver_sat_solver.cpp). Optional and
    // side-effect-free w.r.t. the search; callers driving a multi-solution enumeration set these so the heartbeat can
    // report "seeking 1st solution" vs "N/target solutions" and a coarse stage tag. Persist across solve() calls.
    void set_progress_phase(std::string_view phase);
    void set_solution_progress(std::int64_t found, std::int64_t target);

    /**
     * Must be called immediately after construction, before any add() / encoding.
     * Tunes CaDiCaL for AllSAT-style enumeration: repeated solve() after permanent blocking clauses.
     */
    void configure_for_blocking_clause_enumeration();

    // Set a CaDiCaL option (e.g. "seed", "target"). Returns false if the option/value is rejected. Used by the
    // Goal-1 base-embedding speedup experiments (TT_TOPO_SAT_SEED / TT_TOPO_SAT_FASTSAT). No-op-safe.
    bool set_option(const std::string& name, int value);

    // Point the solver's terminator at a shared cancel flag: once *flag is true, solve() aborts (returns non-SAT).
    // Used by the parallel seed portfolio so the first thread to hit SAT cancels the rest. nullptr = no cancel.
    void set_cancel_flag(std::atomic<bool>* flag);

    // Clause-sharing portfolio: connect a CaDiCaL Learner that publishes every learned clause of size <= `max_size`
    // into `pool` tagged with `producer_id`. Combined with importing peers' clauses via add() between conflict-budget
    // windows, this reproduces gimsatul-style clause sharing while every worker stays a full incremental CaDiCaL
    // (keeps BVE/warmth/enumeration -- unlike CaDiCaL's ExternalPropagator import path, which would freeze observed
    // vars and disable elimination). Call once, before solve(). `pool` must outlive this solver. No-op if pool==null.
    void enable_clause_export(ClauseSharingPool* pool, int producer_id, int max_size);

    // Write the current CNF to a DIMACS file (experiment hook: feed an external parallel/clause-sharing solver).
    // Returns true on success. Variable numbering matches declare_one_more_variable() so a model round-trips.
    bool write_dimacs(const std::string& path);

    // HYBRID: solve the current formula (+ `assumption_units` baked in as temporary hard units, since gimsatul has
    // no assumption API) with the external gimsatul binary (path from TT_TOPO_SAT_GIMSATUL_BIN), `threads` workers.
    // Returns kSat / kUnsat / 0 (unknown, e.g. no binary -> caller should fall back to CaDiCaL). On kSat, val()
    // returns gimsatul's model until the next solve()/solve_limited(). Requires the clause tee (auto-enabled when
    // TT_TOPO_SAT_GIMSATUL is set). Lets our solver keep driving descent/decode/enumeration while delegating the
    // heavy SAT search. `run_cap_k` (0 = uncapped) is the host cap the assumption units enforce for THIS run; with
    // TT_TOPO_SAT_POOL=1 it bounds which distilled pool entries may be injected (entry tag >= min(permanent cap,
    // run_cap_k)). See ONESHOT_EXTERNAL_SAT_EXPERIMENT.md and POOL_EXPERIMENT_PLAN.md.
    int gimsatul_solve(int threads, const std::vector<int>& assumption_units, std::size_t run_cap_k = 0);

    // Distilled pool bookkeeping: record that the formula now PERMANENTLY enforces "occupied <= cap" (a unit clause
    // was add()'ed). Lowers the cap tag stamped on this solver's future pool exports AND the run cap of its future
    // gimsatul dumps. Cheap no-op when the pool is off.
    void note_permanent_cap(std::size_t cap);

    // HYBRID warm-start: after gimsatul_solve() found a model, bias CaDiCaL's decision phases toward it, so a
    // subsequent NATIVE incremental descent starts from gimsatul's feasible solution (gimsatul does the heavy first
    // solve; CaDiCaL does the cheap incremental tightening, reusing its own learned clauses). No-op if no model.
    void phase_hint_from_last_gimsatul_model();

    static constexpr int kSat = 10;
    static constexpr int kUnsat = 20;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int next_var_ = 0;
    std::size_t num_clauses_ = 0;
    std::size_t num_literals_ = 0;
    // EXPERIMENT: faithful DIMACS tee. CaDiCaL::write_dimacs drops clauses added incrementally after a solve()
    // (e.g. the occupancy/host-cap clauses built in solve_minimize_groups), so when dump_record_ is on we record
    // every literal add()'ed and emit DIMACS from this tape instead. Enabled only when TT_TOPO_SAT_DUMP_DIMACS is
    // set, so production pays nothing.
    bool dump_record_ = false;
    std::vector<int> dump_tape_;
    // HYBRID: when a gimsatul_solve() found a model, val() answers from it (per-var sign: +1 true, -1 false, 0 unset)
    // instead of CaDiCaL, until the next native solve()/solve_limited() clears it.
    bool have_gimsatul_model_ = false;
    std::vector<signed char> gimsatul_model_;
};

// Internal SAT function declarations — implemented in topology_solver_sat.cpp.

// quiet_mode suppresses the per-phase [topo-sat-profile] timing lines (they are logged at debug level otherwise).
bool topology_sat_encode_hard_constraints(
    TopologySatSolver& solver,
    const TopologySatGraphView& graph_data,
    const TopologySatConstraintView& constraint_data,
    TopologySatHardEncoding& enc,
    ConnectionValidationMode validation_mode = ConnectionValidationMode::RELAXED,
    bool quiet_mode = false);

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
    ConnectionValidationMode validation_mode = ConnectionValidationMode::RELAXED,
    bool quiet_mode = false) {
    return topology_sat_encode_hard_constraints(
        solver,
        TopologySatGraphView(graph_data),
        TopologySatConstraintView(constraint_data),
        enc,
        validation_mode,
        quiet_mode);
}

}  // namespace tt::tt_fabric::detail
