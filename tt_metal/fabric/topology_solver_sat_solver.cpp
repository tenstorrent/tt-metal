// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topology_solver_sat_solver.hpp"

#include <climits>
#include <cstdlib>
#include <memory>
#include <string_view>

#include <cadical.hpp>

#ifdef TT_METAL_FABRIC_KISSAT
// Clean public wrapper (extern "C"; keeps kissat's src/ off our include path — see the header).
#include <kissat.h>
#endif

namespace tt::tt_fabric::detail {

// ── SAT engine interface ────────────────────────────────────────────────────────────────────────────────
// The deterministic min-host solver runs on a pluggable engine. Historically the only engine was CaDiCaL
// (CadicalSatEngine below); kissat_extras (KissatSatEngine) is a faster deterministic incremental alternative
// selected via TT_TOPO_SAT_ENGINE=kissat. See tt_metal/fabric/KISSAT_SWAP_PLAN.md.
//
// The interface is exactly the core the solver needs: reserve / add / assume / solve / solve_limited / val,
// plus a one-time enumeration configure. (The CaDiCaL-only pool/learner/phase machinery is intentionally not
// part of this interface — it was net-negative and is not carried forward.)
struct SatEngine {
    virtual ~SatEngine() = default;
    virtual void reserve(int max_var) = 0;
    virtual void add(int lit) = 0;
    virtual void assume(int lit) = 0;
    virtual int solve() = 0;
    virtual int solve_limited(int max_conflicts) = 0;
    virtual int val(int lit) const = 0;
    // Called once after construction, before any non-config add(): tune the engine for repeated solve()
    // after permanent blocking clauses (AllSAT-style enumeration).
    virtual void configure_for_enumeration() = 0;
};

// ── CaDiCaL engine (deterministic default; formerly the only engine) ─────────────────────────────────────
struct CadicalSatEngine final : SatEngine {
    mutable CaDiCaL::Solver solver;

    CadicalSatEngine() {
        solver.set("quiet", 1);
        // Congruence closure (gate extraction, CaDiCaL >= 2.1) spends minutes on the guarded at-least-k
        // cardinality encodings this solver emits (preferred-hit objective under a host-group cap) and buys
        // nothing for these CNFs: with it off the same instances solve in seconds. Correctness is unaffected;
        // it is a preprocessing simplification only.
        solver.set("congruence", 0);
    }

    void reserve(int max_var) override {
        if (max_var > 0) {
            solver.reserve(max_var);
        }
    }

    void add(int lit) override { solver.add(lit); }

    void assume(int lit) override { solver.assume(lit); }

    int solve() override { return solver.solve(); }

    int solve_limited(int max_conflicts) override {
        solver.limit("conflicts", max_conflicts);
        const int r = solver.solve();
        solver.limit("conflicts", -1);  // -1 == unlimited; clear so later solve() calls are unbounded
        return r;
    }

    int val(int lit) const override {
        const int a = std::abs(lit);
        const int r = solver.val(a);
        if (r == 0) {
            return 0;
        }
        if (lit > 0) {
            return (r > 0) ? lit : -lit;
        }
        return (r < 0) ? lit : -lit;
    }

    void configure_for_enumeration() override {
        // ILB: incremental lazy backtracking — reuse trail across incremental clause additions (CaDiCaL 1.7.3+).
        (void)solver.set("ilb", 2);
    }
};

#ifdef TT_METAL_FABRIC_KISSAT
// ── kissat engine (incremental Kissat fork; deterministic, faster core) ──────────────────────────────────
// Selected via TT_TOPO_SAT_ENGINE=kissat. kissat is single-threaded (hence deterministic) and incremental
// (add() after solve() is supported, which blocking-clause enumeration relies on). See KISSAT_SWAP_PLAN.md.
struct KissatSatEngine final : SatEngine {
    kissat* solver = nullptr;

    KissatSatEngine() {
        solver = kissat_init();
        kissat_set_option(solver, "quiet", 1);
        // Enable incremental solving so add() after solve() (blocking-clause enumeration) is legal. On by
        // default in kissat_extras; set explicitly so behaviour does not depend on the vendored default.
        kissat_set_option(solver, "incremental", 1);
    }

    ~KissatSatEngine() override {
        if (solver != nullptr) {
            kissat_release(solver);
        }
    }

    void reserve(int max_var) override {
        if (max_var > 0) {
            kissat_reserve(solver, max_var);
        }
    }

    void add(int lit) override { kissat_add(solver, lit); }

    void assume(int lit) override { kissat_assume(solver, lit); }

    int solve() override { return kissat_solve(solver); }  // 10 SAT / 20 UNSAT / 0 (IPASIR), matches CaDiCaL

    int solve_limited(int max_conflicts) override {
        kissat_set_conflict_limit(solver, static_cast<unsigned>(max_conflicts));
        const int r = kissat_solve(solver);
        kissat_set_conflict_limit(solver, UINT_MAX);  // clear so later solve() calls are unbounded
        return r;
    }

    int val(int lit) const override {
        const int a = std::abs(lit);
        const int r = kissat_value(solver, a);  // +a true / -a false / 0 unassigned (same convention as CaDiCaL)
        if (r == 0) {
            return 0;
        }
        if (lit > 0) {
            return (r > 0) ? lit : -lit;
        }
        return (r < 0) ? lit : -lit;
    }

    void configure_for_enumeration() override {
        // kissat has no `ilb` analog; incremental is already enabled in the ctor. (Future: KISSAT_SWAP_PLAN.md
        // §8 suggests kissat_set_configuration(solver, "sat") here for the SAT-dominated min-host solves.)
    }
};
#endif  // TT_METAL_FABRIC_KISSAT

// ── Engine selection ─────────────────────────────────────────────────────────────────────────────────────
// TT_TOPO_SAT_ENGINE = "cadical" (default) | "kissat". Defaulting to CaDiCaL keeps the swap behind an opt-in
// until validated (KISSAT_SWAP_PLAN.md §6); once validated the default flips and CaDiCaL becomes the fallback.
static std::unique_ptr<SatEngine> make_engine() {
    const char* env = std::getenv("TT_TOPO_SAT_ENGINE");
    const std::string_view name = (env != nullptr && env[0] != '\0') ? env : "cadical";
#ifdef TT_METAL_FABRIC_KISSAT
    if (name == "kissat") {
        return std::make_unique<KissatSatEngine>();
    }
#endif
    return std::make_unique<CadicalSatEngine>();
}

struct TopologySatSolver::Impl {
    std::unique_ptr<SatEngine> engine = make_engine();
};

TopologySatSolver::TopologySatSolver() : impl_(std::make_unique<Impl>()) {}

void TopologySatSolver::configure_for_blocking_clause_enumeration() {
    // Only valid in CONFIGURING state (before the first non-config add()).
    impl_->engine->configure_for_enumeration();
}

TopologySatSolver::~TopologySatSolver() = default;

TopologySatSolver::TopologySatSolver(TopologySatSolver&&) noexcept = default;

TopologySatSolver& TopologySatSolver::operator=(TopologySatSolver&&) noexcept = default;

int TopologySatSolver::declare_one_more_variable() {
    ++next_var_;
    impl_->engine->reserve(next_var_);
    return next_var_;
}

void TopologySatSolver::add(int lit) { impl_->engine->add(lit); }

void TopologySatSolver::assume(int lit) { impl_->engine->assume(lit); }

int TopologySatSolver::solve() { return impl_->engine->solve(); }

int TopologySatSolver::solve_limited(int max_conflicts) { return impl_->engine->solve_limited(max_conflicts); }

int TopologySatSolver::val(int lit) const { return impl_->engine->val(lit); }

}  // namespace tt::tt_fabric::detail
