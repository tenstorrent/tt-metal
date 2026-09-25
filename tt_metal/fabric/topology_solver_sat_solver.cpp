// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topology_solver_sat_solver.hpp"

#include <cstdlib>
#include <memory>

#include <cadical.hpp>

namespace tt::tt_fabric::detail {

struct TopologySatSolver::Impl {
    mutable CaDiCaL::Solver solver;

    Impl() {
        // Diagnostic (TT_METAL_SAT_VERBOSE=1): un-quiet CaDiCaL and enable its periodic 'report' lines so we
        // can watch the conflict count climb during a long placement solve and read the final total.
        const bool verbose = std::getenv("TT_METAL_SAT_VERBOSE") != nullptr;
        solver.set("quiet", verbose ? 0 : 1);
        if (verbose) {
            solver.set("report", 1);
        }
        // Congruence closure (gate extraction, CaDiCaL >= 2.1) spends minutes on the guarded at-least-k
        // cardinality encodings this solver emits (preferred-hit objective under a host-group cap) and buys
        // nothing for these CNFs: with it off the same instances solve in seconds. Correctness is unaffected;
        // it is a preprocessing simplification only.
        solver.set("congruence", 0);
    }

    void reserve(int max_var) {
        if (max_var > 0) {
            solver.reserve(max_var);
        }
    }

    void add(int lit) { solver.add(lit); }

    void assume(int lit) { solver.assume(lit); }

    int solve() { return solver.solve(); }

    int solve_limited(int max_conflicts) {
        solver.limit("conflicts", max_conflicts);
        const int r = solver.solve();
        solver.limit("conflicts", -1);  // -1 == unlimited; clear so later solve() calls are unbounded
        return r;
    }

    int val(int lit) const {
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

    int active_vars() const { return solver.active(); }
    int64_t irredundant_clauses() const { return solver.irredundant(); }
    int64_t redundant_clauses() const { return solver.redundant(); }
    void print_statistics() { solver.statistics(); }
};

TopologySatSolver::TopologySatSolver() : impl_(std::make_unique<Impl>()) {}

void TopologySatSolver::configure_for_blocking_clause_enumeration() {
    // Only valid in CONFIGURING state (before the first non-config add()).
    // ILB: incremental lazy backtracking — reuse trail across incremental clause additions (CaDiCaL NEWS 1.7.3+).
    (void)impl_->solver.set("ilb", 2);
}

TopologySatSolver::~TopologySatSolver() = default;

TopologySatSolver::TopologySatSolver(TopologySatSolver&&) noexcept = default;

TopologySatSolver& TopologySatSolver::operator=(TopologySatSolver&&) noexcept = default;

int TopologySatSolver::declare_one_more_variable() {
    ++next_var_;
    impl_->reserve(next_var_);
    return next_var_;
}

void TopologySatSolver::add(int lit) { impl_->add(lit); }

void TopologySatSolver::assume(int lit) { impl_->assume(lit); }

int TopologySatSolver::solve() { return impl_->solve(); }

int TopologySatSolver::solve_limited(int max_conflicts) { return impl_->solve_limited(max_conflicts); }

int TopologySatSolver::val(int lit) const { return impl_->val(lit); }

int TopologySatSolver::active_vars() const { return impl_->active_vars(); }
int64_t TopologySatSolver::irredundant_clauses() const { return impl_->irredundant_clauses(); }
int64_t TopologySatSolver::redundant_clauses() const { return impl_->redundant_clauses(); }
void TopologySatSolver::print_statistics() { impl_->print_statistics(); }

}  // namespace tt::tt_fabric::detail
