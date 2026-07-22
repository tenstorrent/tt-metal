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

    Impl() { solver.set("quiet", 1); }

    void reserve(int max_var) {
        if (max_var > 0) {
            solver.reserve(max_var);
        }
    }

    void add(int lit) { solver.add(lit); }

    void assume(int lit) { solver.assume(lit); }

    void phase(int lit) { solver.phase(lit); }

    void unphase(int lit) { solver.unphase(lit); }

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

void TopologySatSolver::add(int lit) {
    if (lit == 0) {
        ++num_clauses_;
    } else {
        ++num_literals_;
    }
    impl_->add(lit);
}

void TopologySatSolver::assume(int lit) { impl_->assume(lit); }

void TopologySatSolver::phase(int lit) { impl_->phase(lit); }

void TopologySatSolver::unphase(int lit) { impl_->unphase(lit); }

int TopologySatSolver::solve() { return impl_->solve(); }

int TopologySatSolver::solve_limited(int max_conflicts) { return impl_->solve_limited(max_conflicts); }

int TopologySatSolver::val(int lit) const { return impl_->val(lit); }

}  // namespace tt::tt_fabric::detail
