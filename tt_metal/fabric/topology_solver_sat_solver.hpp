// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

extern "C" {
#include <kissat.h>
}

namespace tt::tt_fabric::detail {

/**
 * Thin IPASIR-style wrapper around Kissat (positive DIMACS literals; 0 ends a clause).
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

}  // namespace tt::tt_fabric::detail
