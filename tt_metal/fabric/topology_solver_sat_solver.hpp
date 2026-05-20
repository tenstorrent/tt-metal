// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

namespace tt::tt_fabric::detail {

/**
 * Thin IPASIR-style facade over CaDiCaL (`cadical.hpp`). The wire protocol matches DIMACS conventions: positive
 * integers identify variables, 0 terminates a clause, solve() returns kSat / kUnsat / 0 (undetermined) following
 * IPASIR semantics. CaDiCaL is incremental: add() after solve() is supported.
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
    int solve();
    int val(int lit) const;

    static constexpr int kSat = 10;
    static constexpr int kUnsat = 20;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int next_var_ = 0;
};

}  // namespace tt::tt_fabric::detail
