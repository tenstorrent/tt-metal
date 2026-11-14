// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// What HPCG 3.1's replaceable functions (src/*.cpp in this folder) call: the P150 implementation of the solver. No
// TT-NN types here, so HPCG's own sources compile with HPCG's flags.
#pragma once

#include "SparseMatrix.hpp"
#include "Vector.hpp"

namespace hpcg_tt {

// OptimizeProblem: builds the card's copy of A and its coarse levels (the card writes the matrix), plans the launches
// and builds the ops. Returns 0, or nonzero (with a message) when the problem is not supported.
int optimize(const SparseMatrix& A, const Vector& b);

// The solver on the card. A may be the fine matrix or any coarse level (spmv, mg, symgs). Host vectors are copied
// to and from the card in each call; cg keeps its vectors on the card and copies x back at the end.
int cg(
    const SparseMatrix& A,
    const Vector& b,
    Vector& x,
    int max_iter,
    double tolerance,
    int& niters,
    double& normr,
    double& normr0,
    double* times,
    bool doPreconditioning);
int spmv(const SparseMatrix& A, const Vector& x, Vector& y);
int mg(const SparseMatrix& A, const Vector& r, Vector& x);
int symgs(const SparseMatrix& A, const Vector& r, Vector& x);
// device memory the implementation allocated (bytes)
double device_bytes();
// false when no level has n rows (the caller then uses the reference version)
bool dot(local_int_t n, const Vector& x, const Vector& y, double& result);

}  // namespace hpcg_tt
