// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Replaces HPCG 3.1's src/OptimizeProblem.cpp (the user-tunable setup HPCG times as its optimization phase): builds
// the P150's copy of the problem. The card writes the matrix values and window offsets of all four levels, the grid
// transfer tables and b itself, from the levels' storage order (colour blocks by coordinate parity, a permutation of
// the linear system; tt/custom_ops/stencil_matrix); the host plans each core's rows and x windows.

#include "OptimizeProblem.hpp"

#include "../tt/hpcg_tt.hpp"

int OptimizeProblem(SparseMatrix & A, CGData & data, Vector & b, Vector & x, Vector & xexact) {
  (void)data;
  (void)x;
  (void)xexact;
  return hpcg_tt::optimize(A, b);
}

// Device memory held by the optimized problem (HPCG adds it to the memory use it reports)
double OptimizeProblemMemoryUse(const SparseMatrix & A) {
  (void)A;
  return hpcg_tt::device_bytes();
}
