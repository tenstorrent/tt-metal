// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Replaces HPCG 3.1's src/ComputeSPMV.cpp: y = A x on the P150 (tt/custom_ops/sparse_rows). x and y are host vectors,
// copied to and from the card in each call (the CG on the card does not call this; HPCG's TestSymmetry does).

#include "ComputeSPMV.hpp"

#include "../tt/hpcg_tt.hpp"

int ComputeSPMV(const SparseMatrix & A, Vector & x, Vector & y) {
  return hpcg_tt::spmv(A, x, y);
}
