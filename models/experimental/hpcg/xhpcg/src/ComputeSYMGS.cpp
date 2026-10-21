// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Replaces HPCG 3.1's src/ComputeSYMGS.cpp: one symmetric Gauss-Seidel sweep on the P150, in the 8-colour order (a
// permutation HPCG allows; the extra CG iterations it costs are counted by HPCG's Optimized CG Setup phase). r and x
// are host vectors, copied to and from the card.

#include "ComputeSYMGS.hpp"

#include "../tt/hpcg_tt.hpp"

int ComputeSYMGS(const SparseMatrix & A, const Vector & r, Vector & x) {
  return hpcg_tt::symgs(A, r, x);
}
