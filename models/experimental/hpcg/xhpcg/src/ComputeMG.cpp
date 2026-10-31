// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Replaces HPCG 3.1's src/ComputeMG.cpp: x = M^-1 r, HPCG's multigrid V-cycle (ComputeMG_ref.cpp: one pre- and one
// post-smoothing SYMGS per level, restriction and prolongation by injection) on the P150. r and x are host vectors,
// copied to and from the card (the CG on the card runs the same launches without copies).

#include "ComputeMG.hpp"

#include "../tt/hpcg_tt.hpp"

int ComputeMG(const SparseMatrix & A, const Vector & r, Vector & x) {
  return hpcg_tt::mg(A, r, x);
}
