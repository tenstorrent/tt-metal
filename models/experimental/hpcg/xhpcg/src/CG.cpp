// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Replaces HPCG 3.1's src/CG.cpp: the same preconditioned CG algorithm (CG_ref.cpp), run on the P150. The CG start
// (r = b - A x, |r|) and each iteration (MG preconditioner, the dot products, the p, x and r updates, |r|) are each
// one recorded trace on the card; the scalars alpha and beta are computed and kept on the card. HPCG's per-kernel
// times (times[1..5]) are not split out: the iteration runs as one trace, so only the total (times[0]) is measured.

#include "CG.hpp"

#include "../tt/hpcg_tt.hpp"

int CG(const SparseMatrix & A, CGData & data, const Vector & b, Vector & x,
    const int max_iter, const double tolerance, int & niters, double & normr, double & normr0,
    double * times, bool doPreconditioning) {
  (void)data;  // the CG vectors live on the card
  return hpcg_tt::cg(A, b, x, max_iter, tolerance, niters, normr, normr0, times, doPreconditioning);
}
