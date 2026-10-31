// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Replaces HPCG 3.1's src/ComputeDotProduct.cpp: x . y on the P150 (tt/custom_ops/cg_vector's dot and finish). x and
// y are host vectors, copied to the card in each call. Vectors of a length no level has use the reference version.

#include "ComputeDotProduct.hpp"
#include "ComputeDotProduct_ref.hpp"

#include "../tt/hpcg_tt.hpp"

int ComputeDotProduct(const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce, bool & isOptimized) {
  (void)time_allreduce;  // one process: no all-reduce
  if (hpcg_tt::dot(n, x, y, result)) return 0;
  isOptimized = false;
  return ComputeDotProduct_ref(n, x, y, result, time_allreduce);
}
