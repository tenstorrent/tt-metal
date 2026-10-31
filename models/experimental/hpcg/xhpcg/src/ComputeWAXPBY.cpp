// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Replaces HPCG 3.1's src/ComputeWAXPBY.cpp. The CG on the card (CG.cpp) fuses its three vector updates into its own
// launches (tt/custom_ops/cg_vector's axpy and xr) and does not call this; HPCG's other phases do not either. A call
// would get the reference version, and HPCG's report then says so.

#include "ComputeWAXPBY.hpp"
#include "ComputeWAXPBY_ref.hpp"

int ComputeWAXPBY(const local_int_t n, const double alpha, const Vector & x,
    const double beta, const Vector & y, Vector & w, bool & isOptimized) {
  isOptimized = false;
  return ComputeWAXPBY_ref(n, alpha, x, beta, y, w);
}
