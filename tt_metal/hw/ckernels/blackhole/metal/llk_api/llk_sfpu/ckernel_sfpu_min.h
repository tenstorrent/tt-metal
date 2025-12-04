// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel {
namespace sfpu {

// min(a, b) where a = dst_reg[0], b = dst_reg[32]
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_min() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat a = sfpi::dst_reg[0];
        sfpi::vFloat b = sfpi::dst_reg[32];

        // Branchless min: vec_min_max puts min in first arg, max in second
        sfpi::vec_min_max(a, b);  // After: a = min(a,b), b = max(a,b)
        sfpi::dst_reg[0] = a;

        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
