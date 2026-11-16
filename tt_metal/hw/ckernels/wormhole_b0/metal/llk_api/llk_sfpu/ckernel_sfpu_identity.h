// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "ckernel_sfpu_recip.h"
#include <limits>

#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_identity() {
    static_assert(ITERATIONS % 2 == 0);
    // Two rows per iteration share one dst_reg increment.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d += 2) {
        vFloat a = dst_reg[0];
        vFloat b = dst_reg[1];
        dst_reg[0] = a;
        dst_reg[1] = b;
        dst_reg += 2;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_identity_uint() {
    static_assert(ITERATIONS % 2 == 0);
    // Two rows per iteration share one dst_reg increment.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d += 2) {
        vUInt a = dst_reg[0];
        vUInt b = dst_reg[1];
        dst_reg[0] = a;
        dst_reg[1] = b;
        dst_reg += 2;
    }
}

}  // namespace sfpu
}  // namespace ckernel
