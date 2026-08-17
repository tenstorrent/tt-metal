// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel::sfpu {

inline void square_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_square() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = v * v;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
