// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

inline void alt_complex_rotate90_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS = 4>
inline void calculate_alt_complex_rotate90() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat val = dst_reg[0];
        dst_reg[0] = -vFloat(dst_reg[1]);
        dst_reg[1] = val;
        dst_reg += 2;
    }
}

}  // namespace sfpu
}  // namespace ckernel
