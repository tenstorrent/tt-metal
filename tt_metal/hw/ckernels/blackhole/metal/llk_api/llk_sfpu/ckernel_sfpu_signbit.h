// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
using namespace sfpi;

namespace ckernel::sfpu {

// Optimized signbit using bitwise shift instead of branch.
// Original: v_if(val < 0.0f) = ~8 cycles/row (branch + comparison)
// Optimized: SFPSHFT bitwise = ~4 cycles/row (load + shift + and + store)
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_signbit() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::LO16, ADDR_MOD_7, 0);
        TTI_SFPSHFT((-31) & 0x1fff, p_sfpu::LREG0, p_sfpu::LREG0, 1);
        TTI_SFPAND(1, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::LO16, ADDR_MOD_7, 0);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_signbit_int32() {
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, INT32, ADDR_MOD_7, 0);
        TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG0, p_sfpu::LREG0, 1);
        TTI_SFPSTORE(p_sfpu::LREG0, INT32, ADDR_MOD_7, 0);
        dst_reg++;
    }
}

}  // namespace ckernel::sfpu
