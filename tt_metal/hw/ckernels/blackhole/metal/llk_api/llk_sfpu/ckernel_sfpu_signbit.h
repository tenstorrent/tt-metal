// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "sfpi.h"
using namespace sfpi;

namespace ckernel::sfpu {

// TODO: Implement using bitwise comparision
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_signbit() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat val = dst_reg[0];
        v_if(val < 0.0f) { val = 1.0f; }
        v_else { val = 0.0f; }
        v_endif;
        dst_reg[0] = val;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_signbit_int32() {
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, INT32, ADDR_MOD_7, 0);
        TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG0, p_sfpu::LREG0, 1);
        TTI_SFPSTORE(p_sfpu::LREG0, INT32, ADDR_MOD_3, 0);
        dst_reg++;
    }
}

}  // namespace ckernel::sfpu
