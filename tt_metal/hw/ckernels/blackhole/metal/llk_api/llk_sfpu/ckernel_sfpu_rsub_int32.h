// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_rsub_int32(const uint dst_offset) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // operand A - int32
        TTI_SFPLOAD(p_sfpu::LREG0, INT32, ADDR_MOD_7, 0);
        // operand B - int32 (offset by dst_offset * dest tile size)
        TT_SFPLOAD(p_sfpu::LREG1, INT32, ADDR_MOD_7, dst_offset * 64);
        // Reverse subtraction is performed using 2's complement by adding B to the negation of A: LREG1 + (-LREG0)
        // Use 6 as imod to convert operand A to 2's complement
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 6);
        // Store result from LREG_0 to dest
        TTI_SFPSTORE(p_sfpu::LREG0, INT32, ADDR_MOD_7, 0);
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
