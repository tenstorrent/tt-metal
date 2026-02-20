// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel::sfpu {

// Hardtanh(x) = max_val if x > max_val, min_val if x < min_val, else x
// Equivalent to: clamp(x, min_val, max_val) = min(max(x, min_val), max_val)
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_hardtanh(uint param0, uint param1) {
    // Load both params outside the loop for better performance
    // param0 = min_val -> LREG2, param1 = max_val -> LREG3
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, param0 & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, param0 >> 16);
    TT_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, param1 & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, param1 >> 16);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // x = max(x, min_val) using LREG2
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
        TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, 1 /* smaller value to LREG0 */);
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);  // store max

        // x = min(x, max_val) using LREG3
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
        TTI_SFPMOV(0, p_sfpu::LREG3, p_sfpu::LREG1, 0);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, 1 /* smaller value to LREG0 */);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);  // store min

        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
