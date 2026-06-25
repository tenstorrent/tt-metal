// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_ops.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Left shift by an immediate scalar amount
template <bool APPROXIMATION_MODE, InstrModLoadStore INSTRUCTION_MODE = InstrModLoadStore::INT32, int ITERATIONS = 8>
inline void calculate_left_shift(const uint shift_amt) {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(0, INSTRUCTION_MODE, ADDR_MOD_7, 0);
        TT_SFPSHFT(shift_amt, 0, 0, 1);
        TTI_SFPSTORE(0, INSTRUCTION_MODE, ADDR_MOD_7, 0);
        dst_reg++;
    }
}

// Arithmetic right shift by an immediate scalar amount
template <bool APPROXIMATION_MODE, InstrModLoadStore INSTRUCTION_MODE = InstrModLoadStore::INT32, int ITERATIONS = 8>
inline void calculate_right_shift(const uint shift_amt) {
    sfpi::vConstIntPrgm0 = shift_amt;  // LREG12 = shift amount
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, 0);
        TTI_SFPMOV(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);  // LREG1 - shift amount
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG4, 0);   // save value for sign check
        // if (shift_amount < 0 OR shift_amount >= 32) -> result should be 0
        TTI_SFPSETCC(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);
        TTI_SFPIADD(0xFE0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LCONST_0);  // 0xFE0 = -32
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPENCC(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG1, 6);  // negate shift_amount to shift right
        // shift right
        TTI_SFPSHFT(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        // if value was negative, shift in 1's manually (arithmetic shift)
        TTI_SFPSETCC(0, p_sfpu::LREG4, p_sfpu::LREG0, 0);     // only run if value is negative
        TTI_SFPSETCC(0, p_sfpu::LREG1, p_sfpu::LREG0, 2);     // only needed if shift_amount>0
        TTI_SFPIADD(0x020, p_sfpu::LREG1, p_sfpu::LREG2, 5);  // 32-shift_amount (0x020 = 32)
        TTI_SFPNOT(0, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);    // all 1's into LREG3
        TTI_SFPSHFT(0, p_sfpu::LREG2, p_sfpu::LREG3, 0);      // shift all 1's by 32-shift_amount
        TTI_SFPOR(0, p_sfpu::LREG3, p_sfpu::LREG0, 0);        // OR in the 1's
        TTI_SFPENCC(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, 0);
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
