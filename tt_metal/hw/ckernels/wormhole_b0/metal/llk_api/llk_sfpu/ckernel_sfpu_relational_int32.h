// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_addrmod.h"
#include "sfpi.h"
// #include "tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_sub_int.h"
// #include "ckernel_sfpu_int_sum.h"

namespace ckernel::sfpu {

enum class RelationalOp : uint8_t {
    LT = 0,
    LE = 1,
    GT = 2,
    GE = 3,
};

// Relational ops use int32 subtract (whose result in also in the int32 range) + sign check.
// In order to avoid overflow for inputs of opposite signs, the output is determined directly from a sign check.
template <bool APPROXIMATION_MODE, RelationalOp RELATIONAL_OP, int ITERATIONS = 8>
inline void calculate_sfpu_relational_int32(const uint dst_offset) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // operand A
        TTI_SFPLOAD(p_sfpu::LREG0, INT32, ADDR_MOD_3, 0);
        // operand B
        TT_SFPLOAD(p_sfpu::LREG1, INT32, ADDR_MOD_3, dst_offset * 64);

        // Extract sign bits of A and B
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
        TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG3, 0);
        TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, 1);
        TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG3, p_sfpu::LREG3, 1);

        // LREG_3 -> 0 for inputs of same sign, 1 for inputs of different signs
        TTI_SFPXOR(0, p_sfpu::LREG2, p_sfpu::LREG3, 0);

        // if (LREG_3 == 0) -> use int32 subtract + extract sign
        TTI_SFPSETCC(0, p_sfpu::LREG3, 0, SFPSETCC_MOD1_LREG_EQ0);
        // Use 6 or LO16 as imod to convert operand B to 2's complement
        TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG1, 6);
        TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG1, p_sfpu::LREG1, 1);
        // else -> load 0 for inputs of opposite signs
        TTI_SFPCOMPC(0 /*unused*/, 0 /*unused*/, 0 /*unused*/, 0 /*unused*/);
        TTI_SFPLOADI(p_sfpu::LREG1, SFPLOADI_MOD0_USHORT, 0x00);
        // Load 1 if input A is negative
        TTI_SFPSETCC(0, p_sfpu::LREG2, 0, 2);
        TTI_SFPLOADI(p_sfpu::LREG1, SFPLOADI_MOD0_USHORT, 0x01);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPENCC(0, 0, 0, 0);

        // LREG_1 -> dest
        TTI_SFPSTORE(p_sfpu::LREG1, INT32, ADDR_MOD_3, 0);

        sfpi::dst_reg++;
    }
}

}  //  namespace ckernel::sfpu
