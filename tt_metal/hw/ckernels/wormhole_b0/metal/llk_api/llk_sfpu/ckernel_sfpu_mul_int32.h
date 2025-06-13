// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void mul_int32(const uint dst_offset) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size = 64;
        // operand A - int32
        TTI_SFPLOAD(p_sfpu::LREG0, INT32, ADDR_MOD_3, 0);
        // operand B - int32
        TT_SFPLOAD(p_sfpu::LREG1, INT32, ADDR_MOD_3, dst_offset * dst_tile_size);

        TTI_SFPLOADI(p_sfpu::LREG7, SFPLOADI_MOD0_USHORT, 0xffff);

        // // copy
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
        TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG3, 0);

        // upper bits
        TTI_SFPSHFT((-16), p_sfpu::LREG0, p_sfpu::LREG2, 1);  // LREG2 = A[31:16]
        TTI_SFPAND(0, p_sfpu::LREG7, p_sfpu::LREG2, 0);       // Clear upper bits
        TTI_SFPSHFT((-16), p_sfpu::LREG1, p_sfpu::LREG3, 1);  // LREG3 = B[31:16]
        TTI_SFPAND(0, p_sfpu::LREG7, p_sfpu::LREG3, 0);       // Clear upper bits

        // lower bits
        TTI_SFPAND(0, p_sfpu::LREG7, p_sfpu::LREG0, 0);  // LREG0 = A[15:0]
        TTI_SFPAND(0, p_sfpu::LREG7, p_sfpu::LREG1, 0);  // LREG1 = B[15:0]

        // int16 -> fp32
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);  // a_low
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);  // b_low
        TTI_SFPCAST(p_sfpu::LREG2, p_sfpu::LREG2, 0);  // a_hi
        TTI_SFPCAST(p_sfpu::LREG3, p_sfpu::LREG3, 0);  // b_hi

        // multiply in fp32
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG4, 0);  // L4 = A_lo * B_lo
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG5, 0);  // L5 = A_lo * B_hi
        TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);  // L6 = A_hi * B_lo
        TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG7, 0);  // L7 = A_hi * B_hi

        // cast back
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG4, p_sfpu::LREG4, 7);
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG5, p_sfpu::LREG5, 7);
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG6, p_sfpu::LREG6, 7);
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG7, p_sfpu::LREG7, 7);

        // shift back
        TTI_SFPSHFT(16, 0, p_sfpu::LREG5, 1);  // L5 = (A_lo * B_hi) << 16
        TTI_SFPSHFT(16, 0, p_sfpu::LREG6, 1);  // L6 = (A_hi * B_lo) << 16
        TTI_SFPSHFT(32, 0, p_sfpu::LREG7, 1);  // L7 = (A_hi * B_hi) << 32

        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG4, SFPIADD_MOD1_CC_NONE);
        TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG4, SFPIADD_MOD1_CC_NONE);
        TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG4, SFPIADD_MOD1_CC_NONE);

        TTI_SFPSTORE(p_sfpu::LREG4, INT32, ADDR_MOD_3, 0);

        // // byte 1
        // TTI_SFPSHFT((-8) & 0xfff, p_sfpu::LREG0, p_sfpu::LREG4, 1);        // Shift input 1 >> 8
        // TTI_SFPSHFT((-8) & 0xfff, p_sfpu::LREG0, p_sfpu::LREG5, 1);        // Shift input 2 >> 8
        // TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
        // TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG3, 0);
        // TTI_SFPAND(0, p_sfpu::LREG7, p_sfpu::LREG0, 0);        // Byte 1
        // TTI_SFPAND(0, p_sfpu::LREG7, p_sfpu::LREG1, 0);

        // TTI_SFPSHFT((-8) & 0xfff, 0, p_sfpu::LREG2, 1);        // Shift >> another 8
        // TTI_SFPSHFT((-8) & 0xfff, 0, p_sfpu::LREG3, 1);
        // TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
        // TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG3, 0);
        // TTI_SFPAND(0, p_sfpu::LREG7, p_sfpu::LREG0, 0);        // Byte 2
        // TTI_SFPAND(0, p_sfpu::LREG7, p_sfpu::LREG1, 0);

        // TTI_SFPSHFT((-8) & 0xfff, 0, p_sfpu::LREG2, 1);        // Shift >> 24 total
        // TTI_SFPSHFT((-8) & 0xfff, 0, p_sfpu::LREG3, 1);
        // TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
        // TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG3, 0);
        // TTI_SFPAND(0, p_sfpu::LREG7, p_sfpu::LREG0, 0);        // Byte 3
        // TTI_SFPAND(0, p_sfpu::LREG7, p_sfpu::LREG1, 0);

        // // The following cast+mul+cast method provides accurate results if the product of the inputs < 2**24
        // // since float32 can exactly represent integer up to 2**24. To preserve accuracy beyond 2**24,
        // // we could split the 16-bit input into two 8-bit chunks, cast to fp32, multiply and then cast back.

        // // int32 -> fp32
        // TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);
        // TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);
        // TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        // TTI_NOP;
        // // fp32 -> int32
        // TTI_SFPLOADI(p_sfpu::LREG7, SFPLOADI_MOD0_USHORT, 0xffff);
        // TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
        // TTI_SFPSHFT((-16) & 0xfffff, 0, p_sfpu::LREG2, 1);  // high bits
        // // TTI_SFPAND(0, p_sfpu::LREG7, p_sfpu::LREG2, 0);
        // TTI_SFPAND(0, p_sfpu::LREG7, p_sfpu::LREG0, 0);  // low bits
        // TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG0, 7);
        // TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG2, p_sfpu::LREG2, 7);
        // TTI_SFPSHFT(16, 0, p_sfpu::LREG2, 1); // high bits
        // TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG2, SFPIADD_MOD1_CC_NONE);
        // TTI_SFPSTORE(p_sfpu::LREG0, INT32, ADDR_MOD_3, 0);

        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
