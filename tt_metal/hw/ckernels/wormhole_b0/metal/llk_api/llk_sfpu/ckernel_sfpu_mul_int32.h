// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void mul_int32(const uint dst_offset) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size = 64;
        // operand A - int32
        TTI_SFPLOAD(p_sfpu::LREG0, INT32, ADDR_MOD_3, 0);
        // operand B - int32
        TT_SFPLOAD(p_sfpu::LREG1, INT32, ADDR_MOD_3, dst_offset * dst_tile_size);

        // INT32 split into 8-bit inputs
        // mask
        TTI_SFPLOADI(p_sfpu::LREG7, SFPLOADI_MOD0_USHORT, 0xFF);

        // Copy A
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG3, 0);

        // Extract A = [a3:a2:a1:a0] where each is 8-bit.
        TTI_SFPAND(0, p_sfpu::LREG7, p_sfpu::LREG0, 0);               // LREG0 = a0 = A[7:0]
        TTI_SFPSHFT((-8) & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, 1);   // LREG2 = a1 = A[15:8]
        TTI_SFPAND(0, p_sfpu::LREG7, p_sfpu::LREG2, 0);               // clear other bits
        TTI_SFPSHFT((-16) & 0xfff, p_sfpu::LREG3, p_sfpu::LREG3, 1);  // LREG3 = a2 = A[23:16]
        TTI_SFPAND(0, p_sfpu::LREG7, p_sfpu::LREG3, 0);

        // Copy B
        TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG4, 0);
        TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG5, 0);

        // Extract B = [b3:b2:b1:b0] where each is 8-bit
        TTI_SFPAND(0, p_sfpu::LREG7, p_sfpu::LREG1, 0);              // LREG1 = b0 = B[7:0]
        TTI_SFPSHFT((-8) & 0xfff, p_sfpu::LREG4, p_sfpu::LREG4, 1);  // LREG5 = b1 = B[15:8]
        TTI_SFPAND(0, p_sfpu::LREG7, p_sfpu::LREG4, 0);
        TTI_SFPSHFT((-16) & 0xfff, p_sfpu::LREG5, p_sfpu::LREG5, 1);  // LREG6 = b2 = B[23:16]
        TTI_SFPAND(0, p_sfpu::LREG7, p_sfpu::LREG5, 0);
        // a3 and b3 will be extracted on the go due to limited registers.

        // Cast all 8-bit values to FP32
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);
        TTI_SFPCAST(p_sfpu::LREG2, p_sfpu::LREG2, 0);
        TTI_SFPCAST(p_sfpu::LREG3, p_sfpu::LREG3, 0);
        TTI_SFPCAST(p_sfpu::LREG4, p_sfpu::LREG4, 0);
        TTI_SFPCAST(p_sfpu::LREG5, p_sfpu::LREG5, 0);

        // a0*b0 (bits 0-15)
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG7, 0);
        TTI_SFPNOP;
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG7, p_sfpu::LREG7, 6);  // fp32 -> uint16

        // a0*b1 (bits 8-23)
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);
        TTI_SFPNOP;
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG6, p_sfpu::LREG6, 6);
        TTI_SFPSHFT(8, p_sfpu::LREG6, p_sfpu::LREG6, 1);                     // Shift left by 8
        TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG7, SFPIADD_MOD1_CC_NONE);  // Accumulate in LREG7

        // a0*b2 (bits 16-31)
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);
        TTI_SFPNOP;
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG6, p_sfpu::LREG6, 6);
        TTI_SFPSHFT(16, p_sfpu::LREG6, p_sfpu::LREG6, 1);                    // Shift left by 16
        TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG7, SFPIADD_MOD1_CC_NONE);

        // a1*b0 (bits 8-23)
        TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);
        TTI_SFPNOP;
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG6, p_sfpu::LREG6, 6);
        TTI_SFPSHFT(8, p_sfpu::LREG6, p_sfpu::LREG6, 1);                     // Shift left by 8
        TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG7, SFPIADD_MOD1_CC_NONE);

        // a1*b1 (bits 16-31)
        TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);
        TTI_SFPNOP;
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG6, p_sfpu::LREG6, 6);
        TTI_SFPSHFT(16, p_sfpu::LREG6, p_sfpu::LREG6, 1);                    // Shift left by 16
        TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG7, SFPIADD_MOD1_CC_NONE);

        // a2*b0 (bits 16-31)
        TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);
        TTI_SFPNOP;
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG6, p_sfpu::LREG6, 6);
        TTI_SFPSHFT(16, p_sfpu::LREG6, p_sfpu::LREG6, 1);                    // Shift left by 16
        TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG7, SFPIADD_MOD1_CC_NONE);

        TTI_SFPLOADI(p_sfpu::LREG6, SFPLOADI_MOD0_USHORT, 0x00FF);  // load 00FF to LREG6

        // a1*b2 --> goes beyond 32-bits [24:39]. We need to extract the bits upto 32.
        TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG5, p_sfpu::LCONST_0, p_sfpu::LREG5, 0);  // store result to LREG5
        TTI_SFPNOP;
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG5, p_sfpu::LREG5, 6);
        TTI_SFPAND(0, p_sfpu::LREG6, p_sfpu::LREG5, 0);                      // zero out high overflow bits
        TTI_SFPSHFT(24, p_sfpu::LREG5, p_sfpu::LREG5, 1);                    // Shift left by 24
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG7, SFPIADD_MOD1_CC_NONE);

        // a2*b1 --> goes beyond 32-bits [24:39]. We need to extract the bits upto 32.
        TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG5, 0);
        TTI_SFPNOP;
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG5, p_sfpu::LREG5, 6);
        TTI_SFPAND(0, p_sfpu::LREG6, p_sfpu::LREG5, 0);                      // zero out high overflow bits
        TTI_SFPSHFT(24, p_sfpu::LREG5, p_sfpu::LREG5, 1);                    // Shift left by 24
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG7, SFPIADD_MOD1_CC_NONE);

        // Load operands again to extract a3 and b3
        // operand A - int32
        TTI_SFPLOAD(p_sfpu::LREG2, INT32, ADDR_MOD_3, 0);
        // operand B - int32
        TT_SFPLOAD(p_sfpu::LREG3, INT32, ADDR_MOD_3, dst_offset * dst_tile_size);
        // mask
        TTI_SFPLOADI(p_sfpu::LREG4, SFPLOADI_MOD0_USHORT, 0xFF);

        // Extract A3
        TTI_SFPSHFT((-24) & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, 1);  // LREG2 = a3 = A[31:24]
        TTI_SFPAND(0, p_sfpu::LREG4, p_sfpu::LREG2, 0);

        // Extract B3
        TTI_SFPSHFT((-24) & 0xfff, p_sfpu::LREG3, p_sfpu::LREG3, 1);  // LREG3 = b3 = B[31:24]
        TTI_SFPAND(0, p_sfpu::LREG4, p_sfpu::LREG3, 0);

        // Cast to FP32
        TTI_SFPCAST(p_sfpu::LREG2, p_sfpu::LREG2, 0);
        TTI_SFPCAST(p_sfpu::LREG3, p_sfpu::LREG3, 0);

        // a0*b3 --> goes beyond 32-bits [24:39]. We need to extract the bits upto 32.
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG5, 0);
        TTI_SFPNOP;
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG5, p_sfpu::LREG5, 6);
        TTI_SFPAND(0, p_sfpu::LREG6, p_sfpu::LREG5, 0);    // zero out high overflow bits
        TTI_SFPSHFT(24, p_sfpu::LREG5, p_sfpu::LREG5, 1);  // Shift left by 24
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG7, SFPIADD_MOD1_CC_NONE);

        // a3*b0 --> goes beyond 32-bits [24:39]. We need to extract the bits upto 32.
        TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG5, 0);
        TTI_SFPNOP;
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG5, p_sfpu::LREG5, 6);
        TTI_SFPAND(0, p_sfpu::LREG6, p_sfpu::LREG5, 0);    // zero out high overflow bits
        TTI_SFPSHFT(24, p_sfpu::LREG5, p_sfpu::LREG5, 1);  // Shift left by 24
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG7, SFPIADD_MOD1_CC_NONE);

        TTI_SFPSTORE(p_sfpu::LREG7, INT32, ADDR_MOD_3, 0);
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
