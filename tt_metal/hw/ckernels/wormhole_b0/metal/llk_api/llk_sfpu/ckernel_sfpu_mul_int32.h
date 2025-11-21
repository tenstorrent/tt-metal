// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void mul_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size = 64;

        // Split the 32-bit input values into 11-bit chunks:
        //
        //   a = (a2 << 22) | (a1 << 11) | a0
        //   b = (b2 << 22) | (b1 << 11) | b0
        //
        // This allows us to cast these values to fp32 without loss of
        // precision, and furthermore, we can compute:
        //
        //   a * b = (top << 22) + (mid << 11) + low
        //
        // Where:
        //
        //   top = a0*b2 + a1*b1 + a2*b0 (maximum 24 bits)
        //   mid = a0*b1 + a1*b0         (maximum 23 bits)
        //   low = a0*b0                 (maximum 22 bits)
        //
        // For top and mid, we use FMA to sum values without loss of precision.
        // We cannot use SFPSTOCHRND to convert FP32 to INT32, as the values
        // are larger than 16 bits, so we extract the exponent and mantissa,
        // and shift the mantissa by the appropriate amount.

        // a0
        TT_SFPLOAD(p_sfpu::LREG0, INT32, ADDR_MOD_3, dst_index_in0 * dst_tile_size);
        // a1
        TTI_SFPSHFT2(p_sfpu::LREG0, p_sfpu::LREG13, p_sfpu::LREG2, 5);
        // a2
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG13, p_sfpu::LREG4, 5);

        // a1 = (a1 & 0x7ff) as fp32
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG2, 0);
        TTI_SFPCAST(p_sfpu::LREG2, p_sfpu::LREG2, 0);

        // a2 = a2 as fp32
        TTI_SFPCAST(p_sfpu::LREG4, p_sfpu::LREG4, 0);

        // a0 = (a0 & 0x7ff) as fp32
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);

        // b0
        TT_SFPLOAD(p_sfpu::LREG1, INT32, ADDR_MOD_3, dst_index_in1 * dst_tile_size);
        // b1
        TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG13, p_sfpu::LREG3, 5);
        // b2
        TTI_SFPSHFT2(p_sfpu::LREG3, p_sfpu::LREG13, p_sfpu::LREG5, 5);

        // b2 = b2 as fp32
        TTI_SFPCAST(p_sfpu::LREG5, p_sfpu::LREG5, 0);

        // top = a0*b2
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LCONST_0, p_sfpu::LREG5, 0);

        // b1 = (b1 & 0x7ff) as fp32
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG3, 0);
        TTI_SFPCAST(p_sfpu::LREG3, p_sfpu::LREG3, 0);

        // top += a1*b1
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LREG5, p_sfpu::LREG5, 0);

        // b0 = (b0 & 0x7ff) as fp32
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);

        // top += a2*b0
        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG1, p_sfpu::LREG5, p_sfpu::LREG5, 0);

        // mid = a0*b1
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);

        // low = a0*b0
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG7, 0);

        // mid += a1*b0
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG6, p_sfpu::LREG6, 0);

        // convert low to integer, taking care with zero, since SFPEXMAN always sets the implicit bit.
        TTI_SFPEXEXP(0, p_sfpu::LREG7, p_sfpu::LREG0, 2 | 8);
        TTI_SFPEXMAN(0, p_sfpu::LREG7, p_sfpu::LREG7, 0);
        TTI_SFPIADD(-23 & 0xfff, p_sfpu::LREG0, p_sfpu::LREG0, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_IMM);
        TTI_SFPSHFT2(p_sfpu::LREG7, p_sfpu::LREG0, p_sfpu::LREG7, 5);
        TTI_SFPENCC(0, 0, 0, 0);

        // convert mid to integer, taking care with zero, since SFPEXMAN always sets the implicit bit.
        TTI_SFPEXEXP(0, p_sfpu::LREG6, p_sfpu::LREG0, 2 | 8);
        TTI_SFPEXMAN(0, p_sfpu::LREG6, p_sfpu::LREG6, 0);
        TTI_SFPIADD((11 - 23) & 0xfff, p_sfpu::LREG0, p_sfpu::LREG0, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_IMM);
        TTI_SFPSHFT2(p_sfpu::LREG6, p_sfpu::LREG0, p_sfpu::LREG6, 5);
        TTI_SFPENCC(0, 0, 0, 0);

        // convert top to integer, taking care with zero, since SFPEXMAN always sets the implicit bit.
        TTI_SFPEXEXP(0, p_sfpu::LREG5, p_sfpu::LREG0, 2 | 8);
        TTI_SFPEXMAN(0, p_sfpu::LREG5, p_sfpu::LREG5, 0);
        TTI_SFPIADD((22 - 23) & 0xfff, p_sfpu::LREG0, p_sfpu::LREG0, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_IMM);
        TTI_SFPSHFT2(p_sfpu::LREG5, p_sfpu::LREG0, p_sfpu::LREG5, 5);
        TTI_SFPENCC(0, 0, 0, 0);

        // lo += mid
        TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG7, SFPIADD_MOD1_CC_NONE);

        // lo += top
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG7, SFPIADD_MOD1_CC_NONE);

        TT_SFPSTORE(p_sfpu::LREG7, INT32, ADDR_MOD_2, dst_index_out * dst_tile_size);
    }
}

template <bool APPROXIMATION_MODE>
inline void mul_int32_init() {
    sfpi::vConstIntPrgm0 = 0x7ff;
    sfpi::vConstIntPrgm1 = -11;
}

}  // namespace ckernel::sfpu
