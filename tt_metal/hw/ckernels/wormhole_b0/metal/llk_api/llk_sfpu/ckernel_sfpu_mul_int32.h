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
        //   top = a0*b2 + a1*b1 + a2*b0 (maximum 23 bits)
        //   mid = a0*b1 + a1*b0         (maximum 23 bits)
        //   low = a0*b0                 (maximum 22 bits)
        //
        // We cannot use SFPSTOCHRND to convert FP32 to INT32, as the values
        // are larger than 16 bits; instead we use the trick:
        //   fp32_to_u23(x) = mantissa_bits(x + 2**23)
        // This is exact for 23-bit integers.

        // a0
        TT_SFPLOAD(p_sfpu::LREG0, INT32, ADDR_MOD_3, dst_index_in0 * dst_tile_size);  // A with 32 bits
        // a1
        TTI_SFPSHFT2(p_sfpu::LREG0, p_sfpu::LREG13, p_sfpu::LREG2, 5);  // A without last 11 bits
        // a2
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG13, p_sfpu::LREG4, 5);  // A without last 22 bits
        /*
        the 5 passed as the inst_mode means
        The value in the LREG specified by lreg_dest is shifted by the
        number of bits determined by the value in the LREG specified by
        lreg_src_c. When the sign bit of the shift amount is 1, the value
        will be shifted to the right. When the sign bit of the shift
        amount is 0, the value is shifted to the left. The shift is a
        logical shift, where the value filled will be zero.
        */

        // a1 = (a1 & 0x7ff) as fp32
        /* Purpose of  (a1 & 0x7ff)
         Mask 0x7ff preserves the last 11-bits and sets the remaining bits of an input to zero
         when performed Bitwise AND & operation with any input */
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG2, 0);  // A1 with mid 11 bits
        TTI_SFPCAST(p_sfpu::LREG2, p_sfpu::LREG2, 0);
        // int32 to fp32

        // a2 = a2 as fp32
        TTI_SFPCAST(p_sfpu::LREG4, p_sfpu::LREG4, 0);
        // int32 to fp32 conversion for a2 as well

        // a0 = (a0 & 0x7ff) as fp32
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);  // A0 with low 11 bits
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);
        // int32 to fp32
        /*
        instr_mod1 (4 bits):
        0: the original value is interpreted as an int32 value and is converted
        to an fp32 value using round to nearest even
        1: the original value is interpreted as an int32 value and is converted
        to an fp32 value using stochastic rounding
*/

        // b0
        TT_SFPLOAD(p_sfpu::LREG1, INT32, ADDR_MOD_3, dst_index_in1 * dst_tile_size);
        // b1
        TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG13, p_sfpu::LREG3, 5);
        // b2
        TTI_SFPSHFT2(p_sfpu::LREG3, p_sfpu::LREG13, p_sfpu::LREG5, 5);

        // b2 = b2 as fp32
        TTI_SFPCAST(p_sfpu::LREG5, p_sfpu::LREG5, 0);

        // top = a0*b2 + 2**23
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LREG14, p_sfpu::LREG5, 0);

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

        // mid = a0*b1 + 2**23
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG6, 0);

        // low = a0*b0 + 2**23
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG14, p_sfpu::LREG0, 0);

        // mid += a1*b0
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG6, p_sfpu::LREG6, 0);

        // extract integers from mantissas
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPEXMAN_MOD1_PAD9);  // low
        TTI_SFPEXMAN(0, p_sfpu::LREG6, p_sfpu::LREG6, sfpi::SFPEXMAN_MOD1_PAD9);  // mid
        TTI_SFPEXMAN(0, p_sfpu::LREG5, p_sfpu::LREG5, sfpi::SFPEXMAN_MOD1_PAD9);  // top
        /*
        SFPEXMAN :
        Extracts the 10-bit ?! mantissa field from the LREG specified by lreg_c and writes the result into
        the LREG specified by lreg_dest.
        When the instr_mod1[0] is 1, the mantissa is extracted without modification and the result is
        padded with 9 zeroes in the MSBs. When the instr_mod1[0] is 0, the extracted mantissa is
        extended to include the hidden bit and then padded with 8 zeroes in the MSBs.
        We used mode 1, so 23 bits of mantissa + 9 padded zeros = 32 bit ??
        */

        TTI_SFPSHFT(22, 0, p_sfpu::LREG5, 1);  // top <<= 22
        TTI_SFPSHFT(11, 0, p_sfpu::LREG6, 1);  // mid <<= 11

        TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);

        TT_SFPSTORE(p_sfpu::LREG0, INT32, ADDR_MOD_2, dst_index_out * dst_tile_size);
    }
}

template <bool APPROXIMATION_MODE>
inline void mul_int32_init() {
    sfpi::vConstIntPrgm0 = 0x7ff;  // lreg  12? mask that extracts only last 11 bits of the input it is &-ed with
    sfpi::vConstIntPrgm1 =
        -11;  // lreg 13? no of bits to shift; -ve sign indicates right shift with SFPSHFT2 instr_mode 5
    sfpi::vConstFloatPrgm2 =
        8388608.0f;  // lreg14 ? 2**23 if this mask is added to the result in floating point multiplication, the integer
                     // bits will be pushed to occupy the mantissa bits ?
}

}  // namespace ckernel::sfpu
