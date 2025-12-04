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
        /*
         A number is split into chunks of 11 bits
            A = (a2 . 2^22)  + (a1 . 2^11) + a0 . 2^0
        */

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
        SFPSHFT2
        Shifts values between or within LREGs, either within an SFPU instance or globally across SFPU
        instances. The shift can be either at the granularity of the LREG, such that the entire value held
        in one LREG is shifted to an adjacent LREG in one instruction, or a traditional bitwise shift that
        happens within the LREG.
        The global shift variants which move values from one SFPU instance to another require 2 cycles
        to complete and should always be followed by an SFPNOP instruction.

        Arguments: (imm12_math, lreg_src_c, lreg_dest, instr_mod1) ?
        here the dest specified is only to store result , does not have any preloaded input

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

        instr_mod1 (4 bits):
        0: extracted mantissa is extended to include the hidden bit and padded
        with 8 zeroes in the MSBs
        1: extracted mantissa is padded with 9 zeroes in the MSBs
        #define SFPEXMAN_MOD1_PAD9 1

        */

        TTI_SFPSHFT(22, 0, p_sfpu::LREG5, 1);  // top <<= 22
        // first 22 bits are dropped, last 10 bits of tops are moved to the front and remaining 22 bits padded with 0
        TTI_SFPSHFT(11, 0, p_sfpu::LREG6, 1);  // mid <<= 11
        // first 11 bits are dropped, last 21 bits of mid are moved to the front and remaining 11 bits padded with 0

        /*
        SFPSHFT
        Shifts the value in the LREG specified by lreg_dest by an amount specified either in the
        immediate field or the LREG specified by lreg_c. The shift can be in either direction, left or right,
        depending upon the sign bit of the shift amount.
        When the sign bit of the shift amount is a 1, the value will be shifted to the right.
        When the sign bit of the shift amount is a 0, the value will be shifted to the left.
        In either case, the shift will be a logical shift, where the filled value will be zeroes.
        Arguments: (imm12_math, lreg_c, lreg_dest, instr_mod1)
        instr_mod1 (4 bits):
            0: lreg_c will specify the LREG which contains the shift amount
            1: imm12_math will contain the shift amount
            */

        TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);
        // L0 with last 11 bits of the result  + L6 with mid 11 bits of the result and last 11 bits padded to zero =
        // last 22 bits of the result
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);
        // L0 with 22 bits of the result  + L5 with top 10 bits of the result = 32 bits of the result
        /*
        SFPIADD
        Performs integer addition between the LREG specified by lreg_c and either the immediate field
        or the value in the LREG specified by lreg_dest and writes the result into the LREG specified by
        lreg_dest.
        The instruction optionally updates the condition code register as outlined in the instr_mod1
        settings.
        Arguments: (imm12_math, lreg_c, lreg_dest, instr_mod1)
        instr_mod1 (4 bits):
            0: performs integer addition between LREGs specified in
            lreg_c and lreg_dest
            condition code register is set to MSB of addition result
            1: performs integer addition between LREG specified in
            lreg_c and the sign extended imm12_math value
            condition code register is set to MSB of addition result
            2: performs integer addition between LREG specified in
            lreg_c and the 2’s complement of LREG specified in lreg_dest
            condition code register is set to MSB of addition result
            3: reserved
            we use this here  SFPIADD_MOD1_CC_NONE = 4:
            4: performs integer addition between LREGs specified in
            lreg_c and lreg_dest

            #define SFPIADD_MOD1_ARG_LREG_DST        0
            #define SFPIADD_MOD1_ARG_IMM             1
            #define SFPIADD_MOD1_ARG_2SCOMP_LREG_DST 2
            #define SFPIADD_MOD1_CC_LT0  0
            #define SFPIADD_MOD1_CC_NONE 4
            #define SFPIADD_MOD1_CC_GTE0 8
            condition code register is not modified
*/

        TT_SFPSTORE(p_sfpu::LREG0, INT32, ADDR_MOD_2, dst_index_out * dst_tile_size);
    }
}

template <bool APPROXIMATION_MODE>
inline void mul_int32_init() {
    sfpi::vConstIntPrgm0 = 0x7ff;  // lreg  12? mask that extracts only last 11 bits of the input it is &-ed with
    sfpi::vConstIntPrgm1 =
        -11;  // lreg 13? no of bits to shift; -ve sign indicates right shift with SFPSHFT2 instr_mode 5
    sfpi::vConstFloatPrgm2 = 8388608.0f;  // lreg14 ? 2**23 if this mask is added to the result in floating point
                                          // multiplication, the integer bits will be pushed to occupy the 23 mantissa
                                          // bits and extracted easily by dropping the exponent bits ?
}

}  // namespace ckernel::sfpu
