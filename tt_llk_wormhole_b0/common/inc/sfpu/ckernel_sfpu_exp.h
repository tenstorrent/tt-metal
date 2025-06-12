// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include "sfpi.h"
#include "sfpi_fp16.h"

namespace ckernel
{
namespace sfpu
{

sfpi_inline sfpi::vFloat _sfpu_exp_(sfpi::vFloat val)
{
    // If exponent is > -1 extract it and replace with -1
    sfpi::vInt exp = exexp(val);
    v_if (exp >= 0)
    {
        val = setexp(val, 126);
    }
    v_endif;

    // Run series in Horner form
    sfpi::vFloat tmp = val * sfpi::vConst0p8373 + sfpi::s2vFloat16b(0.863281);
    val              = val * tmp + sfpi::vConst1;

    v_if (exp >= 0)
    {
        val = val * val;
        for (int s_iter = 0; s_iter < 7; s_iter++)
        {
            exp = exp - 1;
            // Narrow predication on each loop
            v_and(exp >= 0);
            val = val * val;
        }
    }
    v_endif;

    return val;
}

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _calculate_exponential_body_(sfpi::vFloat in)
{
    sfpi::vFloat out;

    if constexpr (APPROXIMATION_MODE)
    {
        constexpr int FRAC_BITS = 3;
        constexpr uint SP_BIAS  = 127 << FRAC_BITS;

        // * by 1/ln2 and add convert to 7.3 FxP format
        sfpi::vFloat vConstLn2Recip = sfpi::vConstFloatPrgm0;
        sfpi::vFloat conv           = in * vConstLn2Recip;

        // Clear exp bits
        sfpi::vInt c23_73 = p_exp::C23_73;
        sfpi::vInt tmp    = sfpi::reinterpret<sfpi::vInt>(conv) - c23_73;

        // Add bias
        tmp += SP_BIAS;

        // SHL to move integer bits to exponent
        out = sfpi::reinterpret<sfpi::vFloat>(tmp << (10 - FRAC_BITS));
    }
    else
    {
        // Force sign to 0 (make number positive)
        out = _sfpu_exp_(sfpi::setsgn(in, 0));

        v_if (in < 0)
        {
            out = _sfpu_reciprocal_(out);
        }
        v_endif;
    }

    return out;
}

template <bool APPROXIMATION_MODE, bool SCALE_EN, int ITERATIONS, bool FAST_APPROX, bool SKIP_POSITIVE_CHECK = false>
void _calculate_exponential_(const int iterations, uint16_t exp_base_scale_factor = 0x3F80 /* 1.0f in BF16 */)
{
    if constexpr (FAST_APPROX && APPROXIMATION_MODE)
    {
        // Sanitize the input values by loading from DEST, comparing against the value -88.5, and if the input value is more negative than that, swap the input
        // value with -88.5 and store back to DEST
        //  - in other words, after the sanitize step, the values in DEST will be in the range {-88.5 , +inf}

        // Macro Sequence Register 1 configured to read back in the original values from dest, sanitize them to a range we can handle, and then store them back
        // to dest
        //  LD     : bring in the original value from DEST (y)
        //  MAD    : unused
        //  ROUND  : unused
        //  SIMPLE : SWAP the larger value of y and -88.5 into the LREG
        //  STORE  : store the sanitized value back to dest
        TTI_SFPLOADMACRO(
            4,
            0,
            3,
            0);     // MACRO Sequence Register 1: LD, SWAP, STORE - uses LREG[0] for loaded value - Dest offset  0 is targeting the even columns for rows   3: 0
        TTI_SFPNOP; // NOP is necessary because the SWAP operation takes 2 cycles and unfortunately is not pipelined
        TTI_SFPLOADMACRO(
            5,
            0,
            3,
            2); // MACRO Sequence Register 1: LD, SWAP, STORE - uses LREG[1] for loaded value - Dest offset  2 is targeting the odd  columns for rows   3: 0
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(
            6,
            0,
            3,
            4); // MACRO Sequence Register 1: LD, SWAP, STORE - uses LREG[2] for loaded value - Dest offset  4 is targeting the even columns for rows   7: 4
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(
            7,
            0,
            3,
            6); // MACRO Sequence Register 1: LD, SWAP, STORE - uses LREG[3] for loaded value - Dest offset  6 is targeting the odd  columns for rows   7: 4
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(
            4,
            0,
            3,
            8); // MACRO Sequence Register 1: LD, SWAP, STORE - uses LREG[0] for loaded value - Dest offset  8 is targeting the even columns for rows  11: 8
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(
            5,
            0,
            3,
            10); // MACRO Sequence Register 1: LD, SWAP, STORE - uses LREG[1] for loaded value - Dest offset 10 is targeting the even columns for rows  11: 8
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(
            6,
            0,
            3,
            12); // MACRO Sequence Register 1: LD, SWAP, STORE - uses LREG[2] for loaded value - Dest offset 12 is targeting the odd  columns for rows  15:12
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(
            7,
            0,
            3,
            14); // MACRO Sequence Register 1: LD, SWAP, STORE - uses LREG[3] for loaded value - Dest offset 14 is targeting the even columns for rows  15:12
        // NOP not needed in this spot because the next LoadMacro is a computational macro which doesn't immediately use the SIMPLE unit

        // Macro Sequence Register 0 configured to read back in the sanitized values and calculate the approximate exponential value
        //  LD     : the sanitized value from DEST (y)
        //  MAD    : compute (A * y) + (B-C)  , where A = (2^8)/ln(2) , B = 127 * (2^8) , C = Adjustment parameter of roughly 11.2 to minimize error
        //  ROUND  : convert the MAD result from FP32 to a 16-bit unsigned integer using stochastic rounding
        //  SIMPLE : shift the 16-bit integer to the left by 15 bits to place the MSB of the computed value into the MSB of the exponent bits of the fp32 format
        //  STORE  : store the shifted value back to dest
        TTI_SFPLOADMACRO(0, 0, 3, 0);  // MACRO Sequence Register 0: LD, MAD, ROUND, SHIFT and STORE - uses LREG[0] for loading and intermediate results - Dest
                                       // offset  0 is targeting the even columns for rows   3: 0
        TTI_SFPLOADMACRO(1, 0, 3, 2);  // MACRO Sequence Register 0: LD, MAD, ROUND, SHIFT and STORE - uses LREG[1] for loading and intermediate results - Dest
                                       // offset  2 is targeting the odd  columns for rows   3: 0
        TTI_SFPLOADMACRO(2, 0, 3, 4);  // MACRO Sequence Register 0: LD, MAD, ROUND, SHIFT and STORE - uses LREG[2] for loading and intermediate results - Dest
                                       // offset  4 is targeting the even columns for rows   7: 4
        TTI_SFPLOADMACRO(3, 0, 3, 6);  // MACRO Sequence Register 0: LD, MAD, ROUND, SHIFT and STORE - uses LREG[3] for loading and intermediate results - Dest
                                       // offset  6 is targeting the odd  columns for rows   7: 4
        TTI_SFPLOADMACRO(0, 0, 3, 8);  // MACRO Sequence Register 0: LD, MAD, ROUND, SHIFT and STORE - uses LREG[0] for loading and intermediate results - Dest
                                       // offset  8 is targeting the even columns for rows  11: 8
        TTI_SFPLOADMACRO(1, 0, 3, 10); // MACRO Sequence Register 0: LD, MAD, ROUND, SHIFT and STORE - uses LREG[1] for loading and intermediate results - Dest
                                       // offset 10 is targeting the even columns for rows  11: 8
        TTI_SFPLOADMACRO(2, 0, 3, 12); // MACRO Sequence Register 0: LD, MAD, ROUND, SHIFT and STORE - uses LREG[2] for loading and intermediate results - Dest
                                       // offset 12 is targeting the odd  columns for rows  15:12
        TTI_SFPLOADMACRO(3, 0, 3, 14); // MACRO Sequence Register 0: LD, MAD, ROUND, SHIFT and STORE - uses LREG[3] for loading and intermediate results - Dest
                                       // offset 14 is targeting the even columns for rows  15:12
        // NOP needed to allow time for the final Computation Loadmacro to complete before returning to the Sanitation Loadmacro at the top for the next
        // iteration
        //  - to be completely safe, use 3 NOP; in practice 1 seems to be enough, probably because the overhead of the DEST INCRW stuff introduces 2 cycles of
        //  delay
        TTI_SFPNOP;
        // TTI_SFPNOP;
        // TTI_SFPNOP;
    }
    else
    {
        // Unroll 8 best for approx, unroll 0 for precise, compiler figures this out
        for (int d = 0; d < iterations; d++)
        {
            sfpi::vFloat val = sfpi::dst_reg[0];
            if constexpr (SCALE_EN)
            {
                val = val * sfpi::s2vFloat16b(exp_base_scale_factor);
            }
            if constexpr (APPROXIMATION_MODE)
            {
                if constexpr (!SKIP_POSITIVE_CHECK)
                {
                    v_if (val >= 89)
                    {
                        // Algorithm is incorrect for inputs >= 89, so saturate output to infinity.
                        sfpi::vFloat val_inf = std::numeric_limits<float>::infinity();
                        sfpi::dst_reg[0]     = val_inf;
                    }
                    v_elseif (val < -42)
                    {
                        // Algorithm is incorrect for inputs < -42, so saturate output to 0.
                        sfpi::dst_reg[0] = 0.0f;
                    }
                    v_else
                    {
                        // * by 1/ln2 and add convert to 7.3 FxP format
                        sfpi::vFloat vConstLn2Recip = sfpi::vConstFloatPrgm0;
                        sfpi::vFloat c23_73         = sfpi::vConstFloatPrgm1;
                        sfpi::vInt adj_exp          = sfpi::vConstIntPrgm2;
                        val                         = val * vConstLn2Recip + c23_73;

                        // Remove Exponent of 7 and bias the Mantissa to 127.
                        sfpi::vInt val_short = adj_exp + sfpi::reinterpret<sfpi::vInt>(val);

                        // SHL to move integer bits to exponent
                        val_short <<= 10 - p_exp::FRAC_BITS;
                        sfpi::dst_reg[0] = sfpi::reinterpret<sfpi::vFloat>(val_short);
                    }
                    v_endif;
                }
                else
                {
                    // SKIP_POSITIVE_CHECK is true, so user is responsible for ensuring inputs are <= 89.
                    v_if (val < -42)
                    {
                        sfpi::dst_reg[0] = 0.0f;
                    }
                    v_else
                    {
                        // * by 1/ln2 and add convert to 7.3 FxP format
                        sfpi::vFloat vConstLn2Recip = sfpi::vConstFloatPrgm0;
                        sfpi::vFloat c23_73         = sfpi::vConstFloatPrgm1;
                        sfpi::vInt adj_exp          = sfpi::vConstIntPrgm2;
                        val                         = val * vConstLn2Recip + c23_73;

                        // Remove Exponent of 7 and bias the Mantissa to 127.
                        sfpi::vInt val_short = adj_exp + sfpi::reinterpret<sfpi::vInt>(val);

                        // SHL to move integer bits to exponent
                        val_short <<= 10 - p_exp::FRAC_BITS;
                        sfpi::dst_reg[0] = sfpi::reinterpret<sfpi::vFloat>(val_short);
                    }
                    v_endif;
                }
            }
            else
            {
                sfpi::vFloat result = _sfpu_exp_(sfpi::setsgn(val, 0));

                v_if (val < 0)
                {
                    result = _sfpu_reciprocal_(result);
                }
                v_endif;

                sfpi::dst_reg[0] = result;
            }

            sfpi::dst_reg++;
        }
    }
}

constexpr auto bits = [](float x) constexpr { return __builtin_bit_cast(std::uint32_t, x); };
constexpr auto lo16 = [](float x) constexpr { return static_cast<std::uint16_t>(bits(x) & 0xFFFFu); };
constexpr auto hi16 = [](float x) constexpr { return static_cast<std::uint16_t>(bits(x) >> 16); };

template <bool APPROXIMATION_MODE, bool FAST_APPROX, uint32_t scale = 0x3F800000 /* 1.0f in FP32 */>
inline void _init_exponential_()
{
    if constexpr (FAST_APPROX && APPROXIMATION_MODE)
    {
        // Algorithm is adapted from:
        //      A Fast, Compact Approximation of the Exponential Function
        //      Nicol N. Schraudolph
        //      IDSIA, Lugano, Switzerland

        // First, set up constant values which are needed for the computation
        //      We will first sanitize the input values (y) to be in the range that won't cause underflow, which for our hardware means we need to limit
        //      negative values to be greater than or equal to -88.5 The computation that is needed is (A * y) + (B - C) , where A = (2^8)/ln(2) , B = 127 *
        //      (2^8) , C = Adjustment parameter of roughly 11.2 to minimize error
        //          - NOTE: we would like to be able to use 2^23 instead of 2^8 and compute a 32-bit quantity, but our hardware only supports rounding FP32 into
        //          a 16-bit integer, so we use 2^8 and then shift left by 15 bits after rounding
        //      So we will set up the following constants:
        //          LREG[14] =       =    -88.5               = 0xc2b10000
        //          LREG[12] = A     =    369.329925537109375 = 0x43b8aa3b
        //          LREG[13] = (B-C) =  32500.818359375       = 0x46fde9a3

        constexpr float LN2_RECIP = 1.4426950408889634f;
        constexpr float A         = 256.0f * LN2_RECIP;
        constexpr float B_minus_C = 32500.818359375f;
        constexpr float THRESHOLD = -88.5f;

        constexpr float scale_fp32 = __builtin_bit_cast(float, scale);

        constexpr float A_scaled         = A * scale_fp32;
        constexpr float THRESHOLD_scaled = THRESHOLD / scale_fp32;

        TTI_SFPLOADI(0, 0xA, lo16(THRESHOLD_scaled));
        TTI_SFPLOADI(0, 0x8, hi16(THRESHOLD_scaled));
        TTI_SFPCONFIG(0, 14, 0); // SFPCONFIG Dest 14 = LREG[14] =            -88.5               = 0xc2b10000

        TTI_SFPLOADI(0, 0xA, lo16(A_scaled));
        TTI_SFPLOADI(0, 0x8, hi16(A_scaled));
        TTI_SFPCONFIG(0, 12, 0); // SFPCONFIG Dest 12 = LREG[12] = A     =    369.329925537109375 = 0x43b8aa3b

        TTI_SFPLOADI(0, 0xA, lo16(B_minus_C));
        TTI_SFPLOADI(0, 0x8, hi16(B_minus_C));
        TTI_SFPCONFIG(0, 13, 0); // SFPCONFIG Dest 13 = LREG[13] = (B-C) =  32500.818359375       = 0x46fde9a3

        // Next, set up the macro instructions which will be necessary
        //  - for the sanitize function: we will need a SWAP instruction
        //  - for the main computation function: we will need MAD, ROUND, and SHIFT instructions

        // There are two ways to program the macro instruction registers, and this setup leverages both ways
        //  - we can either use the SFPCONFIG flow, by setting up the bits of the instruction into LREG[0] and then targeting the Macro instruction register
        //  - or we can use the shortcut / backdoor load method which relies on having some illegal destination register values as part of the instruction

        // Use SFPCONFIG method for the SWAP instruction, since we want the SWAP itself to use a destination register which is not normally a legal value
        //      (we are cheating a bit here, since we only care about one half of the swap and we want to use a constant for the other half)
        //
        //              imm12 = 0,       lreg_src_c = 0 (will be fed by value loaded from Dest into Loadmacro lreg_dest),  lreg_dest = LREG[14] = - 88.5,
        //              instr_mod1 = 1 swap the values with the larger of the two ending up in lreg_dest -> but we will use the Loadmacro lreg_dest register as
        //              output
        // TTI_SFP_SWAP(0,               0,                                                                                14,                            1);
        TTI_SFPLOADI(0, 0xA, 0x00E1);
        TTI_SFPLOADI(0, 0x8, 0x9200);
        TTI_SFPCONFIG(0, 0, 0); // SFPCONFIG Dest 0 = Programmable Macro instruction 0: TTI_SFPSWAP(0, 0, 14, 1); // compare against LREG[14] (-88.5), and put
                                // the larger value into LREG[loadmacro_lreg_dest]
        TTI_SFPNOP;

        // Backdoor load of Macro Instruction 1
        // Dummy version of MAD instruction with lreg_dest = 4'b11_01 = 13 to install into Programmable Macro instruction register 1, which is Macro Instruction
        // Register 5
        TTI_SFPMAD(12, 0, 13, 13, 0); // MACRO Instruction 1 <--- lreg X = lreg[12] (A) * lreg[0] (y) + lreg[13] (B-C)

        // Backdoor load of Macro Instruction 2
        // ROUND instruction to convert FP32 result into an integer value (int16)
        //                Stochastic = 0,  Imm(Descale),  SrcB(unused),   SrcC(input value),  Lreg_dest = 14 to install in Programmable Macro Instruction reg
        //                2'b10,  instr_mod1 = 14 to treat input as fp32, output as unsigned int16, use imm as descale
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 14, 14); // Round to unsigned Int16

        // Backdoor load of Macro Instruction 3
        // If using the unsigned int rounding mode, then shift by 15; SHL to move integer bits to exponent;
        TTI_SFPSHFT(15, 0, 15, 1); // imm = 15 to shift left by 15 bits; lreg_c = 0 (will use macro reg); lreg_dest = 15 to install in Programmable Macro
                                   // Instruction reg 2'b11, which is Macro Instruction Register 7

        // So at this point, we have the following instructions loaded into our macro registers:
        //
        // 00: (no macro instruction, just execute whatever is issued from Tensix) <-- these are fixed / not programmable
        // 01: ( Rsvd                                                            ) <-- these are fixed / not programmable
        // 02: ( NOP                                                             ) <-- these are fixed / not programmable
        // 03: ( SFPSTORE                                                        ) <-- these are fixed / not programmable
        // 04: TTI_SFPSWAP       (0, 0, 11, 1)
        // 05: TTI_SFPMAD        (12, 0, 13, 13, 0)
        // 06: TTI_SFP_STOCH_RND (1, 0, 0, 0, 14, 14)
        // 07: TTI_SFPSHFT       (15,0,15,1)

        // Now we want to set up our two sequences

        // Sequence 1 setup: we want to Load, SWAP, <delay>, Store
        //       Delay slot:                  0     1        2
        //                                                                                                                                                                                                 Use
        //                                                                                                                                                                                                 Loaded  Result          Macro
        //                                                                                                                                                                                                 Value   Value   Delay   Instruction
        //                                                                                                                                                                                                 SRCB    Stage   Slot    Select
        TTI_SFPLOADI(0, 0xA, 0x0004); // slot1 : SIMPLE UNIT, want SWAP  instruction which is in macro instruction mux[4], delayed by 0 ; not using staging flop
                                      // as dest; not using load reg as srcb : 8'b0_______0_______000_____100          = 0x04 slot2 : MAD    UNIT, unused :
                                      // 8'b0_______0_______000_____000          = 0x00
        TTI_SFPLOADI(0, 0x8, 0x1300); // slot3 : ROUND  UNIT, unused : 8'b0_______0_______000_____000          = 0x00 slot4 : STORE  UNIT, want STORE
                                      // instruction which is in macro instruction mux[3], delayed by 2 ; not using staging flop as src ; :
                                      // 8'b0_______0_______010_____011          = 0x13
        TTI_SFPCONFIG(0, 5, 0);       // SFPCONFIG Dest 5 = Macro Sequence Register 1

        // Sequence 0 setup: we want to Load, MAD, <delay>, ROUND, SHIFT, Store
        //       Delay slot:                  0    1        2      3      4
        //                                                                                                                                                                                                 Use
        //                                                                                                                                                                                                 Loaded  Result          Macro
        //                                                                                                                                                                                                 Value   Value   Delay   Instruction
        //                                                                                                                                                                                                 SRCB    Stage   Slot    Select
        TTI_SFPLOADI(
            0,
            0xA,
            0x85DF); // slot1 : SIMPLE UNIT, want SHIFT instruction which is in macro instruction mux[7], delayed by 3 ;     using staging flop as dest; using
                     // load reg as srcb : 8'b1_______1_______011_____111          = 0xDF slot2 : MAD    UNIT, want MAD   instruction which is in macro
                     // instruction mux[5], delayed by 0 ; not using staging flop as dest;     using load reg as srcb : 8'b1_______0_______000_____101 = 0x85
        TTI_SFPLOADI(
            0,
            0x8,
            0x6316); // slot3 : ROUND  UNIT, want ROUND instruction which is in macro instruction mux[6], delayed by 2 ; not using staging flop as dest; using
                     // : 8'b0_______0_______010_____110          = 0x16 slot4 : STORE  UNIT, want STORE instruction which is in macro instruction mux[3],
                     // delayed by 4 ;     using staging flop as src ;     using                  : 8'b0_______1_______100_____011          = 0x63
        TTI_SFPCONFIG(0, 4, 0); // Load it into macro sequence register 0 (destination = 4)

        // Reset LoadMacroConfig[Lane].Misc for all lanes, in case it has been previously set by another use of macros.
        TTI_SFPCONFIG(0, 8, 1);
    }
    else if constexpr (APPROXIMATION_MODE)
    {
        sfpi::vConstFloatPrgm0 = 1.442695f; // ln2_recip
        sfpi::vConstFloatPrgm1 = sfpi::s2vFloat16b(p_exp::C23_73);
        sfpi::vConstFloatPrgm2 = sfpi::s2vFloat16b(p_exp::ADJ_EXP);
    }
    else
    {
        sfpi::vConstFloatPrgm0 = 1.442695f; // ln2_recip
        sfpi::vConstFloatPrgm1 = 2.0f;
        sfpi::vConstFloatPrgm2 = 0.863281f;
    }
}

} // namespace sfpu
} // namespace ckernel
