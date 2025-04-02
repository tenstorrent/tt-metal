// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_ops.h"
#include "sfpi.h"
#include "sfpi_fp16.h"

namespace ckernel
{
namespace sfpu
{

// probability should be between 0 - INT_MAX (signed)
// scale should be binary representation of a float32
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_dropout_(const int iterations, uint probability, uint scale)
{
    // SFPU microcode

    TT_SFPLOADI(p_sfpu::LREG1, 10, scale & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG1, 8, scale >> 16);
    TT_SFPLOADI(p_sfpu::LREG2, 10, probability & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, 8, probability >> 16);
#pragma GCC unroll 0
    for (int d = 0; d < iterations; d++)
    {
        ////////////////////////
        // Scale samples
        // dst_reg[0] = dst_reg[0] * s2vFloat16b(scale);
        ///////////////////////
        TTI_SFPLOAD(p_sfpu::LREG0, 0, 3, 0);
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);

        ////////////////////////
        // Instruction SFPMOV generates a uint32_t pseudorandom number
        // when instr_mod1 = 8 and lreg_c =  9.
        // Arguments: (imm12_math, lreg_c, lreg_dest, instr_mod1)
        // Unset sign-bit for easy comparison with probability
        ////////////////////////
        TTI_SFPMOV(0, 9, p_sfpu::LREG3, 8);
        TTI_SFPSETSGN(0, p_sfpu::LREG3, p_sfpu::LREG3, 1);

        ////////////////////////
        // Drop samples
        // v_if (rand < probability)
        //   dst_reg[0] = vConst0;
        ///////////////////////
        TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG3, 10);
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSTORE(0, 0, 3, 0);

        sfpi::dst_reg++;
    }
}

inline void _init_dropout_(const uint seed)
{
    init_prng_seed(seed);
}

} // namespace sfpu
} // namespace ckernel
