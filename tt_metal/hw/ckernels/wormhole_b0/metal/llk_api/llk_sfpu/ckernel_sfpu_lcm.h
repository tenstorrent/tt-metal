// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_gcd.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// inputs are lreg0, lreg1. output is lreg4
inline void calculate_sfpu_mul_u16_to_u32_body() {
    // mask
    TTI_SFPLOADI(p_sfpu::LREG7, SFPLOADI_MOD0_USHORT, 0xff);
    // copy
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
    TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG3, 0);
    // shift
    TTI_SFPSHFT((-8) & 0xfff, 0, p_sfpu::LREG2, 1);
    TTI_SFPSHFT((-8) & 0xfff, 0, p_sfpu::LREG3, 1);
    // mask lowest 8 bits
    TTI_SFPAND(0, p_sfpu::LREG7, p_sfpu::LREG0, 0);
    TTI_SFPAND(0, p_sfpu::LREG7, p_sfpu::LREG1, 0);
    // cast to fp32
    TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);
    TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);
    TTI_SFPCAST(p_sfpu::LREG2, p_sfpu::LREG2, 0);
    TTI_SFPCAST(p_sfpu::LREG3, p_sfpu::LREG3, 0);
    // multiply in fp32
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG4, 0);
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG5, 0);
    TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);
    TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG7, 0);
    // cast back
    TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG4, p_sfpu::LREG4, 6);
    TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG5, p_sfpu::LREG5, 6);
    TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG6, p_sfpu::LREG6, 6);
    TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG7, p_sfpu::LREG7, 6);
    // shift back
    TTI_SFPSHFT(8, 0, p_sfpu::LREG5, 1);
    TTI_SFPSHFT(8, 0, p_sfpu::LREG6, 1);
    TTI_SFPSHFT(16, 0, p_sfpu::LREG7, 1);
    TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG4, SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG4, SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG4, SFPIADD_MOD1_CC_NONE);
}

template <int ITERATIONS = 8>
inline void calculate_sfpu_lcm(const uint dst_offset)
{
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size = 64;

        TTI_SFPLOAD(p_sfpu::LREG0, 4, 3, 0); // a
        TT_SFPLOAD(p_sfpu::LREG1, 4, 3, dst_offset * dst_tile_size); // b

        // Binary GCD algorithm; assumes abs(a) < 2^15 and abs(b) < 2^15, hence gcd(a, b) < 2^15
        calculate_sfpu_gcd_body<15>();

        // Two iterations of Newton's method to find reciprocal of gcd(a, b)
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG2, 0);
        TTI_SFPSETSGN(1, p_sfpu::LREG2, p_sfpu::LREG1, 1);
        TTI_SFPSETEXP(126, p_sfpu::LREG1, p_sfpu::LREG1, 1);
        TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG13, p_sfpu::LREG12, p_sfpu::LREG0, 0);
        TTI_SFPEXEXP(0, p_sfpu::LREG2, p_sfpu::LREG3, 0);

        // 1st iteration
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG2, 0);
        TTI_SFPNOP;
        TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPNOP;
        // 2nd iteration
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG2, 0);
        TTI_SFPNOP;
        TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPIADD((-126) & 0xfff, p_sfpu::LREG3, p_sfpu::LREG3, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_IMM);

        // Re-bias
        TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG4, 0);
        TTI_SFPIADD(0, p_sfpu::LREG4, p_sfpu::LREG3, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST);
        TTI_SFPSETEXP(0, p_sfpu::LREG0, p_sfpu::LREG3, 0);

	// Load a and multiply by 1/gcd(a, b)
        TTI_SFPLOAD(p_sfpu::LREG0, 4, 3, 0);
        TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TT_SFPLOAD(p_sfpu::LREG1, 4, 3, dst_offset * dst_tile_size);
        TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG1, 0);

	// Convert a/gcd(a, b) to int32
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG0, 6);

	// Finally, compute lcm(a, b) = a/gcd(a, b) * b
        calculate_sfpu_mul_u16_to_u32_body();

        TTI_SFPSTORE(p_sfpu::LREG4, 4, 3, 0);
        dst_reg++;
    }
}

inline void calculate_sfpu_lcm_init()
{
    calculate_sfpu_gcd_init();

    // constants for reciprocal calculation
    sfpi::vConstFloatPrgm0 = 48.0f / 17.0f;
    sfpi::vConstFloatPrgm1 = 32.0f / 17.0f;
}

}  // namespace sfpu
}  // namespace ckernel
