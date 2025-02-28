// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <int ITERATIONS = 8>
inline void calculate_sfpu_gcd(const uint dst_offset)
{
    // Binary GCD algorithm.
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size = 64;

        TTI_SFPLOAD(p_sfpu::LREG0, 4, 3, 0); // a
        TT_SFPLOAD(p_sfpu::LREG1, 4, 3, dst_offset * dst_tile_size); // b

        TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG0, 0); // a = abs(a)
        TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG1, 0); // b = abs(b)
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0); // c = a
        TTI_SFPOR(0, p_sfpu::LREG1, p_sfpu::LREG2, 0); // c |= b

        TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG3, 0); // d = c
        TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG3, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST); // d = -d
        TTI_SFPAND(0, p_sfpu::LREG2, p_sfpu::LREG3, 0); // d &= c (isolate LSB)
        TTI_SFPLZ(0, p_sfpu::LREG3, p_sfpu::LREG3, 0); // d = clz(d)
        TTI_SFPIADD((-31) & 0xfff, p_sfpu::LREG3, p_sfpu::LREG3, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_IMM); // d -= 31

        TTI_SFPSHFT(0, p_sfpu::LREG3, p_sfpu::LREG0, 0); // a >>= d
        TTI_SFPSHFT(0, p_sfpu::LREG3, p_sfpu::LREG1, 0); // b >>= d

        // Ensure that b is odd: if LSB is zero, then swap with a.
        TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG12, p_sfpu::LREG2, 5); // c = b << 31
        TTI_SFPSETCC(0, p_sfpu::LREG2, 0, 6); // if c == 0 then b is even
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, 0); // swap(a, b)
        TTI_SFPENCC(0, 0, 0, 0);

        TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG0, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST); // a = -a

        // We store and execute one iteration of binary GCD.
        TTI_REPLAY(0, 7, 1, 1);

        // We store {-a, a} in {LREG0, LREG2}, which is convenient for isolating the LSB of a.
        TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG2, 0); // LREG2 = +a
        TTI_SFPAND(0, p_sfpu::LREG2, p_sfpu::LREG0, 0); // LREG0 &= a (isolate LSB and overwrite -a)
        TTI_SFPLZ(0, p_sfpu::LREG0, p_sfpu::LREG0, SFPLZ_MOD1_CC_NE0); // LREG0 = clz(LREG0), disable lanes where a == 0
        TTI_SFPIADD((-31) & 0xfff, p_sfpu::LREG0, p_sfpu::LREG0, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_IMM); // LREG0 -= 31
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LREG0, 5); // LREG0 = a >> LREG0, making a definitely odd (now both a and b are odd)
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, SFPSWAP_MOD1_VEC_MIN_MAX); // ensure b < a
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST); // a = b - a (now a is even)

        // We replay 29 iterations, making a total of 30 iterations so far.
        for (int i = 0; i < 29; ++i) {
            TTI_REPLAY(0, 7, 0, 0);
        }

        // Strictly speaking, the worst case for 31-bit inputs is 32 iterations.
        // However, the 32nd iteration doesn't modify b, so we can skip it unless we need to check the termination condition a == 0 for correctness.
        // One more iteration, but skipping the final operation, as it only affects a, making a total of 31 iterations.
        TTI_REPLAY(0, 6, 0, 0);

        // If we want to check a == 0, remove the last REPLAY call, replace 29 with 31 in the loop above, and replace SFPENCC with SFPCOMPC below.
        TTI_SFPENCC(0, 0, 0, 0);

        TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG3, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST); // d = -d
        TTI_SFPSHFT(0, p_sfpu::LREG3, p_sfpu::LREG1, 0); // b <<= d

        // dst_reg[0] = b;
        TTI_SFPSTORE(p_sfpu::LREG1, 4, 3, 0);
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
