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

        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0); // c = a
        TTI_SFPOR(0, p_sfpu::LREG1, p_sfpu::LREG2, 0); // c |= b

        TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG3, 0); // d = c
        TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG3, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST); // d = -d
        TTI_SFPAND(0, p_sfpu::LREG2, p_sfpu::LREG3, 0); // d &= c (isolate LSB)
        TTI_SFPLZ(0, p_sfpu::LREG3, p_sfpu::LREG3, 0); // d = clz(d)

        // Ensure that b is odd: if LSB is zero, then swap with a.
        TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LREG2, SFPSHFT2_MOD1_SHFT_LREG); // c = b << d
        TTI_SFPSETCC(0, p_sfpu::LREG2, 0, 6); // if c == 0 then b is even
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, 0); // swap(a, b)
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG0, 0); // a = abs(a)
        TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG1, 0); // b = abs(b)

        TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG0, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST); // a = -a
        TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG3, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST); // d = -d

        if (d == 0) {
            // Store and execute 4 iterations of binary GCD.
            TTI_REPLAY(0, 7 * 4, 1, 1);

            #pragma GCC unroll 4
            for (int i = 0; i < 4; ++i) {
                // We store {-a, a} in {LREG0, LREG2}, which is convenient for isolating the LSB of a.
                TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG2, 0); // LREG2 = +a
                TTI_SFPAND(0, p_sfpu::LREG2, p_sfpu::LREG0, 0); // LREG0 &= a (isolate LSB and overwrite -a)
                TTI_SFPLZ(0, p_sfpu::LREG0, p_sfpu::LREG0, SFPLZ_MOD1_CC_NE0); // LREG0 = clz(LREG0), disable lanes where a == 0
                TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG0, SFPIADD_MOD1_CC_NONE); // LREG0 += d
                TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LREG0, SFPSHFT2_MOD1_SHFT_LREG); // LREG0 = a >> -LREG0, making a definitely odd (now both a and b are odd)
                TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, SFPSWAP_MOD1_VEC_MIN_MAX); // ensure b < a
                TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST); // a = b - a (now a is even)
            }
        } else {
            TTI_REPLAY(0, 7 * 4, 0, 0);
        }

        // Replay 6 times, making a total of 28 iterations so far.
        #pragma GCC unroll 6
        for (int i = 0; i < 6; ++i) {
            TTI_REPLAY(0, 7 * 4, 0, 0);
        }

        // Replay 2 more iterations, making a total of 30 iterations.
        // The worst case for 31-bit inputs is 31 iterations, but we can skip the final iteration as it only affects a.
        // In addition, we can skip the final operation of the 30th iteration as it only affects a.
        TTI_REPLAY(0, 7 * 2 - 1, 0, 0);

        TTI_SFPENCC(0, 0, 0, 0);

        // dst_reg[0] = b;
        TTI_SFPSTORE(p_sfpu::LREG1, 4, 3, 0);
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
