// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_sfpu_gcd(const uint dst_offset)
{
    // Binary GCD algorithm
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size = 64;

        TTI_SFPLOAD(p_sfpu::LREG0, 4, 3, 0); // a
        TT_SFPLOAD(p_sfpu::LREG1, 4, 3, dst_offset * dst_tile_size); // b
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG2, 0); // k
        // c = a | b
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG3, 0);
        TTI_SFPOR(0, p_sfpu::LREG1, p_sfpu::LREG3, 0);
        // a = abs(a), b = abs(b)
        TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG1, 0);

        // Binary GCD algorithm
        // First we find the largest power of two that divides both a and b exactly.
        // Set c = (a | b) and find the largest power of two, k, that divides c.

        // Emulate ctz instruction by counting trailing zeroes using a series of bit masks.
        // There is a maximum of 31 trailing zero bits.  The value 31 can be represented with 5 bits.
        // We progressively set these 5 bits by checking the masks 0xffff, 0xff, 0xf, 0x7, 0x3, 0x1,
        // each time shifting right by the width of the mask if the masked region is all zeroes.

        uint16_t mask = 0xffff;
        uint16_t mask_width = 16;
        #pragma GCC unroll 5
        for (int i = 0; i < 5; ++i) {
            // l4 = mask
            TT_SFPLOADI(p_sfpu::LREG4, 2, mask);
            // l4 &= c
            TTI_SFPAND(0, p_sfpu::LREG3, p_sfpu::LREG4, 0);
            // if (c & mask) == 0
            TTI_SFPSETCC(0, p_sfpu::LREG4, 0, 6);
            // c >>= mask_width
            TT_SFPSHFT((-mask_width) & ((1<<12)-1), p_sfpu::LREG3, p_sfpu::LREG3, 1);
            // k += mask_width
            TT_SFPLOADI(p_sfpu::LREG4, 2, mask_width);
            TTI_SFPOR(0, p_sfpu::LREG4, p_sfpu::LREG2, 0);
            TTI_SFPENCC(0, 0, 0, 0);

            mask_width >>= 1;
            mask >>= mask_width;
        }

        // a >>= k and b >>= k
        // k = -k for right shift
        // TODO we can track -k in the above loop to avoid this two-cycle negation
        TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG3, 4 | 2);
        TTI_SFPNOP;
        TTI_SFPSHFT(0, p_sfpu::LREG3, p_sfpu::LREG0, 1);
        TTI_SFPSHFT(0, p_sfpu::LREG3, p_sfpu::LREG1, 1);
        // restore k = -k
        TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG3, 4 | 2);

        // now, one of a or b might be even: if b is even then swap so that a is even
        // we can reuse the mask_width register as we know it should contain 1 after the above loop
        TTI_SFPAND(0, p_sfpu::LREG1, p_sfpu::LREG4, 0);
        // if (b & mask) == 0
        TTI_SFPSETCC(0, p_sfpu::LREG4, 0, 6);
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG4, 0);
        TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG1, 0);
        TTI_SFPENCC(0, 0, 0, 0);

        // The worst case for binary GCD with signed 32-bit integers requires 32 iterations.

        #pragma GCC unroll 32
        for (int i = 0; i < 32; ++i) {
            // Again, we emulate ctz, but this time for a only.
            // Since we're removing at least 1 bit with each iteration, we also
            // track the maximum number of trailing zeroes and only check as
            // many masks as necessary, e.g. only 4 are needed to count up to
            // 15 trailing zeroes, and so on.

            int max_zero_bits = 32 - i - 1; // maximum 5 bits
            mask_width = max_zero_bits >> 1; // maximum 4 bits

            // Round up to next highest power of 2 (assumes mask_width is up to 4 bits wide).
            mask_width--;
            mask_width |= mask_width >> 1;
            mask_width |= mask_width >> 2;
            mask_width++;
            mask = (1 << mask_width) - 1;

            #pragma GCC unroll 5
            while (mask_width != 0) {

                // l4 = mask
                TT_SFPLOADI(p_sfpu::LREG4, 2, mask);
                // l4 &= a
                TTI_SFPAND(0, p_sfpu::LREG0, p_sfpu::LREG4, 0);
                // if (a & mask) == 0
                TTI_SFPSETCC(0, p_sfpu::LREG4, 0, 6);
                // then, a >>= mask_width
                TT_SFPSHFT((-mask_width) & ((1<<12)-1), p_sfpu::LREG0, p_sfpu::LREG0, 1);
                TTI_SFPENCC(0, 0, 0, 0);

                mask_width >>= 1;
                mask >>= mask_width;
            }

            // if (b > a) swap(a, b)
            // a -= b

            // l4 = b
            TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG4, 0);
            // l4 = a - b
            TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG4, 8 | 2);
            TTI_SFPNOP;
            // if (a - b) >= 0 then that's the new value of a
            TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG0, 0);
            // else we set b = a and set a = -l4
            TTI_SFPCOMPC(0, 0, 0, 0);
            // set b = a
            TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
            // set a = abs(l4)
            TTI_SFPABS(0, p_sfpu::LREG4, p_sfpu::LREG0, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }

        //dst_reg[0] = a << sa;
        TTI_SFPSHFT(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, 4, 3, 0);
        //dst_reg[0] = a;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
