// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
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
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG2, 0); // sa
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG3, 0); // sb
        TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG1, 0);

        // Emulate ctz instruction by counting trailing zeroes using a series of bit masks.
        // There is a maximum of 31 trailing zero bits.  The value 31 can be represented with 5 bits.
        // We progressively set these 5 bits by checking the masks 0xffff, 0xff, 0xf, 0x7, 0x3, 0x1,
        // each time shifting right by the width of the mask if the masked region is all zeroes.

        // For the binary GCD algorithm, we do this for both a and b.

        uint32_t mask = 0xffff;
        uint32_t mask_width = 16;
        #pragma GCC unroll 5
        for (int i = 0; i < 5; ++i) {
            // l4 = mask
            TT_SFPLOADI(p_sfpu::LREG4, 2, mask);
            // l4 &= a
            TTI_SFPAND(0, p_sfpu::LREG0, p_sfpu::LREG4, 0);
            // if (a & mask) == 0
            TTI_SFPSETCC(0, p_sfpu::LREG4, 0, 6);
            // then, a >>= mask_width
            TT_SFPSHFT((-mask_width) & ((1<<12)-1), p_sfpu::LREG0, p_sfpu::LREG0, 1);
            // sa += mask_width
            TT_SFPLOADI(p_sfpu::LREG4, 2, mask_width);
            TTI_SFPOR(0, p_sfpu::LREG4, p_sfpu::LREG2, 0);
            TTI_SFPENCC(0, 0, 0, 0);

            // l4 = mask
            TT_SFPLOADI(p_sfpu::LREG4, 2, mask);
            // l4 &= b
            TTI_SFPAND(0, p_sfpu::LREG1, p_sfpu::LREG4, 0);
            // if (b & mask) == 0
            TTI_SFPSETCC(0, p_sfpu::LREG4, 0, 6);
            // then, b >>= mask_width
            TT_SFPSHFT((-mask_width) & ((1<<12)-1), p_sfpu::LREG1, p_sfpu::LREG1, 1);
            // sb += mask_width
            TT_SFPLOADI(p_sfpu::LREG4, 2, mask_width);
            TTI_SFPOR(0, p_sfpu::LREG4, p_sfpu::LREG3, 0);
            TTI_SFPENCC(0, 0, 0, 0);

            mask_width >>= 1;
            mask >>= mask_width;
        }

        // This is possibly a convoluted way to say: sa = min(sa, sb)

        // set sb = sa - sb
        TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG3, 8 | 2);
        TTI_SFPNOP;
        // if (sa - sb) >= 0 then sa >= sb and we need to swap
        // now we do sb = sa - (sa - sb), which gives us the original sb
        TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG3, 4 | 2);
        TTI_SFPNOP;
        // now just copy back to sa
        TTI_SFPMOV(0, p_sfpu::LREG3, p_sfpu::LREG2, 0);
        TTI_SFPENCC(0, 0, 0, 0);

        // The worst case for binary GCD with signed 32-bit integers requires 32 iterations.

        #pragma GCC unroll 32
        for (int i = 0; i < 32; ++i) {
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

            // Again, we emulate ctz, but this time for a only, and we don't need to store the shift amount this time.

            mask = 0xffff;
            mask_width = 16;

            #pragma GCC unroll 5
            for (int j = 0; j < 5; ++j) {

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
