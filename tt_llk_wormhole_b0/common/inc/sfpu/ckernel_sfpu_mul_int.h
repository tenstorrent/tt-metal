// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_addrmod.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _mul_int_(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out)
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        // Split u16 inputs a and b into a = (a1 << 8) | a0; b = (b1 << 8) | b0,
        // where a0, a1, b0, b1 are u8.  Then cast to fp32, and calculate:
        //   lo  = a0*b0
        //   hi0 = a0*b1
        //   hi1 = a1*b0
        // Observe that these are < 2**16.  This allows conversion back to u16
        // using TTI_SFP_STOCH_RND.
        // Finally, the result will be lo + ((hi0 + hi1) << 8).

        constexpr uint dst_tile_size = 64;

        // a0
        TT_SFPLOAD(p_sfpu::LREG0, LO16, ADDR_MOD_3, dst_index_in0 * dst_tile_size);

        // a1
        TTI_SFPSHFT2(p_sfpu::LREG0, p_sfpu::LREG13, p_sfpu::LREG2, 5);
        TTI_SFPCAST(p_sfpu::LREG2, p_sfpu::LREG2, 0);

        // a0 = (a0 & mask) as fp32
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);

        // b0
        TT_SFPLOAD(p_sfpu::LREG1, LO16, ADDR_MOD_3, dst_index_in1 * dst_tile_size);

        // b1
        TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG13, p_sfpu::LREG3, 5);
        TTI_SFPCAST(p_sfpu::LREG3, p_sfpu::LREG3, 0);

        // b0 = (b0 & mask) as fp32
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);

        // hi0 = a0*b1
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);
        // lo = a0*b0
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        // hi1 = a1*b0
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);

        // lo = rnd(lo)
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG0, 6);

        // hi1 = rnd(hi1)
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG2, p_sfpu::LREG2, 6);
        // hi0 = rnd(hi0)
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG3, p_sfpu::LREG3, 6);

        // hi = hi0 + hi1
        TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_CC_NONE);
        // hi <<= 8
        TTI_SFPSHFT(8, 0, p_sfpu::LREG2, 1);

        // lo += hi
        TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);

        TT_SFPSTORE(p_sfpu::LREG0, LO16, ADDR_MOD_3, dst_index_out * dst_tile_size);

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void _init_mul_int_()
{
    sfpi::vConstIntPrgm0 = 0xff; // LREG12
    sfpi::vConstIntPrgm1 = -8;   // LREG13
}

} // namespace sfpu
} // namespace ckernel
