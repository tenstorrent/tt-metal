// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_ops.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_binary_left_shift_(const uint dst_offset)
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        constexpr uint dst_tile_size = 64;
        // load
        TTI_SFPLOAD(0, 4, 3, 0);
        TT_SFPLOAD(1, 4, 3, dst_offset * dst_tile_size);
        // if (shift_amount < 0 OR shift_amount >= 32) -> result should be 0
        TTI_SFPSETCC(0, 1, 0, 4);
        TTI_SFPIADD(0xFE0, 1, 2, 1); // 0xFE0 = -32
        TTI_SFPCOMPC(0, 0, 0, 0);
        TTI_SFPMOV(0, 9, 0, 0);
        TTI_SFPENCC(0, 0, 0, 0);
        // shift left
        TTI_SFPSHFT(0, 1, 0, 0);
        // store result
        TTI_SFPSTORE(0, 4, 3, 0);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_binary_right_shift_(const uint dst_offset)
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        constexpr uint dst_tile_size = 64;
        // load
        TTI_SFPLOAD(0, 4, 3, 0);
        TT_SFPLOAD(1, 4, 3, dst_offset * dst_tile_size);
        TTI_SFPMOV(0, 0, 4, 0); // save shift_value for later
        // shift right
        TTI_SFPIADD(0, 9, 1, 6); // take negative of shift_amount to shift right
        TTI_SFPSHFT(0, 1, 0, 0);
        // if shift_value was negative, need to shift in 1's manually
        TTI_SFPSETCC(0, 4, 0, 0);    // only run if shift_value is negative
        TTI_SFPSETCC(0, 1, 0, 2);    // only needed if shift_amount>0
        TTI_SFPIADD(0x020, 1, 2, 5); // take 32-shift_amount (0x020 = 32)
        TTI_SFPNOT(0, 9, 3, 0);      // put all 1's into LREG3
        TTI_SFPSHFT(0, 2, 3, 0);     // shift all 1's by 32-shift_amount
        TTI_SFPOR(0, 3, 0, 0);       // OR in the 1's
        TTI_SFPENCC(0, 0, 0, 0);
        // store result
        TTI_SFPSTORE(0, 4, 3, 0);
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
