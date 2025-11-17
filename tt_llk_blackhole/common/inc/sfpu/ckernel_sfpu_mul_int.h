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
        // size of each tile in Dest is 64 rows
        constexpr uint dst_tile_size = 64;
        // operand A - uint16
        TT_SFPLOAD(p_sfpu::LREG0, LO16, ADDR_MOD_7, dst_index_in0 * dst_tile_size);
        // operand B - uint16
        TT_SFPLOAD(p_sfpu::LREG1, LO16, ADDR_MOD_7, dst_index_in1 * dst_tile_size);

        TTI_SFPMUL24(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TT_SFPSTORE(p_sfpu::LREG0, LO16, ADDR_MOD_7, dst_index_out * dst_tile_size);

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void _init_mul_int_()
{
}

} // namespace sfpu
} // namespace ckernel
