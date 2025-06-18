// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
inline void _mul_int_(const uint dst_offset)
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        constexpr uint dst_tile_size = 64;
        // operand A - uint16
        TTI_SFPLOAD(p_sfpu::LREG0, LO16, ADDR_MOD_3, 0);
        // operand B - uint16
        TT_SFPLOAD(p_sfpu::LREG1, LO16, ADDR_MOD_3, dst_offset * dst_tile_size);

        // The following cast+mul+cast method provides accurate results if the product of the inputs < 2**24
        // since float32 can exactly represent integer up to 2**24. To preserve accuracy beyond 2**24,
        // we could split the 16-bit input into two 8-bit chunks, cast to fp32, multiply and then cast back.
        // uint16 -> fp32
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_NOP;
        // fp32 -> uint16
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG0, 6);
        TTI_SFPSTORE(p_sfpu::LREG0, LO16, ADDR_MOD_3, 0);

        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
