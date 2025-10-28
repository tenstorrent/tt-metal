// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_trisc_common.h"
#include "cmath_common.h"

namespace ckernel
{
namespace sfpu
{
// Calculates Typecast for number of rows of output SFPU ops (Quasar = 2 rows)
inline void _calculate_typecast_int32_to_fp32_rows()
{
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, 0); // load from dest into lreg[0], uses ADDR_MOD_7 (set to all zeroes)
    // TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG2, 3); //convert from 2s completent to sign+magnitude

    TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG1, 0); // convert from int32 sign+mag to fp32 using rnd nearest even

    TTI_SFPSTORE(p_sfpu::LREG1, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0, 0); // Store from lreg[1] into dest register
}

inline void _calculate_typecast_int32_to_fp32_(const int iterations)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        _calculate_typecast_int32_to_fp32_rows();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>(); // does the dest_reg++ (increments by 2 rows)
    }
}

} // namespace sfpu
} // namespace ckernel
