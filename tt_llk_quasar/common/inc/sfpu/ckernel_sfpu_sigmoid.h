// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"

namespace ckernel
{
namespace sfpu
{
// Calculates SIGMOID for results already in registers, used as a helper function.
// Input of sigmoid is expected to already be in src_reg, and the result will be stored in dest_reg.
// work_reg is an intermediate register used for calculations.
inline void _calculate_sigmoid_regs_(const std::uint32_t src_reg, const std::uint32_t work_reg, const std::uint32_t dest_reg)
{
    // ALthough SFPMUL/SFPADD are 2 cycle instructions, we don't need a TTI_NOP
    // because the hardware implicitly stalls if the next instruction depends on results
    TTI_SFPMUL(src_reg, p_sfpu::LCONST_neg1, p_sfpu::LCONST_0, work_reg, 0); // Multiply src_reg * -1, store result in work_reg, takes 2 cycles
    TTI_SFPNONLINEAR(work_reg, dest_reg, p_sfpnonlinear::EXP_MODE);          // Read value from work_reg, take exponential, load back into dest_reg
    TTI_SFPADD(p_sfpu::LCONST_1, dest_reg, p_sfpu::LCONST_1, work_reg, 0);   // Add 1 to dest_reg, store in work_reg, takes 2 cycles
    TTI_SFPNONLINEAR(work_reg, dest_reg, p_sfpnonlinear::RECIP_MODE);        // Read value from work_reg, approximate recip, load back into dest_reg
}

// Calculates SIGMOID for number of rows of output SFPU ops (Quasar = 2 rows)
inline void _calculate_sigmoid_sfp_rows_()
{
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, 0); // load from dest into lreg[0], uses ADDR_MOD_7 (set to all zeroes)

    _calculate_sigmoid_regs_(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG0); // calculate sigmoid using lreg[0] as src and dest, and lreg[1] as work register

    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, 0, 0); // store from lreg[0] into dest register
}

inline void _calculate_sigmoid_(const int iterations)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        _calculate_sigmoid_sfp_rows_();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>(); // does the dest_reg++ (increments by 2 rows)
    }
}

} // namespace sfpu
} // namespace ckernel
