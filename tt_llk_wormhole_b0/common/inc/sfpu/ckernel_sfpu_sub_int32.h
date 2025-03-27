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

template <bool APPROXIMATION_MODE, bool SIGN_MAGNITUDE_FORMAT, int ITERATIONS>
inline void _sub_int32_(const uint dst_offset)
{
    // Use '12' if Dest is in sign-magnitude format and '4' for 2's complement,
    // because TTI_SFPIADD requires 2's complement format in LREGs
    constexpr int sfpload_instr_mod = SIGN_MAGNITUDE_FORMAT ? 12 : 4;

// Operand A is input1 (int32)
// Operand B is input2 (int32)
// Output is int32
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        // operand A - int32
        TTI_SFPLOAD(1 /*lreg*/, sfpload_instr_mod, 3 /*addr_mode*/, 0 /*dest*/);
        // operand B - int32
        TT_SFPLOAD(0 /*lreg*/, sfpload_instr_mod, 3 /*addr_mode*/, dst_offset * 64 /*dest*/);
        // Use 6 as imod to convert operand B to 2's complement
        TTI_SFPIADD(0 /*imm*/, 1 /*lreg*/, 0 /*ldest*/, 6 /*imod*/);
        // LREG_0 -> dest as int32
        TTI_SFPSTORE(0 /*lreg_ind*/, sfpload_instr_mod, 3 /*addr_mode*/, 0 /*dest*/);
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
