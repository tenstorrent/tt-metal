// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "noc_nonblocking_api.h"

#include "sfpi.h"

using namespace sfpi;

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _add_int32_(const int iterations, const uint dst_offset) {
    // Operand A is input1 (int32)
    // Operand B is input2 (int32)
    // Output is int32
    #pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // operand A - int32
        TTI_SFPLOAD(0, 12, 3, 0);
        // operand B - int32
        TT_SFPLOAD(1, 12, 3, dst_offset * 64);
        TTI_SFPIADD(0, 1, 0, 4);
        // MAD has a 2-cycle pipeline latency so we need one cycle latency until next instr can consume the result
        TTI_NOP;
        // LREG_0 -> dest as int32
        TTI_SFPSTORE(0, 12, 3, 0);
        dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
