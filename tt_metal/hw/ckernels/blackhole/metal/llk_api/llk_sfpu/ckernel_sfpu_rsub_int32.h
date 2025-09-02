// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_rsub_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // size of each tile in Dest is 64 rows
        constexpr uint dst_tile_size = 64;
        // operand A - int32
        TT_SFPLOAD(p_sfpu::LREG0, INT32, ADDR_MOD_7, dst_index_in0 * dst_tile_size);
        // operand B - int32 (offset by dst_offset * dest tile size)
        TT_SFPLOAD(p_sfpu::LREG1, INT32, ADDR_MOD_7, dst_index_in1 * dst_tile_size);
        // Reverse subtraction is performed using 2's complement by adding B to the negation of A: LREG1 + (-LREG0)
        // Use 6 as imod to convert operand A to 2's complement
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 6);
        // Store result from LREG_0 to dest
        TT_SFPSTORE(p_sfpu::LREG0, INT32, ADDR_MOD_7, dst_index_out * dst_tile_size);
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
