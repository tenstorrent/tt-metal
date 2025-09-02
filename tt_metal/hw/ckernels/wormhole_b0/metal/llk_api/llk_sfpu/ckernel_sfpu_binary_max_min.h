// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <InstrModLoadStore INSTRUCTION_MODE, bool IS_MAX_OP = true, int ITERATIONS = 8>
inline void calculate_binary_max_min(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        // size of each tile in Dest is 64 rows
        constexpr uint dst_tile_size = 64;

        TT_SFPLOAD(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_3, dst_index_in0 * dst_tile_size);  // a
        TT_SFPLOAD(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_3, dst_index_in1 * dst_tile_size);  // b

        // Swap and store maximum in lreg1, minimum in lreg0
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, 1);

        if constexpr (IS_MAX_OP) {
            TT_SFPSTORE(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_3, dst_index_out * dst_tile_size);
        } else {
            TT_SFPSTORE(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_3, dst_index_out * dst_tile_size);
        }
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
