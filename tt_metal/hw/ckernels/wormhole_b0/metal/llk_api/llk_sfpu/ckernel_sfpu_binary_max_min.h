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
inline void calculate_binary_max_min(const uint dst_offset) {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size = 64;

        TTI_SFPLOAD(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_3, 0);                          // a
        TT_SFPLOAD(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_3, dst_offset * dst_tile_size);  // b

        // Swap and store maximum in lreg1, minimum in lreg0
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, 1);

        if constexpr (IS_MAX_OP) {
            TTI_SFPSTORE(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_3, 0);
        } else {
            TTI_SFPSTORE(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_3, 0);
        }
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
