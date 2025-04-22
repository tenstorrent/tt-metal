// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void add_int16(const uint16_t dst_offset) {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size = 64;
        // operand A - int16
        TTI_SFPLOAD(p_sfpu::LREG0, 6, ADDR_MOD_7, 0);
        // operand B - int16
        TT_SFPLOAD(p_sfpu::LREG1, 6, ADDR_MOD_7, dst_offset * dst_tile_size);

        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);
        TTI_SFPSTORE(p_sfpu::LREG0, 6, ADDR_MOD_7, 0);

        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
