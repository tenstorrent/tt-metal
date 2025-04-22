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
inline void add_uint16(const uint dst_offset) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size = 64;
        // operand A - uint16
        TTI_SFPLOAD(p_sfpu::LREG0, 6, ADDR_MOD_3, 0);
        // operand B - uint16
        TT_SFPLOAD(p_sfpu::LREG1, 6, ADDR_MOD_3, dst_offset * dst_tile_size);

        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);
        TTI_SFPSTORE(p_sfpu::LREG0, 6, ADDR_MOD_3, 0);

        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
