// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
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
inline void calculate_bitwise_not(std::uint32_t dst_index_in, std::uint32_t dst_index_out) {
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::LREG4, ADDR_MOD_3, dst_index_in * SFP_DST_TILE_ROWS);
        TTI_SFPNOT(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TT_SFPSTORE(p_sfpu::LREG0, p_sfpu::LREG4, ADDR_MOD_3, dst_index_out * SFP_DST_TILE_ROWS);
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
