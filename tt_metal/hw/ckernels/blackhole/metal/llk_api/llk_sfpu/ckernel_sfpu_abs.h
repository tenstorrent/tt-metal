// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs(std::uint32_t dst_index_in, std::uint32_t dst_index_out) {
    // SFPU microcode
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[dst_index_in * SFP_DST_TILE_ROWS];
        dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = sfpi::abs(v);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs_int32(std::uint32_t dst_index_in, std::uint32_t dst_index_out) {
    // SFPU microcode
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(1, 12, ADDR_MOD_7, dst_index_in * SFP_DST_TILE_ROWS);
        TTI_SFPABS(0, 1, 0, 0);
        TT_SFPSTORE(0, 12, ADDR_MOD_7, dst_index_out * SFP_DST_TILE_ROWS);
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
