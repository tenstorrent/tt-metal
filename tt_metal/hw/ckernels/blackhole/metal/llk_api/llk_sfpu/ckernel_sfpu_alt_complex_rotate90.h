// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 4>
inline void calculate_alt_complex_rotate90(std::uint32_t dst_index_in, std::uint32_t dst_index_out) {
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat val = dst_reg[dst_index_in * SFP_DST_TILE_ROWS];
        dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = -dst_reg[dst_index_in * SFP_DST_TILE_ROWS + 1];
        dst_reg[dst_index_out * SFP_DST_TILE_ROWS + 1] = val;
        dst_reg += 2;
    }
}

}  // namespace sfpu
}  // namespace ckernel
