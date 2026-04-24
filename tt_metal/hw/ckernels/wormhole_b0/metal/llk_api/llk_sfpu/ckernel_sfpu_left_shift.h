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
inline void calculate_left_shift(std::uint32_t dst_index_in, std::uint32_t dst_index_out, const uint shift_amt) {
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(0, 4, 3, dst_index_in * SFP_DST_TILE_ROWS);
        TT_SFPSHFT(shift_amt,0,0,1);
        TT_SFPSTORE(0, 4, 3, dst_index_out * SFP_DST_TILE_ROWS);
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
