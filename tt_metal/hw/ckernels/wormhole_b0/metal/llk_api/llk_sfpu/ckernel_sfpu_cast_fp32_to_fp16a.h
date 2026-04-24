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
inline void cast_fp32_to_fp16a(std::uint32_t dst_index_in, std::uint32_t dst_index_out) {
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // vFloat val = dst_reg[dst_index_in * SFP_DST_TILE_ROWS];
        // dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = float_to_fp16a(val, sfpi::RoundMode::NearestEven);
        TT_SFPLOAD(0, 0, 3, dst_index_in * SFP_DST_TILE_ROWS);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 8);
        TT_SFPSTORE(0, 1, 3, dst_index_out * SFP_DST_TILE_ROWS);
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
