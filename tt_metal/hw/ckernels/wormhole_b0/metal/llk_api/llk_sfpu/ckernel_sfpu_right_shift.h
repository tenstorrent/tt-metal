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
inline void calculate_right_shift(std::uint32_t dst_index_in, std::uint32_t dst_index_out, const uint shift_amt) {
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vInt input = dst_reg[dst_index_in * SFP_DST_TILE_ROWS];
        vUInt val = reinterpret<vUInt>(input);

        v_if(input < 0) { val = setsgn(val - 1, 0); }
        v_endif;
        vInt res = reinterpret<vInt>(val >> shift_amt);

        v_if(input < 0) { res = setsgn(res + 1, input); }
        v_endif;

        dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = res;
        dst_reg++;
    }
}
}  // namespace sfpu
}  // namespace ckernel
