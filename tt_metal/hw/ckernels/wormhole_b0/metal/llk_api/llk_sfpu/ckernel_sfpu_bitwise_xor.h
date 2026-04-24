// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "limits.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_bitwise_xor(std::uint32_t dst_index_in, std::uint32_t dst_index_out, const uint value) {
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vInt input = dst_reg[dst_index_in * SFP_DST_TILE_ROWS];
        vInt v = value;
        vInt res = input ^ v;
        v_if(res > INT_MIN && res < 0) {
            res = 0 - res;
            res = setsgn(res, v);
        }
        v_endif dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = res;
        dst_reg++;
    }
}
}  // namespace sfpu
}  // namespace ckernel
