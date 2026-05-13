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
inline void calculate_bitwise_and(std::uint32_t dst_index_in, std::uint32_t dst_index_out, const uint value) {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vInt input = dst_reg[0];
        vInt res = input & value;
        dst_reg[(dst_index_out - dst_index_in) * TILE_R_DIM] = res;
        dst_reg++;
    }
}
}  // namespace sfpu
}  // namespace ckernel
