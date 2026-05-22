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
inline void calculate_add1(std::uint32_t dst_index_in, std::uint32_t dst_index_out) {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat val = dst_reg[0];
        dst_reg[(dst_index_out - dst_index_in) * TILE_R_DIM] = 1.0f + val;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
