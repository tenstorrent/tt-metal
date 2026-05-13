// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_prelu(std::uint32_t dst_index_in, std::uint32_t dst_index_out, uint value) {
    // SFPU microcode
    vFloat init = Converter::as_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat a = dst_reg[0];
        v_if(a < 0.0f) { a = a * init; }
        v_endif;
        dst_reg[(dst_index_out - dst_index_in) * TILE_R_DIM] = a;
        dst_reg++;
    }
}
}  // namespace sfpu
}  // namespace ckernel
