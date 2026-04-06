// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softshrink(uint32_t param0) {
    // param0 = bit_cast<uint32_t>(lambda)
    sfpi::vFloat lambda = Converter::as_float(param0);
    sfpi::vFloat neg_lambda = -lambda;

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat result = 0.0f;  // default: 0 for -lambda <= x <= lambda

        v_if(v > lambda) { result = v - lambda; }
        v_endif;

        v_if(v < neg_lambda) { result = v + lambda; }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
