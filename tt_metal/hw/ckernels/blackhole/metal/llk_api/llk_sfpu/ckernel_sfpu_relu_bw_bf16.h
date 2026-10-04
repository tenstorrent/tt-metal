// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "sfpi.h"

namespace ckernel::sfpu {

// relu_bw: grad times its piecewise derivative, selected by the interval that holds x
// (activations/relu_bw.json). DEST tile idst holds x and idst + 1 holds grad; the result replaces x.
template <int ITERATIONS = 32>
inline void calculate_relu_bw_bf16() {
    using namespace sfpi;
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x = dst_reg[d];
        vFloat grad = dst_reg[32 + d];
        vFloat result = grad;
        v_if(x <= 0.0f) { result = 0.0f; }
        v_endif;
        // A NaN compares by its sign; select it by its BF16 encoding: exponent all ones, mantissa nonzero.
        vUInt raw = dst_reg[d].mode<::sfpi::DataLayout::U16>();
        v_if((raw & 0x00ff) == 0x00ff && (raw & 0x7f00) != 0) { result = grad; }
        v_endif;
        dst_reg[d] = result;
    }
}

}  // namespace ckernel::sfpu
