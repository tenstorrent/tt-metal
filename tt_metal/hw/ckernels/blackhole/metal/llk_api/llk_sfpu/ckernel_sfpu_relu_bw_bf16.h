// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <limits>
#include "sfpi.h"

namespace ckernel::sfpu {

// relu_bw: grad times its piecewise derivative, selected by the interval that holds x
// (activations/relu_bw.json). DEST tile idst holds x and idst + 1 holds grad; the result replaces x.
template <int ITERATIONS = 32>
inline void calculate_relu_bw_bf16() {
    using namespace sfpi;
    const vInt magnitude = 0x7fffffff;
    const vInt infinity = 0x7f800000;
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x = dst_reg[d];
        vFloat grad = dst_reg[32 + d];
        v_if((as<vInt>(grad) & magnitude) > infinity) { grad = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;
        vFloat result = grad;
        v_if((as<vInt>(x) & magnitude) <= infinity) {
            v_if(x <= 0.0f) { result = 0.0f; }
            v_endif;
        }
        v_endif;
        dst_reg[d] = result;
    }
}

}  // namespace ckernel::sfpu
