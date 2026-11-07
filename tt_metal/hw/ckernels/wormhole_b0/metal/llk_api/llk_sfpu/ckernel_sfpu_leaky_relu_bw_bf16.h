// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <limits>
#include "sfpi.h"

namespace ckernel::sfpu {

template <int ITERATIONS = 32>
inline void calculate_leaky_relu_bw_bf16() {
    using namespace sfpi;
    const vInt magnitude = 0x7fffffff;
    const vInt infinity = 0x7f800000;
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x = dst_reg[d];
        vFloat grad = dst_reg[32 + d];
        v_if((as<vInt>(grad) & magnitude) > infinity) { grad = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;
        vFloat scaled0 = convert<vFloat16b>(grad * 0.009999999776482582f, RoundMode::Nearest);
        vFloat result = grad;
        v_if(x <= 0.0f) { result = scaled0; }
        v_endif;
        v_if((as<vInt>(x) & magnitude) > infinity) { result = scaled0; }
        v_endif;
        dst_reg[d] = result;
    }
}

}  // namespace ckernel::sfpu
