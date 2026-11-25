// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <limits>
#include "sfpi.h"

namespace ckernel::sfpu {

template <int ITERATIONS = 32>
inline void calculate_abs_bw_bf16() {
    using namespace sfpi;
    const vInt magnitude = 0x7fffffff;
    const vInt infinity = 0x7f800000;
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x = dst_reg[d];
        vFloat grad = dst_reg[32 + d];
        vFloat scaled0 = grad * -1.0f;
        vFloat scaled1 = grad * 0.0f;
        vFloat result = grad;
        v_if(x <= -1.1754943508222875e-38f) { result = scaled0; }
        v_elseif(x <= 0.0f) { result = scaled1; }
        v_endif;
        v_if((as<vInt>(x) & magnitude) > infinity) { result = scaled1; }
        v_endif;
        v_if((as<vInt>(result) & magnitude) > infinity) { result = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;
        dst_reg[d] = result;
    }
}

}  // namespace ckernel::sfpu
