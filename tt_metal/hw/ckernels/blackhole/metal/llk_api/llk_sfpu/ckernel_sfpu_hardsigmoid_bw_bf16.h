// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "sfpi.h"

namespace ckernel::sfpu {

template <int ITERATIONS = 32>
inline void calculate_hardsigmoid_bw_bf16() {
    using namespace sfpi;
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x = dst_reg[d];
        vFloat grad = dst_reg[32 + d];
        vFloat scaled0 = convert<vFloat16b>(grad * 0.1666666716337204f, RoundMode::Nearest);
        vFloat result = 0.0f;
        v_if(x <= -3.0f) { result = 0.0f; }
        v_elseif(x < 3.0f) { result = scaled0; }
        v_endif;
        dst_reg[d] = result;
    }
}

}  // namespace ckernel::sfpu
