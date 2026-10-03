// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "sfpi.h"

namespace ckernel::sfpu {

// hardshrink_bw: grad times its piecewise-constant derivative, selected by the interval that holds x
// (activations/hardshrink_bw.json). DEST tile idst holds x and idst + 1 holds grad; the result replaces x.
template <int ITERATIONS = 32>
inline void calculate_hardshrink_bw_bf16() {
    using namespace sfpi;
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x = dst_reg[d];
        vFloat grad = dst_reg[32 + d];
        vFloat result = grad;
        v_if(x < -0.5f) { result = grad; }
        v_elseif(x <= 0.5f) { result = 0.0f; }
        v_endif;
        dst_reg[d] = result;
    }
}

}  // namespace ckernel::sfpu
