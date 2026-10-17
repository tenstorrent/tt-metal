// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <limits>
#include "sfpi.h"

namespace ckernel::sfpu {

// hardshrink_bw: grad times its piecewise derivative, selected by the interval that holds x
// (activations/hardshrink_bw.json). DEST tile idst holds x and idst + 1 holds grad; the result replaces x.
template <int ITERATIONS = 32>
inline void calculate_hardshrink_bw_bf16() {
    using namespace sfpi;
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x = dst_reg[d];
        vFloat grad = dst_reg[32 + d];
        v_if(sfpi::is_nan(grad)) { grad = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;
        vFloat result = grad;
        v_if(x < -0.5f) { result = grad; }
        v_elseif(x <= 0.5f) { result = 0.0f; }
        v_endif;
        dst_reg[d] = result;
    }
}

}  // namespace ckernel::sfpu
