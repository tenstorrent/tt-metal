// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpu/ckernel_sfpu_exp.h"

namespace ckernel {
namespace sfpu {

// SELU(x) = scale ∗ ( max(0, x) + min(0, α ∗ (exp(x)−1) ) )
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_selu(uint param0, uint param1) {
    sfpi::vFloat scale = Converter::as_float(param0);
    sfpi::vFloat alpha = Converter::as_float(param1);
#pragma GCC unroll 8

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat exp_calc = (_calculate_exponential_body_<APPROXIMATION_MODE>(v) - 1.0) * alpha;
        v_if(exp_calc > 0.0f) { exp_calc = 0.0; }
        v_endif;
        v_if(v < 0.0f) { v = 0.0; }
        v_endif;
        sfpi::vFloat sum_max = v + exp_calc;
        sfpi::dst_reg[0] = sum_max * scale;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void selu_init() {
    _init_exponential_<APPROXIMATION_MODE, false>();
}

}  // namespace sfpu
}  // namespace ckernel
