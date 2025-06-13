// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "ckernel_sfpu_exp.h"

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
        v_if(v > 0.0f) { sfpi::dst_reg[0] = v * scale; }
        v_elseif(v < 0.0f) {
            sfpi::vFloat exp_calc = (_sfpu_exp_21f_(v) - 1.0) * alpha;
            sfpi::dst_reg[0] = exp_calc * scale;
        }
        v_endif;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
