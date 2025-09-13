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
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_selu(uint scale, uint alpha) {
    sfpi::vFloat scale_value = Converter::as_float(scale);
    sfpi::vFloat alpha_value = Converter::as_float(alpha);
#pragma GCC unroll 8

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        v_if(v >= 0.0f) { sfpi::dst_reg[0] = v * scale_value; }
        v_else {
            sfpi::vFloat exp_calc = _sfpu_exp_21f_<true>(
                v);  // is_fp32_dest_acc_en set to true to avoid rounding as it has to be done at the end of operation
            sfpi::vFloat minus_mul = exp_calc - sfpi::vConst1;
            sfpi::vFloat result = minus_mul * alpha_value * scale_value;

            if constexpr (!is_fp32_dest_acc_en) {
                result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
            }
            sfpi::dst_reg[0] = result;
        }
        v_endif;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
