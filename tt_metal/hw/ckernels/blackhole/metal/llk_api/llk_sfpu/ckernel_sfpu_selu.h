// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_converter.h"
#include "ckernel_sfpu_exp.h"
#include "sfpi.h"
#include "sfpi_fp16.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_selu() {
    // SELU(x) = scale * (max(0,x) + min(0, alpha*(exp(x)-1)))
    // Equivalently:
    //   x >= 0: scale * x
    //   x <  0: scale * alpha * (exp(x) - 1)
    //
    // Fixed constants (not user-configurable):
    //   scale = 1.0507009873554804934193349852946
    //   alpha = 1.6732632423543772848170429916717

    constexpr bool SCALE_EN = false;
    constexpr bool SKIP_POSITIVE_CHECK = false;
    constexpr std::uint16_t exp_base_scale_factor = p_sfpu::kCONST_1_FP16B;

    // alpha = 1.6732632... in FP32: 0x3FD63840
    sfpi::vFloat v_alpha = Converter::as_float(0x3FD63840);
    // scale = 1.0507009... in FP32: 0x3F868640
    sfpi::vFloat v_scale = Converter::as_float(0x3F868640);

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];

        v_if(v < 0.0f) {
            // Negative branch: compute alpha * (exp(x) - 1)
            sfpi::vFloat v_exp = _calculate_exponential_piecewise_<APPROXIMATION_MODE, SCALE_EN, SKIP_POSITIVE_CHECK>(
                v, exp_base_scale_factor);
            v = v_alpha * (v_exp - 1.0f);
        }
        v_endif;

        // Unconditionally multiply all lanes by scale:
        //   positive: scale * x
        //   negative: scale * alpha * (exp(x) - 1)
        v = v_scale * v;

        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void selu_init() {
    const std::uint32_t EXP_BASE_SCALE_FACTOR = 0x3F800000;
    const bool FAST_APPROX = false;
    _init_exponential_<APPROXIMATION_MODE, FAST_APPROX, EXP_BASE_SCALE_FACTOR>();
}

}  // namespace sfpu
}  // namespace ckernel
