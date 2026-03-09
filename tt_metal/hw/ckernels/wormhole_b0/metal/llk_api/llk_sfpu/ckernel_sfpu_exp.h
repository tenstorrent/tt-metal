// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "sfpu/ckernel_sfpu_exp.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <bool SCALE_EN, bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _ckernel_sfpu_exp_(sfpi::vFloat val, const uint exp_base_scale_factor) {
    if constexpr (SCALE_EN) {
        val = val * sfpi::s2vFloat16b(exp_base_scale_factor);
    }
    sfpi::vFloat result = _sfpu_exp_improved_<is_fp32_dest_acc_en>(val);
    return result;
}

template <
    bool APPROXIMATION_MODE,
    bool FAST_APPROX,
    bool is_fp32_dest_acc_en,
    bool SCALE_EN = false,
    int ITERATIONS = 8,
    bool SKIP_POSITIVE_CHECK = false,
    bool CLAMP_NEGATIVE = true>
void calculate_exponential(const uint exp_base_scale_factor = p_sfpu::kCONST_1_FP16B) {
    if constexpr (APPROXIMATION_MODE) {
        _calculate_exponential_<
            APPROXIMATION_MODE,
            SCALE_EN,
            ITERATIONS,
            FAST_APPROX,
            SKIP_POSITIVE_CHECK,
            CLAMP_NEGATIVE>(exp_base_scale_factor);
    } else {
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];
            sfpi::dst_reg[0] = _ckernel_sfpu_exp_<SCALE_EN, is_fp32_dest_acc_en>(val, exp_base_scale_factor);
            sfpi::dst_reg++;
        }
    }
}

template <bool APPROXIMATION_MODE, bool FAST_APPROX, uint32_t scale = 0x3F800000, bool CLAMP_NEGATIVE = true>
void exp_init() {
    _init_exponential_<APPROXIMATION_MODE, FAST_APPROX, scale, CLAMP_NEGATIVE>();
}

}  // namespace sfpu
}  // namespace ckernel
