// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_assert.h"
#include "sfpu/ckernel_sfpu_exp.h"

namespace ckernel {
namespace sfpu {

template <
    bool APPROXIMATION_MODE,
    [[maybe_unused]] bool is_fp32_dest_acc_en,
    [[maybe_unused]] bool SCALE_EN = false,
    int ITERATIONS = 8,
    [[maybe_unused]] bool CLAMP_NEGATIVE = true>
void calculate_exponential([[maybe_unused]] const uint exp_base_scale_factor = p_sfpu::kCONST_1_FP16B) {
    static_assert(is_fp32_dest_acc_en == false);
    static_assert(SCALE_EN == false);
    static_assert(CLAMP_NEGATIVE == true);
    LLK_ASSERT(
        exp_base_scale_factor == p_sfpu::kCONST_1_FP16B,
        "Scaling is not supported in the current version of exp on quasar.");
    _calculate_exp_<APPROXIMATION_MODE>(ITERATIONS);
}

template <
    [[maybe_unused]] bool APPROXIMATION_MODE,
    [[maybe_unused]] uint32_t scale = 0x3F800000,
    [[maybe_unused]] bool CLAMP_NEGATIVE = true>
void exp_init() {
    static_assert(scale == 0x3F800000);
    static_assert(CLAMP_NEGATIVE == true);
    llk_math_eltwise_unary_sfpu_init<SfpuType::exponential>();
}

}  // namespace sfpu
}  // namespace ckernel
