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
    bool EN_32BIT_DEST,
    [[maybe_unused]] bool SCALE_EN = false,
    int ITERATIONS = SFPU_ITERATIONS,
    [[maybe_unused]] bool CLAMP_NEGATIVE = true>
void calculate_exponential([[maybe_unused]] const std::uint32_t exp_base_scale_factor = p_sfpu::kCONST_1_FP16B) {
    static_assert(SCALE_EN == false, "Non-default SCALE_EN not supported in Quasar exp");
    static_assert(CLAMP_NEGATIVE == true, "Non-default CLAMP_NEGATIVE not supported in Quasar exp");
    LLK_ASSERT(
        exp_base_scale_factor == p_sfpu::kCONST_1_FP16B,
        "Scaling is not supported in the current version of exp on Quasar.");
    // Two implementations only: APPROXIMATION_MODE = HW LUT (approx), else = full-precision fp32.
    // EN_32BIT_DEST (is_fp32_dest_acc_en) is threaded through for ABI compatibility.
    _calculate_exp_<APPROXIMATION_MODE, EN_32BIT_DEST, ITERATIONS>();
}

template <
    [[maybe_unused]] bool APPROXIMATION_MODE,
    [[maybe_unused]] uint32_t scale = 0x3F800000,
    [[maybe_unused]] bool CLAMP_NEGATIVE = true,
    [[maybe_unused]] bool EN_32BIT_DEST>
void exp_init() {
    static_assert(scale == 0x3F800000, "Non-default scale not supported in Quasar exp");
    static_assert(CLAMP_NEGATIVE == true, "Non-default CLAMP_NEGATIVE not supported in Quasar exp");
    llk_math_eltwise_unary_sfpu_init<SfpuType::exponential>();
    // Program ADDR_MOD_6 (Dest post-increment of SFP_ROWS) the per-pass exp store walks Dest with.
    _init_exp_();
}

}  // namespace sfpu
}  // namespace ckernel
