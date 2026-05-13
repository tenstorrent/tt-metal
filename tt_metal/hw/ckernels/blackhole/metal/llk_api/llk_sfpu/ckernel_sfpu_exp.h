// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "sfpu/ckernel_sfpu_exp.h"
#include "ckernel_sfpu_conversions.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en,
    bool SCALE_EN = false,
    int ITERATIONS = 8,
    bool CLAMP_NEGATIVE = true>
void calculate_exponential(
    std::uint32_t dst_index_in,
    std::uint32_t dst_index_out,
    const uint exp_base_scale_factor = p_sfpu::kCONST_1_FP16B) {
    _calculate_exponential_<APPROXIMATION_MODE, SCALE_EN, ITERATIONS, CLAMP_NEGATIVE, is_fp32_dest_acc_en>(
        dst_index_in, dst_index_out, exp_base_scale_factor);
}

template <bool APPROXIMATION_MODE, uint32_t scale = 0x3F800000, bool CLAMP_NEGATIVE = true>
void exp_init() {
    _init_exponential_<APPROXIMATION_MODE, scale, CLAMP_NEGATIVE>();
}

}  // namespace sfpu
}  // namespace ckernel
