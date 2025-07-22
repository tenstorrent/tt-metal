// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "sfpu/ckernel_sfpu_exp.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

sfpi_inline sfpi::vFloat sfpu_exp(sfpi::vFloat val) { return _sfpu_exp_(val); }

template <
    bool APPROXIMATION_MODE,
    bool FAST_APPROX,
    bool SCALE_EN = false,
    int ITERATIONS = 8,
    bool SKIP_POSITIVE_CHECK = false>
void calculate_exponential(const uint exp_base_scale_factor = p_sfpu::kCONST_1_FP16B) {
    _calculate_exponential_<APPROXIMATION_MODE, SCALE_EN, ITERATIONS, FAST_APPROX, SKIP_POSITIVE_CHECK>(
        exp_base_scale_factor);
}

template <bool APPROXIMATION_MODE, bool FAST_APPROX, uint32_t scale = 0x3F800000>
void exp_init() {
    _init_exponential_<APPROXIMATION_MODE, FAST_APPROX, scale>();
}

}  // namespace sfpu
}  // namespace ckernel
