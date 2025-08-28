// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpu/ckernel_sfpu_exp.h"
#include "llk_defs.h"
namespace ckernel {
namespace sfpu {

template <ApproximationMode APPROX_MODE, int ITERATIONS = 8>
inline void calculate_expm1() {
    const bool SCALE_EN = false;             // Expm1 does not use scale.
    const bool SKIP_POSITIVE_CHECK = false;  // Expm1 does not skip positive check.
    const uint16_t exp_base_scale_factor = p_sfpu::kCONST_1_FP16B;

    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        v = _calculate_exponential_piecewise_<APPROX_MODE, SCALE_EN, SKIP_POSITIVE_CHECK>(
            v, exp_base_scale_factor);
        sfpi::dst_reg[0] = v - 1.0f;
        sfpi::dst_reg++;
    }
}

template <ApproximationMode APPROX_MODE>
void expm1_init() {
    const uint32_t EXP_BASE_SCALE_FACTOR = 0x3F800000;
    _init_exponential_<APPROX_MODE, false /*fast_mode*/, EXP_BASE_SCALE_FACTOR>();
}

}  // namespace sfpu
}  // namespace ckernel
