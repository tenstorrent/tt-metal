// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_log.h"

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_lgamma_part_positive() {
    constexpr float LOG_SQRT_2PI = 0.9189385332046727f;
    constexpr float M_PI = 3.14159265358979323846f;

    // Minimal coefficients for 0-3 ULP
    constexpr float r0 = 0.0833333333f;   // 1/12
    constexpr float r1 = -0.0027777777f;  // -1/360

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat z = in;

        // 1. Reflection for x < 0.5
        v_if(in < 0.5f) { z = 1.0f - in; }
        v_endif;

        // 2. Stirling base: (z - 0.5) * log(z) - z + log(sqrt(2*pi))
        sfpi::vFloat res = ((z - 0.5f) * _calculate_log_body_no_init_(z) - z + LOG_SQRT_2PI);

        // 3. High-Accuracy Correction (The "Bernoulli" series)
        // We use a minimax rational fit for 1/z.
        sfpi::vFloat inv_z2 = _sfpu_reciprocal_<2>(z * z);

        // correction = (1/z) * (r0 + r1/z^2)
        sfpi::vFloat correction = _sfpu_reciprocal_<2>(z) * (r0 + inv_z2 * r1);
        res = res + correction;

        // adjustment for inputs < 0.5 are done in composite.

        sfpi::dst_reg[0] = res;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void lgamma_init() {
    // log_init<false, false, is_fp32_dest_acc_en>();
    _init_reciprocal_<APPROXIMATION_MODE, is_fp32_dest_acc_en, false>();
}

}  // namespace sfpu
}  // namespace ckernel
