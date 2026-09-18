// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_log.h"
#include "cmath_common.h"

#include "sfpi.h"
#include "sfpu/ckernel_sfpu_log.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_lgamma_stirling() {
    constexpr float LOG_SQRT_2PI = 0.9189385332046727f;

    // Minimal coefficients for 0-3 ULP
    constexpr float r0 = 0.0833333333f;   // 1/12
    constexpr float r1 = -0.0027777777f;  // -1/360

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat z = in;

        // 1. Reflection for x < 0.5
        v_if(in < 0.5f) { z = 1.0f - in; }
        v_endif;

        // 2. Argument shift (N=4) to move z in [0.5, 2.5] into stable Stirling domain [4.5, 6.5]
        sfpi::vFloat shift_corr = 0.0f;
        v_if(z < 2.5f) {
            sfpi::vFloat prod = z * (z + 1.0f) * (z + 2.0f) * (z + 3.0f);
            shift_corr = _calculate_log_body_no_init_(prod);
            z = z + 4.0f;
        }
        v_endif;

        // 3. Stirling base: (z - 0.5) * log(z) - z + log(sqrt(2*pi))
        sfpi::vFloat res = ((z - 0.5f) * _calculate_log_body_no_init_(z) - z + LOG_SQRT_2PI);

        // 4. Bernoulli correction: (1/z)(r0 + r1/z^2).
        sfpi::vFloat inv_z = sfpu_reciprocal_iter<2>(z);
        sfpi::vFloat correction = inv_z * (r0 + (inv_z * inv_z) * r1);
        res = (res + correction) - shift_corr;

        v_if(in == 1.0f || in == 2.0f) { res = 0.0f; }
        v_endif;

        // reflection adjustment for inputs < 0.5 are done in calculate_lgamma_adjusted.

        if constexpr (!is_fp32_dest_acc_en) {
            res = sfpi::convert<sfpi::vFloat16b>(res, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = res;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_lgamma_stirling_fp32(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    constexpr float LOG_SQRT_2PI = 0.9189385332046727f;
    constexpr uint dst_tile_size_sfpi = 32;

    // Minimal coefficients for 0-3 ULP
    constexpr float r0 = 0.0833333333f;   // 1/12
    constexpr float r1 = -0.0027777777f;  // -1/360
    constexpr float r2 = 0.0007936507f;   // 1/1260
    constexpr float r3 = -0.0005952380f;  // -1/1680

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat log_z = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat z = in;

        // 1. Reflection for x < 0.5
        v_if(in < 0.5f) { z = 1.0f - in; }
        v_endif;

        // 2. Argument shift (N=4) to move z in [0.5, 2.5] into stable Stirling domain [4.5, 6.5]
        sfpi::vFloat shift_corr = 0.0f;
        sfpi::vFloat l_z = log_z;
        v_if(z < 2.5f) {
            sfpi::vFloat prod = z * (z + 1.0f) * (z + 2.0f) * (z + 3.0f);
            shift_corr = _calculate_log_body_no_init_(prod);
            z = z + 4.0f;
            l_z = _calculate_log_body_no_init_(z);
        }
        v_endif;

        // 3. Stirling base + Bernoulli correction on shifted argument
        sfpi::vFloat res = ((z - 0.5f) * l_z - z + LOG_SQRT_2PI);
        sfpi::vFloat inv_z = sfpu_reciprocal_iter<2>(z);
        sfpi::vFloat inv_z2 = (inv_z * inv_z);
        sfpi::vFloat correction = PolynomialEvaluator::eval(inv_z2, r0, r1, r2, r3);
        res = (res + inv_z * correction) - shift_corr;

        v_if(in == 1.0f || in == 2.0f) { res = 0.0f; }
        v_endif;

        // reflection adjustment for inputs < 0.5 are done in calculate_lgamma_adjusted.
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = res;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_lgamma_adjusted(
    const uint dst_index_in0,  // lgamma_stirling result
    const uint dst_index_in1,  // log|sin(pi * frac(x))| with integer adjustments
    const uint dst_index_in2,  // input x
    const uint dst_index_out) {
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;
    constexpr float ln_pi = 1.1447298858f;

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat res_stirling = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat log_sin_pi_x = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat in = sfpi::dst_reg[dst_index_in2 * dst_tile_size_sfpi];

        // ln(pi) - log|sin(pi * frac(x))|
        sfpi::vFloat reflection_adj = ln_pi - log_sin_pi_x;

        sfpi::vFloat result = res_stirling;

        // For x < 0.5: lgamma(x) = reflection_adj - lgamma(1-x); otherwise use res_stirling.
        v_if(in < 0.5f) { result = reflection_adj - res_stirling; }
        v_endif;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        } else {
            sfpi::vInt exp = sfpi::exexp(in);
            sfpi::vInt man = sfpi::exman(in);
            v_if(exp == 128 && man == 0) { result = std::numeric_limits<float>::infinity(); }
            v_endif;
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void lgamma_stirling_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    recip_init<APPROXIMATION_MODE, false, false>();
}

}  // namespace ckernel::sfpu
