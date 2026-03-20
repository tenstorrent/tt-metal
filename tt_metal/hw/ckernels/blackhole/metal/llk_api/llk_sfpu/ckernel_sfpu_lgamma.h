// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_log.h"

#include "sfpi.h"

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

        // 2. Stirling base: (z - 0.5) * log(z) - z + log(sqrt(2*pi))
        sfpi::vFloat res = ((z - 0.5f) * _calculate_log_body_no_init_(z) - z + LOG_SQRT_2PI);

        // 3. Bernoulli correction: (1/z)(r0 + r1/z^2).
        sfpi::vFloat inv_z = _sfpu_reciprocal_<2>(z);
        sfpi::vFloat correction = inv_z * (r0 + (inv_z * inv_z) * r1);
        res = res + correction;

        // TODO: use a polynomial bridge here instead
        v_if(in == 1.0f || in == 2.0f) { res = 0.0f; }
        v_endif;

        // reflection adjustment for inputs < 0.5 are done in calculate_lgamma_adjusted.

        if constexpr (!is_fp32_dest_acc_en) {
            res = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(res, 0));
        }

        sfpi::dst_reg[0] = res;
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
            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
        } else {
            sfpi::vInt exp = sfpi::exexp(in);
            sfpi::vInt man = sfpi::exman9(in);
            v_if(exp == 128 && man == 0) { result = std::numeric_limits<float>::infinity(); }
            v_endif;
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_lgamma_stirling_fp32(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    constexpr float LOG_SQRT_2PI = 0.9189385332046727f;
    constexpr float LOG_SQRT_PI = 0.57236494f;  // lgamma(0.5) = ln(sqrt(pi))
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

        sfpi::vFloat range1 = z - 1.0f;  // (0.8 to 1.2)
        sfpi::vFloat range2 = z - 2.0f;  // (1.8 to 2.2)
        sfpi::vFloat abs_range1 = sfpi::abs(range1);
        sfpi::vFloat abs_range2 = sfpi::abs(range2);

        sfpi::vFloat res = z;

        // Polynomial bridge for small range inputs (z near 1 or 2 only)
        v_if((abs_range1 < 0.2f) || (abs_range2 < 0.2f)) {
            sfpi::vFloat d = range2;
            v_if(abs_range1 < abs_range2) { d = range1; }
            v_endif;
            constexpr float p0 = -0.57721566f;
            constexpr float p1 = 0.82246703f;
            constexpr float p2 = -0.40068563f;
            constexpr float p3 = 0.27058081f;
            constexpr float p4 = -0.20738555f;
            // res = d * (p0 + d * (p1 + d * (p2 + d * (p3 + d * p4))));
            res = d * PolynomialEvaluator::eval(d, p0, p1, p2, p3, p4);
        }
        v_else {
            // Stirling base + Bernoulli correction
            res = ((z - 0.5f) * log_z - z + LOG_SQRT_2PI);
            sfpi::vFloat inv_z = _sfpu_reciprocal_<2>(z);
            sfpi::vFloat inv_z2 = (inv_z * inv_z);
            // Bernoulli correction: r0 + inv_z2 * (inv_z2 * (r2 + inv_z2 * r3) + r1);
            sfpi::vFloat correction = PolynomialEvaluator::eval(inv_z2, r0, r1, r2, r3);
            res = res + inv_z * correction;
        }
        v_endif;

        // Handle boundary case
        v_if(sfpi::abs(z - 0.5f) < 0.01f) { res = LOG_SQRT_PI; }
        v_endif;

        // reflection adjustment for inputs < 0.5 are done in calculate_lgamma_adjusted.
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = res;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void lgamma_stirling_init() {
    _init_reciprocal_<APPROXIMATION_MODE, is_fp32_dest_acc_en, false>();
}

}  // namespace ckernel::sfpu
