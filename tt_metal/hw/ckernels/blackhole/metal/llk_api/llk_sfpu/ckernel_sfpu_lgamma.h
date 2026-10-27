// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_log.h"
#include "cmath_common.h"

#include "sfpi.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_lgamma_stirling() {
    constexpr float LOG_SQRT_2PI = 0.9189385332046727f;

    // Minimal coefficients for 0-3 ULP
    constexpr float r0 = 0.0833333333f;   // 1/12
    constexpr float r1 = -0.0027777777f;  // -1/360

    // Two of the log body's constants are bound here and held in LREGs across the loop; as
    // literals inside the loop they would be re-materialised on every row. Two is what
    // this loop has room for next to the reciprocal: a third fails to allocate.
    const sfpi::vFloat log_ln2 = LogPolyNoInit::LN2;
    const sfpi::vFloat log_d = LogPolyNoInit::D;

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat z = in;

        // 1. Reflection for x < 0.5
        v_if(in < 0.5f) { z = 1.0f - in; }
        v_endif;

        // 2. Stirling base: (z - 0.5) * log(z) - z + log(sqrt(2*pi))
        sfpi::vFloat res = ((z - 0.5f) * _calculate_log_body_no_init_(z, log_ln2, log_d) - z + LOG_SQRT_2PI);

        // 3. Bernoulli correction: (1/z)(r0 + r1/z^2).
        sfpi::vFloat inv_z = sfpu_reciprocal_iter<2>(z);
        sfpi::vFloat correction = inv_z * (r0 + (inv_z * inv_z) * r1);
        res = res + correction;

        // TODO: use a polynomial bridge here instead
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
            v_if(sfpi::is_inf(in)) { result = std::numeric_limits<float>::infinity(); }
            v_endif;
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
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
    constexpr float r4 = 0.0008417508f;   // 1/1188
    constexpr float r5 = -0.0019175269f;  // -691/360360

    // Chebyshev fit for (w-1)(w-2)*Q(w-1.5) on [1, 2]
    constexpr float c0 = 4.8312890043e-01f;
    constexpr float c1 = -1.4595974798e-01f;
    constexpr float c2 = 6.2918526481e-02f;
    constexpr float c3 = -3.1317045370e-02f;
    constexpr float c4 = 1.6643589408e-02f;
    constexpr float c5 = -9.2951826577e-03f;
    constexpr float c6 = 6.4672372806e-03f;
    constexpr float c7 = -3.9237047468e-03f;

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat log_z = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat z = in;

        // 1. Reflection for x < 0.5
        v_if(in < 0.5f) { z = 1.0f - in; }
        v_endif;

        // Flat selection of Chebyshev parameters without nested v_if
        sfpi::vFloat t = z - 1.5f;
        sfpi::vFloat p = (z - 1.0f) * (z - 2.0f);
        sfpi::vFloat sub = 0.0f;

        v_if(z < 1.0f) {
            t = z - 0.5f;
            p = z * (z - 1.0f);
            sub = log_z;
        }
        v_endif;

        sfpi::vFloat res = 0.0f;

        v_if(z < 2.0f) {
            sfpi::vFloat q = PolynomialEvaluator::eval(t, c0, c1, c2, c3, c4, c5, c6, c7);
            res = p * q - sub;
        }
        v_elseif(z < 2.2f) {
            // Taylor Expansion around z=2 (d2 = z - 2.0)
            constexpr float q0 = 0.42278434f;   // 1 - gamma
            constexpr float q1 = 0.32246703f;   // (zeta(2)-1)/2
            constexpr float q2 = -0.06735230f;  // -(zeta(3)-1)/3
            constexpr float q3 = 0.02058081f;   // (zeta(4)-1)/4
            constexpr float q4 = -0.00738555f;  // -(zeta(5)-1)/5
            sfpi::vFloat d2 = z - 2.0f;
            res = d2 * PolynomialEvaluator::eval(d2, q0, q1, q2, q3, q4);
        }
        v_else {
            sfpi::vFloat inv_z = sfpu_reciprocal_iter<2>(z);
            sfpi::vFloat base = (z - 0.5f) * log_z - z + LOG_SQRT_2PI;
            sfpi::vFloat inv_z2 = (inv_z * inv_z);
            sfpi::vFloat correction = PolynomialEvaluator::eval(inv_z2, r0, r1, r2, r3, r4, r5);
            res = base + inv_z * correction;
        }
        v_endif;

        // reflection adjustment for inputs < 0.5 are done in calculate_lgamma_adjusted.
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = res;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void lgamma_stirling_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    // init for sfpu_reciprocal_iter<2> for Blackhole
    sfpi::vConstFloatPrgm0 = 2.0f;
}

}  // namespace ckernel::sfpu
