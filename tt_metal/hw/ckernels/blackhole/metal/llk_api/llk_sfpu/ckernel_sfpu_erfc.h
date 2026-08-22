// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_converter.h"

#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include "sfpu/ckernel_sfpu_polyval.h"

#include "ckernel_sfpu_piecewise_rational.h"

namespace ckernel::sfpu {

#ifdef INP_FLOAT32

constexpr float ERFC_FP32_ASYMPTOTIC_THRESHOLD = 1.5f;

inline sfpi::vFloat calculate_erfc_fp32_central_(const sfpi::vFloat abs_x) {
    return PolynomialEvaluator::eval(
        abs_x,
        1.0000000000e+00f,
        -1.1283793449e+00f,
        4.6854288485e-06f,
        3.7606471777e-01f,
        4.4099494698e-04f,
        -1.1475947499e-01f,
        5.4435851052e-03f,
        1.6482034698e-02f,
        1.3505007140e-02f,
        -1.7055012286e-02f,
        6.6216276027e-03f,
        -1.1403404642e-03f,
        7.0706504630e-05f);
}

inline sfpi::vFloat calculate_erfc_fp32_asymptotic_(const sfpi::vFloat abs_x) {
    const sfpi::vFloat inv = sfpu_reciprocal_iter<2>(abs_x);
    const sfpi::vFloat t = inv * inv;

    const sfpi::vFloat correction = PolynomialEvaluator::eval(
        t,
        5.6418323517e-01f,
        -2.8166875243e-01f,
        4.1138011217e-01f,
        -8.7715709209e-01f,
        1.8973754644e+00f,
        -3.1854724884e+00f,
        3.3371934891e+00f,
        -1.5720963478e+00f);

    const sfpi::vFloat neg_x2 = -(abs_x * abs_x);
    const sfpi::vFloat exp_value = _sfpu_exp_fp32_accurate_unsafe_(neg_x2);

    return exp_value * inv * correction;
}

#endif

// ======================================================================
// LUT-based erfc via piecewise rational P(x)/Q(x)
//
// Uses abs(x) symmetry: erfc(-x) = 2 - erfc(x)
// Fit on [0, 5.0] only, 2 segments with n4/d5 rational per segment.
// BF16 MaxULP=118 (was 128 with 3-seg n4/d4 on [-5,5])
// FP32 MaxULP≈9M  (was 1.47B)
// 18 FMAs          (was 24)
// ======================================================================

constexpr std::uint32_t ERFC_NUM_DEGREE = 4;
constexpr std::uint32_t ERFC_DEN_DEGREE = 5;
constexpr std::uint32_t ERFC_NUM_SEGMENTS = 2;
constexpr std::uint32_t ERFC_LUT_SIZE = 25;
constexpr std::array<float, ERFC_LUT_SIZE> ERFC_LUT = {{// Breakpoints
                                                        0.0000000000e+00f,
                                                        2.5000000000e+00f,
                                                        5.0000000000e+00f,
                                                        // Segment 0 [0, 2.5]: numerator (degree 4)
                                                        1.0000233650e+00f,
                                                        -1.3375675678e+00f,
                                                        6.8185544014e-01f,
                                                        -1.5691982210e-01f,
                                                        1.3746744953e-02f,
                                                        // Segment 0 [0, 2.5]: denominator (degree 5)
                                                        1.0000000000e+00f,
                                                        -2.0801517367e-01f,
                                                        4.3667086959e-01f,
                                                        -3.4568668343e-03f,
                                                        2.5104774162e-02f,
                                                        2.8375532478e-02f,
                                                        // Segment 1 [2.5, 5.0]: numerator (degree 4)
                                                        -2.5655237550e-05f,
                                                        2.1275576728e-05f,
                                                        -6.6162156145e-06f,
                                                        9.1439767402e-07f,
                                                        -4.7387182178e-08f,
                                                        // Segment 1 [2.5, 5.0]: denominator (degree 5)
                                                        1.0000000000e+00f,
                                                        -1.6457208991e-01f,
                                                        -2.0572184026e-01f,
                                                        -1.3888636231e-01f,
                                                        1.2677097321e-01f,
                                                        -2.1375391632e-02f}};

template <int ITERATIONS = 8>
inline void calculate_erfc() {
#pragma GCC unroll 1
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat ax = sfpi::min(sfpi::abs(x), 5.0f);

#ifdef INP_FLOAT32
        {
            sfpi::vFloat r = calculate_erfc_fp32_central_(ax);

            v_if(x < 0.0f) { r = 2.0f - r; }
            v_endif;

            sfpi::dst_reg[0] = r;
        }

        v_if(ax >= ERFC_FP32_ASYMPTOTIC_THRESHOLD) {
            sfpi::vFloat r = calculate_erfc_fp32_asymptotic_(ax);

            v_if(x < 0.0f) { r = 2.0f - r; }
            v_endif;

            sfpi::dst_reg[0] = r;
        }
        v_endif;
#else
        sfpi::vFloat r =
            piecewise_rational_eval<ERFC_NUM_DEGREE, ERFC_DEN_DEGREE, ERFC_NUM_SEGMENTS, ERFC_LUT_SIZE, false, true>(
                ERFC_LUT, ax);

        v_if(x < 0.0f) { r = 2.0f - r; }
        v_endif;

        sfpi::dst_reg[0] = r;
#endif

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void erfc_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
#ifdef INP_FLOAT32
    sfpu_reciprocal_init<false>();
#else
    sfpu_reciprocal_init<true>();
#endif
}

}  // namespace ckernel::sfpu
