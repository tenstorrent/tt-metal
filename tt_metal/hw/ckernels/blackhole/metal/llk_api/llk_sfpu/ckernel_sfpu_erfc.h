// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_converter.h"

#include "ckernel_sfpu_piecewise_rational.h"
#include "ckernel_sfpu_exp.h"

namespace ckernel::sfpu {

// ======================================================================
// LUT-based erfc via piecewise rational P(x)/Q(x)
//
// Uses abs(x) symmetry: erfc(-x) = 2 - erfc(x)
// Fit on [0, 5.0] only.
//
// BF16 path: 2 segments, n4/d5 rational per segment (unchanged).
//   BF16 MaxULP=118
//
// FP32 path: 2 segments with distinct strategies:
//   Segment 0 [0, 2.5]:   n8/d8 rational P(x)/Q(x)
//   Segment 1 [2.5, 5.0]: exponential-envelope form
//                          erfc(x) = exp(-x^2) * R(x)/S(x)
//                          where R/S is a n4/d5 rational correction.
//   FP32 MaxULP=1 (source model, validated over 1M points)
// ======================================================================

#ifdef INP_FLOAT32

// ---------- FP32-accurate path ----------

// Segment 0 [0, 2.5]: plain rational n8/d8
constexpr uint32_t ERFC_NUM_DEGREE = 8;
constexpr uint32_t ERFC_DEN_DEGREE = 8;
constexpr uint32_t ERFC_NUM_SEGMENTS = 1;
constexpr uint32_t ERFC_LUT_SIZE = 20;
constexpr std::array<float, ERFC_LUT_SIZE> ERFC_LUT = {{
    // Breakpoints
    0.0000000000e+00f, 2.5000000000e+00f,
    // Numerator (degree 8)
    9.9999999678e-01f, -7.0727174777e-01f, -1.4560369021e-01f, 1.6168460269e-01f,
    7.2433112234e-02f, -8.8993446959e-02f, 3.0758605907e-02f, -4.8153333118e-03f,
    2.9478693893e-04f,
    // Denominator (degree 8)
    1.0000000000e+00f, 4.2110744589e-01f, 3.2956417076e-01f, 1.5744189198e-01f,
    9.1642212857e-02f, 3.4669416248e-03f, 2.2650991219e-02f, -3.1036115204e-03f,
    1.9501843584e-03f}};

// Segment 1 [2.5, 5.0]: exponential-envelope correction
// erfc(x) = exp(-x^2) * R(x)/S(x)
constexpr uint32_t ERFC_ENV_NUM_DEGREE = 4;
constexpr uint32_t ERFC_ENV_DEN_DEGREE = 5;
constexpr uint32_t ERFC_ENV_LUT_SIZE = 13;
constexpr std::array<float, ERFC_ENV_LUT_SIZE> ERFC_ENV_LUT = {{
    // Breakpoints
    2.5000000000e+00f, 5.0000000000e+00f,
    // Correction numerator (degree 4)
    9.8776553254e-01f, 4.8952836324e-01f, -3.2467559064e-01f,
    -3.2678127386e-01f, -1.7765040407e-01f,
    // Correction denominator (degree 5)
    1.0000000000e+00f, 1.5558549987e+00f, 5.8819271588e-01f,
    -7.3394549714e-01f, -5.7914226616e-01f, -3.1487883838e-01f}};

template <int ITERATIONS = 8>
inline void calculate_erfc() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat ax = sfpi::min(sfpi::abs(x), 5.0f);

        // Segment 0 [0, 2.5]: plain rational
        sfpi::vFloat r =
            piecewise_rational_eval<ERFC_NUM_DEGREE, ERFC_DEN_DEGREE, ERFC_NUM_SEGMENTS, ERFC_LUT_SIZE, false, false>(
                ERFC_LUT, ax);

        // Segment 1 [2.5, 5.0]: exponential-envelope form
        // erfc(x) = exp(-x^2) * correction(x)
        v_if(ax > 2.5f) {
            sfpi::vFloat neg_x2 = -(ax * ax);
            sfpi::vFloat exp_neg_x2 = _sfpu_exp_fp32_accurate_(neg_x2);
            sfpi::vFloat correction =
                piecewise_rational_eval<ERFC_ENV_NUM_DEGREE, ERFC_ENV_DEN_DEGREE, 1, ERFC_ENV_LUT_SIZE, false, false>(
                    ERFC_ENV_LUT, ax);
            r = exp_neg_x2 * correction;
        }
        v_endif;

        // erfc(-x) = 2 - erfc(x)
        v_if(x < 0.0f) { r = 2.0f - r; }
        v_endif;

        sfpi::dst_reg[0] = r;
        sfpi::dst_reg++;
    }
}

#else

// ---------- BF16 path (unchanged) ----------

constexpr uint32_t ERFC_NUM_DEGREE = 4;
constexpr uint32_t ERFC_DEN_DEGREE = 5;
constexpr uint32_t ERFC_NUM_SEGMENTS = 2;
constexpr uint32_t ERFC_LUT_SIZE = 25;
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
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        // Clamp |x| to 5.0 before evaluation (avoids extrapolation, saves one branch)
        sfpi::vFloat ax = sfpi::min(sfpi::abs(x), 5.0f);
        sfpi::vFloat r =
            piecewise_rational_eval<ERFC_NUM_DEGREE, ERFC_DEN_DEGREE, ERFC_NUM_SEGMENTS, ERFC_LUT_SIZE, false, true>(
                ERFC_LUT, ax);
        // erfc(-x) = 2 - erfc(x)
        v_if(x < 0.0f) { r = 2.0f - r; }
        v_endif;
        sfpi::dst_reg[0] = r;
        sfpi::dst_reg++;
    }
}

#endif

template <bool APPROXIMATION_MODE>
void erfc_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<true>();
}

}  // namespace ckernel::sfpu
