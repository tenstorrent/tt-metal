// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"

#include "ckernel_sfpu_piecewise_rational.h"

namespace ckernel::sfpu {

// ======================================================================
// LUT-based erfc via piecewise rational P(x)/Q(x)
//
// Uses abs(x) symmetry: erfc(-x) = 2 - erfc(x)
// Fit on [0, 5.0] only, 2 segments with n4/d5 rational per segment.
// BF16 MaxULP=118 (was 128 with 3-seg n4/d4 on [-5,5])
// FP32 MaxULP≈9M  (was 1.47B)
// 18 FMAs          (was 24)
// ======================================================================

constexpr uint32_t ERFC_NUM_DEGREE = 4;
constexpr uint32_t ERFC_DEN_DEGREE = 5;
constexpr uint32_t ERFC_NUM_SEGMENTS = 2;
constexpr uint32_t ERFC_LUT_SIZE = 25;
constexpr std::array<float, 25> ERFC_LUT = {{// Breakpoints
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
        sfpi::vFloat ax = sfpi::setsgn(x, 0);
        sfpi::vFloat limit = 5.0f;
        sfpi::vec_min_max(ax, limit);
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

template <bool APPROXIMATION_MODE>
void erfc_init() {
    sfpu_reciprocal_init<true>();
}

}  // namespace ckernel::sfpu
