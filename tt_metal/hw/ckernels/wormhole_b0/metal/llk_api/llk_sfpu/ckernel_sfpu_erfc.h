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

#include "ckernel_sfpu_erf.h"  // erf_appx_load_lut() -- erfc reuses erf's PWL table
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

// ======================================================================
// APPROXIMATION_MODE: erfc(x) = 1 - erf(x), evaluated through erf's own
// 6-segment SFPLUTFP32 table (see ckernel_sfpu_erf.h for the table, the
// breakpoints and the SGN_UPDATE/copysgn constraint).
//
// Max abs error 0.0148, which is the right metric here because erfc's range is
// [0, 2]. The *relative* error in the tail is another matter: erfc(3) is
// 2.2e-5 and the table's last segment returns erf = 1 exactly, so erfc reads 0
// beyond |x| ~ 3 and relative error there is 100%. That is acceptable for an
// approximate mode but wrong anywhere the tail feeds a division or a log, so
// callers wanting the tail must use the accurate path.
// ======================================================================
template <int ITERATIONS>
inline void calculate_erfc_appx() {
    sfpi::vUInt l0 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vUInt l1 = sfpi::l_reg[sfpi::LRegs::LReg1];
    sfpi::vUInt l2 = sfpi::l_reg[sfpi::LRegs::LReg2];
    sfpi::vUInt l4 = sfpi::l_reg[sfpi::LRegs::LReg4];
    sfpi::vUInt l5 = sfpi::l_reg[sfpi::LRegs::LReg5];
    sfpi::vUInt l6 = sfpi::l_reg[sfpi::LRegs::LReg6];

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat e = sfpi::copysgn(lut2_sign(x, l0, l1, l2, l4, l5, l6), x);
        // erf's segment [2.0, 3.0) reaches 1.0012 at x = 3, so 1 - erf goes
        // slightly negative there. erfc is non-negative by definition, and an
        // unclamped -0.0012 against a true 2.2e-5 is a relative error of ~57,
        // so clamp at zero -- one SFPMAX, negligible beside the accurate path.
        sfpi::dst_reg[0] = sfpi::max(1.0f - e, 0.0f);
        sfpi::dst_reg++;
    }

    sfpi::l_reg[sfpi::LRegs::LReg0] = l0;
    sfpi::l_reg[sfpi::LRegs::LReg1] = l1;
    sfpi::l_reg[sfpi::LRegs::LReg2] = l2;
    sfpi::l_reg[sfpi::LRegs::LReg4] = l4;
    sfpi::l_reg[sfpi::LRegs::LReg5] = l5;
    sfpi::l_reg[sfpi::LRegs::LReg6] = l6;
}

template <bool APPROXIMATION_MODE = false, int ITERATIONS = 8>
inline void calculate_erfc() {
    if constexpr (APPROXIMATION_MODE) {
        calculate_erfc_appx<ITERATIONS>();
        return;
    }
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        // Clamp |x| to 5.0 before evaluation (avoids extrapolation, saves one branch)
        sfpi::vFloat ax = sfpi::min(sfpi::abs(x), 5.0f);
        sfpi::vFloat r =
            // APPROX_RECIP = false: this is the accurate path. It used to pass true, which
            // was harmless while sfpu_reciprocal_iter<0> aliased iter<1>; now that <0>
            // really is seed-only (2.6 bf16 ULP) it would cost this path a ULP.
            piecewise_rational_eval<ERFC_NUM_DEGREE, ERFC_DEN_DEGREE, ERFC_NUM_SEGMENTS, ERFC_LUT_SIZE, false, false>(
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
    math::reset_counters(p_setrwc::SET_ABD_F);
    if constexpr (APPROXIMATION_MODE) {
        erf_appx_load_lut();
    } else {
        sfpu_reciprocal_init<false>();
    }
}

}  // namespace ckernel::sfpu
