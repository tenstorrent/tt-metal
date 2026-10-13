// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"  // math::reset_counters, p_setrwc

#include "ckernel_sfpu_exp.h"  // For _sfpu_round_to_nearest_int32_
#include "sfpu/ckernel_sfpu_polyval.h"
#include "ckernel_sfpu_recip.h"
#include "sfpu/ckernel_sfpu_cdf.h"
#include "sfpu/ckernel_sfpu_load_config.h"
#include "ckernel_sfpu_erf.h"  // ERF_LUT, ERF_NUM_DEGREE, ERF_DEN_DEGREE (INP_FLOAT32 branch for FP32 path)
#include "ckernel_sfpu_piecewise_rational.h"
#include "ckernel_sfpu_tanh.h"  // _sfpu_tanh_fp32_accurate_ for gelu_tanh
#include "sfpi.h"

namespace ckernel::sfpu {

// =============================================================================
// Fused x * exp(t) for GELU negative tail - avoids intermediate underflow
// =============================================================================
sfpi_inline sfpi::vFloat x_times_exp_negative_tail(sfpi::vFloat x, sfpi::vFloat t) {
    // Cody-Waite constants for extended precision range reduction
    constexpr float INV_LN2 = 1.4426950408889634f;  // 1/ln(2)
    constexpr float LN2_HI = -0.6931152343750000f;  // -ln(2) high bits (exact in float)
    constexpr float LN2_LO = -3.19461832987e-05f;   // -ln(2) low bits

    // Step 1: Range reduction - split t into k*ln(2) + r
    sfpi::vFloat z = t * INV_LN2;
    sfpi::vInt k_int;
    sfpi::vFloat k = _sfpu_round_to_nearest_int32_(z, k_int);

    // Step 2: Extended precision range reduction (Cody-Waite)
    sfpi::vFloat r = k * LN2_HI + t;
    r = k * LN2_LO + r;

    // Step 3: Polynomial approximation of exp(r) for |r| < 0.5
    constexpr float C2 = 0.5f;
    constexpr float C3 = 0.166666667f;
    constexpr float C4 = 0.0416666667f;
    constexpr float C5 = 0.00833333333f;
    sfpi::vFloat poly = PolynomialEvaluator::eval(r, 1.0f, 1.0f, C2, C3, C4, C5);

    // Step 4: FUSED MULTIPLY - multiply by x BEFORE exponent shift!
    sfpi::vFloat x_poly = x * poly;  // x * poly ≈ -9 to -13 (safe range)

    // Step 5: Exponent bit manipulation on the FUSED result
    sfpi::vInt xpoly_exp = sfpi::exexp(x_poly, sfpi::ExponentMode::Biased);  // Extract exponent of x*poly
    sfpi::vInt new_exp = xpoly_exp + k_int;                                  // Shift by 2^k

    // Step 6: FTZ check on FINAL result (x * exp(t)), not intermediate exp(t)
    sfpi::vFloat result = 0.0f;
    v_if(new_exp > 0) { result = sfpi::setexp(x_poly, new_exp); }
    v_endif;

    return result;
}

// =============================================================================
// Forward GELU - Piecewise CDF Approximation
// =============================================================================

// Degree-13 CDF polynomial for Phi(x) over [-3.125, 2.78125]
constexpr float GELU_CDF_CORE_C0 = 5.000000000e-01f;
constexpr float GELU_CDF_CORE_C1 = 3.9894227818e-01f;
constexpr float GELU_CDF_CORE_C3 = -6.6361041488e-02f;
constexpr float GELU_CDF_CORE_C5 = 9.7720050615e-03f;
constexpr float GELU_CDF_CORE_C7 = -1.0717806322e-03f;
constexpr float GELU_CDF_CORE_C9 = 8.1812159812e-05f;
constexpr float GELU_CDF_CORE_C11 = -3.8082057209e-06f;
constexpr float GELU_CDF_CORE_C13 = 7.9821413868e-08f;

// Degree-3 H-form correction polynomial for the exp-based region (-5.54259443, -3.125].
constexpr float GELU_HCORR_C0 = 3.0369991064e-01f;
constexpr float GELU_HCORR_C1 = 9.5413386822e-02f;
constexpr float GELU_HCORR_C2 = 1.3809983619e-02f;
constexpr float GELU_HCORR_C3 = 7.5950479368e-04f;

// Forward GELU Evaluation with CDF Polynomial Approximation
// GELU(x) = x * Phi(x) where Phi is approximated piecewise
sfpi_inline sfpi::vFloat calculate_gelu_piecewise(sfpi::vFloat x) {
    sfpi::vFloat result = 0.0f;  // Default: 0 for x <= -5.54259443 (torch saturation)
    sfpi::vFloat x2 = x * x;

    // NaN propagation: GELU(NaN) must be NaN, not 0.
    // The default result is 0, but any comparison with NaN is false,
    // so without this guard NaN would silently become 0.
    v_if(sfpi::is_nan(x)) {
        result = x;
    } v_elseif(x > -5.54259443f) {
        // Shared smooth H ≈ ½·erfc(|x|/√2) via Moroz exp_21f · corr_H.
        constexpr float NEG_HALF_ONE_LN2 = -0.72134752044f;  // -0.5 / ln(2)
        sfpi::vFloat xlog2 = x2 * NEG_HALF_ONE_LN2 + 127.0f;

        sfpi::vInt z = _float_to_int32_for_exp_21f_(xlog2);
        sfpi::vInt exponential_part = sfpi::exexp(sfpi::as<sfpi::vFloat>(z), sfpi::ExponentMode::Biased);
        sfpi::vMag fractional_part = sfpi::exman(sfpi::as<sfpi::vFloat>(z));

        sfpi::vFloat frac = sfpi::convert<sfpi::vFloat>(fractional_part, sfpi::RoundMode::Nearest);
        frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f);
        sfpi::vFloat exp_val = sfpi::setexp(frac, exponential_part);

        sfpi::vFloat H =
            exp_val * PolynomialEvaluator::eval(x, GELU_HCORR_C0, GELU_HCORR_C1, GELU_HCORR_C2, GELU_HCORR_C3);

        // Exp region (-5.54259443, -3.125]: GELU = x · round_to_grid(H).
        constexpr float ROUND_TO_GRID = 0.375f;
        sfpi::vFloat Hs = (H + ROUND_TO_GRID) - ROUND_TO_GRID;
        result = x * Hs;

        // Core CDF region [-3.125, 2.78125): GELU = x · Phi_core(x).
        v_and(x >= -3.125f);
        sfpi::vFloat odd_poly = PolynomialEvaluator::eval(
            x2,
            GELU_CDF_CORE_C1,
            GELU_CDF_CORE_C3,
            GELU_CDF_CORE_C5,
            GELU_CDF_CORE_C7,
            GELU_CDF_CORE_C9,
            GELU_CDF_CORE_C11,
            GELU_CDF_CORE_C13);
        sfpi::vFloat phi = GELU_CDF_CORE_C0 + x * odd_poly;
        result = x * phi;

        // Identity region x >= 2.78125.
        v_and(x >= 2.78125f);
        result = x;
    }
    v_endif;

    return result;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void gelu_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    if constexpr (APPROXIMATION_MODE) {
        sfpi::vConstFloatPrgm0 = 0.5f;

        sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vLut16ss(0.1928f, 0.4939f);
        sfpi::l_reg[sfpi::LRegs::LReg4] = sfpi::vLut16ii(-0.00010443f, -0.1604f);

        sfpi::l_reg[sfpi::LRegs::LReg1] = sfpi::vLut16ss(0.6188f, 0.6099f);
        sfpi::l_reg[sfpi::LRegs::LReg5] = sfpi::vLut16ii(-0.2795f, -0.2635f);

        sfpi::l_reg[sfpi::LRegs::LReg2] = sfpi::vLut16ss(0.5402f, 0.5f);
        sfpi::l_reg[sfpi::LRegs::LReg6] = sfpi::vLut16ii(-0.1194f, 0.0f);
    } else if constexpr (is_fp32_dest_acc_en) {
        sfpu_reciprocal_init<false>();
    }
}

template <int ITERATIONS>
inline void calculate_gelu_appx() {
    sfpi::vLut16ss s01 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vLut16ss s23 = sfpi::l_reg[sfpi::LRegs::LReg1];
    sfpi::vLut16ss s45 = sfpi::l_reg[sfpi::LRegs::LReg2];
    sfpi::vLut16ii i01 = sfpi::l_reg[sfpi::LRegs::LReg4];
    sfpi::vLut16ii i23 = sfpi::l_reg[sfpi::LRegs::LReg5];
    sfpi::vLut16ii i45 = sfpi::l_reg[sfpi::LRegs::LReg6];

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat half = sfpi::vConstFloatPrgm0;
        sfpi::vFloat half_in = in * half;
        sfpi::vFloat result = sfpi::lut(in, s01, i01, s23, i23, s45, i45, sfpi::LutSign::Update);
        result = half_in + result;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }

    sfpi::l_reg[sfpi::LRegs::LReg0] = s01;
    sfpi::l_reg[sfpi::LRegs::LReg1] = s23;
    sfpi::l_reg[sfpi::LRegs::LReg2] = s45;
    sfpi::l_reg[sfpi::LRegs::LReg4] = i01;
    sfpi::l_reg[sfpi::LRegs::LReg5] = i23;
    sfpi::l_reg[sfpi::LRegs::LReg6] = i45;
}

constexpr float GELU_ERF_NUM[17] = {
    0.0f,
    1.1283791065e+00f,
    0.0f,
    2.1477432549e-01f,
    0.0f,
    6.2133435160e-02f,
    0.0f,
    5.6230435148e-03f,
    0.0f,
    6.1307044234e-04f,
    0.0f,
    1.7678321456e-05f,
    0.0f,
    2.7384647439e-08f,
    0.0f,
    -2.8632063387e-10f,
    0.0f};
constexpr float GELU_ERF_DEN[17] = {
    1.0f,
    0.0f,
    5.2367275953e-01f,
    0.0f,
    1.2961706519e-01f,
    0.0f,
    1.9642570987e-02f,
    0.0f,
    1.9545555115e-03f,
    0.0f,
    1.3179056987e-04f,
    0.0f,
    1.3156344494e-06f,
    0.0f,
    -3.5153888689e-09f,
    0.0f,
    -6.7350725691e-12f};

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_gelu() {
    if constexpr (APPROXIMATION_MODE) {
        calculate_gelu_appx<ITERATIONS>();
    } else if constexpr (is_fp32_dest_acc_en) {
        constexpr float INV_SQRT2 = 0.7071067811865475f;
        constexpr float GELU_SAT = -5.54259443f;
#pragma GCC unroll 0
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat x = sfpi::dst_reg[0];
            sfpi::vFloat result = 0.0f;
            v_if(x > GELU_SAT) {
                sfpi::vFloat scaled = x * INV_SQRT2;
                scaled = sfpi::min(scaled, 10.0f);
                sfpi::vFloat x2 = scaled * scaled;
                sfpi::vFloat erf_n, erf_d;
                piecewise_rational_eval_parity_numer_denom<16, 16>(
                    GELU_ERF_NUM, GELU_ERF_DEN, scaled, x2, erf_n, erf_d);
                sfpi::vFloat erf_val = erf_n * sfpu_reciprocal<false>(erf_d);
                erf_val = sfpi::clamp(erf_val, -1.0f, 1.0f);
                result = x * (0.5f + 0.5f * erf_val);
                constexpr float HALF_ULP_AT_1 = 2.9802322387695313e-08f;
                v_if(erf_val == -1.0f) { result = x * HALF_ULP_AT_1; }
                v_endif;
            }
            v_endif;
            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
    } else {
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat in = sfpi::dst_reg[0];
            sfpi::vFloat result = calculate_gelu_piecewise(in);
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
    }
}

template <bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_gelu_tanh() {
    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    constexpr float GELU_TANH_K = 0.044715f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        sfpi::vFloat x2 = x * x;
        sfpi::vFloat p = GELU_TANH_K * x2 + 1.0f;
        sfpi::vFloat q = x * p;
        sfpi::vFloat u = SQRT_2_OVER_PI * q;
        sfpi::vFloat t = _sfpu_tanh_fp32_accurate_(u);

        x = sfpi::dst_reg[0];
        sfpi::vFloat result = sfpi::copysgn(sfpi::vFloat(0.0f), x);

        sfpi::vFloat half_x = 0.5f * x;
        result = half_x * t + half_x;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

inline void gelu_tanh_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    tanh_init<false, true>();
}

constexpr float GELU_DERIV_H0 = 7.9788309336e-01f;
constexpr float GELU_DERIV_H1 = -2.6593554020e-01f;
constexpr float GELU_DERIV_H2 = 5.9766173363e-02f;
constexpr float GELU_DERIV_H3 = -9.4142099842e-03f;
constexpr float GELU_DERIV_H4 = 1.1061388068e-03f;
constexpr float GELU_DERIV_H5 = -9.7448595625e-05f;
constexpr float GELU_DERIV_H6 = 6.0921474869e-06f;
constexpr float GELU_DERIV_H7 = -2.3764030743e-07f;
constexpr float GELU_DERIV_H8 = 4.2734988881e-09f;

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat calculate_gelu_derivative_simple(sfpi::vFloat x) {
    sfpi::vFloat result = 0.0f;

    v_if(x >= 3.1719f) { result = 1.0f; }
    v_elseif(x >= -3.0f) {
        sfpi::vFloat u = x * x;
        sfpi::vFloat h = PolynomialEvaluator::eval(
            u,
            GELU_DERIV_H0,
            GELU_DERIV_H1,
            GELU_DERIV_H2,
            GELU_DERIV_H3,
            GELU_DERIV_H4,
            GELU_DERIV_H5,
            GELU_DERIV_H6,
            GELU_DERIV_H7,
            GELU_DERIV_H8);
        result = 0.5f + x * h;
    }
    v_elseif(x > -13.375f) {
        constexpr float INV_SQRT_2PI = 0.3989422804014327f;

        sfpi::vFloat x2 = x * x;
        sfpi::vFloat t = x2 * (-0.5f);

        sfpi::vFloat x_exp = x_times_exp_negative_tail(x, t);

        if constexpr (APPROXIMATION_MODE) {
            result = x_exp * INV_SQRT_2PI;
        } else {
            sfpi::vFloat inv_x2 = sfpu_reciprocal_iter<is_fp32_dest_acc_en ? 2 : 1>(x2);
            sfpi::vFloat inv_x4 = inv_x2 * inv_x2;
            sfpi::vFloat correction = 1.0f - inv_x2 + inv_x4;
            result = x_exp * INV_SQRT_2PI * correction;
        }
    }
    v_endif;

    return result;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_gelu_derivative_polynomial() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat result = calculate_gelu_derivative_simple<APPROXIMATION_MODE, is_fp32_dest_acc_en>(val);
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void gelu_derivative_polynomial_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    if constexpr (!APPROXIMATION_MODE) {
        sfpu_reciprocal_init<false>();
    }
}

}  // namespace ckernel::sfpu
