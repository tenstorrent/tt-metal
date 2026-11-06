// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "ckernel_sfpu_converter.h"
#include "ckernel_sfpu_exp.h"

namespace ckernel::sfpu {

sfpi_inline sfpi::vFloat _sfpu_neg_exp_f32_(sfpi::vFloat val) {
    sfpi::vFloat result = 0.0f;

    constexpr float UNDERFLOW_THRESHOLD = -126.5f;

    // Step 1: Compute k = round(x / ln(2))
    // z = x / ln(2) = x * (1/ln(2))
    sfpi::vFloat z = val * sfpi::vConstFloatPrgm0;

    // Clamp z to -126.5: exp(x) underflows to 0 for large negative x
    z = sfpi::max(z, UNDERFLOW_THRESHOLD);

    // Round z to nearest integer using round-to-nearest
    sfpi::vInt k_int;
    sfpi::vFloat k = _sfpu_round_to_nearest_int32_(z, k_int);

    // Step 2: Cody-Waite range reduction
    // Compute r = x - k*ln(2) in extended precision
    // r = x - k*LN2_HI - k*LN2_LO
    // This provides better accuracy than simple r = x - k*ln(2)
    // Cody-Waite constants: ln(2) split into high and low parts for extended precision.
    // LN2_HI is chosen so that k*LN2_HI can be computed exactly for integer k in the valid range.
    // LN2_LO contains the remainder: LN2_HI + LN2_LO ≈ -ln(2)

    // We want to do:
    // 1) r_hi = val - k * LN2_HI
    // 2) r = r_hi - k * LN2_LO
    // On Wormhole, we can only do VD = VA * VB + VC, so we need to transform the expressions to
    // ensure optimization to a single SFPMAD instruction.
    // On Blackhole, SFFPMAD has SFPMAD_MOD1_NEGATE_VA and SFPMAD_MOD1_NEGATE_VC for this purpose.
    // However, negating constants maintains consistency with Wormhole, and ensures higher chance
    // of optimization to a single SFPMAD instruction.
    // The transformation is as follows:
    // 1) r_hi = val + k * (-LN2_HI)
    // 2) r = r_hi + k * (-LN2_LO)
    // Where LN2_HI and LN2_LO are negated.
    // This way, compiler can more easily optimize this expression to a single SFPMAD instruction.
    constexpr float LN2_HI = -0.6931152343750000f;  // High bits of ln(2)
    constexpr float LN2_LO = -3.19461832987e-05f;   // Low bits of ln(2)

    // First subtract k * LN2_HI
    sfpi::vFloat r_hi = k * LN2_HI + val;

    // Then subtract k * LN2_LO
    sfpi::vFloat r = k * LN2_LO + r_hi;

    // Step 3: Polynomial approximation for exp(r) using Taylor series
    // exp(r) ~= 1 + r + r²/2! + r³/3! + r⁴/4! + r⁵/5! + r⁶/6! + r⁷/7!
    // Use 7th order polynomial (Taylor series coefficients) for < 1 ULP accuracy
    // Coefficients in ascending order of powers: c0, c1, c2, c3, c4, c5, c6, c7
    sfpi::vFloat p = PolynomialEvaluator::eval(
        r,
        1.0f,           // c0 = 1
        1.0f,           // c1 = 1
        0.5f,           // c2 = 1/2!
        1.0f / 6.0f,    // c3 = 1/3!
        1.0f / 24.0f,   // c4 = 1/4!
        1.0f / 120.0f,  // c5 = 1/5!
        1.0f / 720.0f,  // c6 = 1/6!
        1.0f / 5040.0f  // c7 = 1/7!
    );

    // Step 4: Scale by 2^k using exponent manipulation
    // ldexp(p, k_int) = p * 2^k
    // We do this by adding k_int to the exponent of p
    // Get the biased exponent of p
    sfpi::vInt p_exp = sfpi::exexp(p, sfpi::ExponentMode::Biased);
    // Add k_int to get the new exponent
    sfpi::vInt new_exp = p_exp + k_int;

    // Set the new exponent.
    //
    // Guard against exponent underflow.  The clamp on z above stops the Cody-Waite
    // range reduction short for very negative inputs, so p keeps a small exponent
    // while k_int stays pinned near the clamp, which leaves new_exp <= 0.  setexp
    // writes only the low bits of new_exp into the exponent field, so a negative
    // new_exp wraps to a large positive exponent and this helper returns a huge
    // value or Inf where the true result underflows to zero.  result is already
    // initialised to 0.0f, which is the correct underflow answer.
    //
    // This mirrors the guard in _sfpu_exp_fp32_accurate_ (ckernel_sfpu_exp.h).
    // Only the underflow half is needed here: this helper is called exclusively
    // with negative inputs, so exp(val) < 1 and the exponent cannot overflow.
    v_if(new_exp >= 1) { result = sfpi::setexp(p, new_exp); }
    v_endif;

    return result;
}

// Evaluates (exp(r) - 1 - r) / r^2 = sum_{k=2}^{11} r^{k-2} / k! in Sterbenz-compensated
// Horner form for |r| <= 0.5, returning the high part and writing the rounding residual
// to y_lo. On |r| <= 0.5 each compensated step satisfies |r * y_{k+1}| < c_k / 2, so
// c_k - y_k is exact by Sterbenz's lemma and a single SFPMAD recovers the FMA residual.
sfpi_inline sfpi::vFloat _sfpu_expm1_m_x_over_x2_hi_lo_(sfpi::vFloat r, sfpi::vFloat& y_lo) {
    constexpr float C11 = 1.0f / 39916800.0f;
    constexpr float C10 = 1.0f / 3628800.0f;
    constexpr float C9 = 1.0f / 362880.0f;
    constexpr float C8 = 1.0f / 40320.0f;
    constexpr float C7_HI = 1.0f / 5040.0f;
    constexpr float C7_LO = -2.7255968749e-12f;
    constexpr float C6_HI = 1.0f / 720.0f;
    constexpr float C6_LO = -3.3631094437e-11f;
    constexpr float C5_HI = 1.0f / 120.0f;
    constexpr float C5_LO = -4.3461720334e-10f;
    constexpr float C4_HI = 1.0f / 24.0f;
    constexpr float C4_LO = -1.2417634698e-9f;
    constexpr float C3_HI = 1.0f / 6.0f;
    constexpr float C3_LO = -4.9670538793e-9f;
    constexpr float C2 = 0.5f;

    sfpi::vFloat y = r * C11 + C10;
    y = r * y + C9;
    y = r * y + C8;

    sfpi::vFloat y_next = r * y + C7_HI;
    y_lo = (r * y + (C7_HI - y_next)) + C7_LO;
    y = y_next;

    y_next = r * y + C6_HI;
    sfpi::vFloat err = (r * y + (C6_HI - y_next)) + C6_LO;
    y_lo = r * y_lo + err;
    y = y_next;

    y_next = r * y + C5_HI;
    err = (r * y + (C5_HI - y_next)) + C5_LO;
    y_lo = r * y_lo + err;
    y = y_next;

    y_next = r * y + C4_HI;
    err = (r * y + (C4_HI - y_next)) + C4_LO;
    y_lo = r * y_lo + err;
    y = y_next;

    y_next = r * y + C3_HI;
    err = (r * y + (C3_HI - y_next)) + C3_LO;
    y_lo = r * y_lo + err;
    y = y_next;

    y_next = r * y + C2;
    err = r * y + (C2 - y_next);
    y_lo = r * y_lo + err;
    return y_next;
}

// mul_a * mul_b + addend (MAD)
template <bool is_fp32_dest_acc_en>
sfpi_inline void _xielu_mad_(sfpi::vFloat mul_a, sfpi::vFloat mul_b, sfpi::vFloat addend) {
    sfpi::vFloat result = mul_a * mul_b + addend;
    if constexpr (!is_fp32_dest_acc_en) {
        result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
    }
    sfpi::dst_reg[0] = result;
}

/*
 * This function implements the xIELU (Expanded Integral of ELU) activation function,
 * a trainable piecewise function with learnable alpha_p and alpha_n parameters.
 * based on "Deriving Activation Functions Using Integration"
 *
 * @see "Deriving Activation Functions Using Integration" (https://arxiv.org/abs/2411.13010)
 *
 * @param x : The input tensor
 * @param alpha_p : The positive alpha parameter
 * @param alpha_n : The negative alpha parameter
 *
 * Positive input
 * if x > 0 :  alpha_p * x * x + beta * x
 * Negative input
 * if x < 0 : alpha_n * expm1(minimum(x, eps)) - alpha_n * x + beta * x
 *        --> alpha_n * (expm1(minimum(x, eps)) - x) + beta * x
 */
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_xielu(const uint32_t param0, const uint32_t param1) {
    if constexpr (is_fp32_dest_acc_en) {
        // In float32 mode, alpha_n * (expm1(x) - x) and 0.5 * x cancel at the negative-branch
        // zero crossing x* (e.g. x* ~= -2.4325 for alpha_n = 0.8, x* ~= -1.5936 for alpha_n = 1.0).
        // Evaluate expm1(x) - x and the final product in compensated hi+lo form so the crossing
        // stays faithful (<= 2 fp32 ULP) while keeping <= 6 LRegs live across the Horner polynomial.
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat x = sfpi::dst_reg[0];
            v_if(x > 0.0f) {
                sfpi::vFloat alpha_p = Converter::as_float(param0);
                _xielu_mad_<true>(alpha_p * x, x, 0.5f * x);
            }
            v_elseif(x >= sfpi::vConstFloatPrgm1) {
                sfpi::vFloat exp_term = sfpi::vConstFloatPrgm2 - x;
                _xielu_mad_<true>(Converter::as_float(param1), exp_term, 0.5f * x);
            }
            v_elseif(x > -0.5f) {
                sfpi::vFloat y_lo;
                sfpi::vFloat y_hi = _sfpu_expm1_m_x_over_x2_hi_lo_(x, y_lo);
                sfpi::vFloat x2 = x * x;
                sfpi::vFloat neg_x2 = -x2;
                sfpi::vFloat x2_err = x * x + neg_x2;
                sfpi::vFloat p_hi = x2 * y_hi;
                sfpi::vFloat neg_p_hi = -p_hi;
                sfpi::vFloat p_err = x2 * y_hi + neg_p_hi;
                sfpi::vFloat p_lo = x2 * y_lo + (x2_err * y_hi + p_err);

                sfpi::vFloat alpha_n = Converter::as_float(param1);
                sfpi::vFloat t_hi = p_hi * alpha_n;
                sfpi::vFloat neg_t_hi = -t_hi;
                sfpi::vFloat t_err = p_hi * alpha_n + neg_t_hi;
                sfpi::vFloat res_hi = 0.5f * x + t_hi;
                sfpi::dst_reg[0] = res_hi + (p_lo * alpha_n + t_err);
            }
            v_elseif(x > -20.0f) {
                // 0x4B400000 = 1.5 * 2^23 rounds x / ln(2) to nearest integer k; 0x4B3FFF81
                // subtracts the magic bias and adds the IEEE-754 exponent bias 127 in one ISUB.
                constexpr float C231 = 12582912.0f;
                constexpr int C231_BIAS = 0x4B3FFF81;
                constexpr float LN2_HI = -6.93145751953125e-1f;
                constexpr float LN2_LO = -1.428606765330187e-6f;
                constexpr float LN2_LO2 = -5.495603971894525e-14f;

                sfpi::vFloat k = (x * sfpi::vConstFloatPrgm0 + C231) - C231;
                sfpi::vFloat r = k * LN2_LO + (k * LN2_HI + x);

                sfpi::vFloat y_lo;
                sfpi::vFloat y2 = _sfpu_expm1_m_x_over_x2_hi_lo_(r, y_lo);

                sfpi::vFloat y1 = r * y2 + 1.0f;
                sfpi::vFloat e1 = r * y2 + (1.0f - y1);
                y_lo = r * y_lo + e1;

                sfpi::vFloat p_hi = r * y1 + 1.0f;
                sfpi::vFloat e0 = r * y1 + (1.0f - p_hi);
                sfpi::vFloat p_lo = r * y_lo + e0;

                // Reload x from Dest rather than holding it live across the Horner sequence.
                sfpi::vFloat x_reload = sfpi::dst_reg[0];
                sfpi::vFloat r_hi = k * LN2_HI + x_reload;
                sfpi::vFloat r_lo = k * LN2_LO2 + (k * LN2_LO + (r_hi - r));
                p_lo = r_lo * p_hi + p_lo;

                sfpi::vFloat tmp = k + C231;
                sfpi::vFloat two_k = sfpi::setexp(1.0f, sfpi::as<sfpi::vInt>(tmp) - C231_BIAS);
                sfpi::vFloat e_hi = p_hi * two_k;
                sfpi::vFloat e_lo = p_lo * two_k;

                sfpi::vFloat em1_hi = e_hi - 1.0f;
                sfpi::vFloat em1_lo = (e_hi - (em1_hi + 1.0f)) + e_lo;

                sfpi::vFloat ex_hi = em1_hi - x_reload;
                sfpi::vFloat ex_lo = (em1_hi - (ex_hi + x_reload)) + em1_lo;

                sfpi::vFloat alpha_n = Converter::as_float(param1);
                sfpi::vFloat t_hi = ex_hi * alpha_n;
                sfpi::vFloat neg_t_hi = -t_hi;
                sfpi::vFloat t_err = ex_hi * alpha_n + neg_t_hi;
                sfpi::vFloat res_hi = 0.5f * x_reload + t_hi;
                sfpi::dst_reg[0] = res_hi + (ex_lo * alpha_n + t_err);
            }
            v_else {
                // For x <= -20, exp(x) < 2.06e-9 < 2^-33 * |1 - x|, so expm1(x) == -1 in float32.
                sfpi::vFloat exp_term = -1.0f - x;
                _xielu_mad_<true>(Converter::as_float(param1), exp_term, 0.5f * x);
            }
            v_endif;
            sfpi::dst_reg++;
        }
        return;
    }
    // Quasar deviation: alpha_p, alpha_n and beta * x are materialised in each branch rather than held
    // across the loop. Held live beside the _sfpu_neg_exp_f32_ temporaries they exceed the LReg file at
    // -O2 (the metal JIT's level; -O3 happened to fit), a hard "too few lregs" compile error.
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        v_if(x > 0.0f) {  // positive
            sfpi::vFloat alpha_p = Converter::as_float(param0);
            _xielu_mad_<is_fp32_dest_acc_en>(alpha_p * x, x, 0.5f * x);
        }
        v_elseif(x >= sfpi::vConstFloatPrgm1) {  // very small negative
            sfpi::vFloat exp_term = sfpi::vConstFloatPrgm2 - x;
            _xielu_mad_<is_fp32_dest_acc_en>(Converter::as_float(param1), exp_term, 0.5f * x);
        }
        v_elseif(x > -0.5f) {  // moderate negative region
            // For small x >- 0.5: use Taylor series to avoid cancellation
            // Use a polynomial approximation around 0 to avoid catastrophic cancellation
            // Polynomial coefficients found using Sollya with the following commands:
            // > fpminimax(exp(x)-1, [|1,2,3,4,5,6,7|], [|single...|], [-0.5; -2^(-40)] + [2^(-40); 0.5], relative);
            // expm1(x) = 0 + x + c2*x^2 + c3*x^3 + ... + c7*x^7
            // Hence, expm1(x)-x = c2*x^2 + ... + c7*x^7 = x^2 * (c2 + c3*x + c4*x^2 + ... + c7*x^5)
            sfpi::vFloat exp_term = x * x *
                                    PolynomialEvaluator::eval(
                                        x,
                                        0.500000059604644775390625f,
                                        0.16666667163372039794921875f,
                                        4.16650883853435516357421875e-2f,
                                        8.333188481628894805908203125e-3f,
                                        1.400390756316483020782470703125e-3f,
                                        1.99588379473425447940826416015625e-4f);
            _xielu_mad_<is_fp32_dest_acc_en>(Converter::as_float(param1), exp_term, 0.5f * x);
        }
        v_else {  // large negative
            sfpi::vFloat exp_term = _sfpu_neg_exp_f32_(x) - 1.0f - x;
            _xielu_mad_<is_fp32_dest_acc_en>(Converter::as_float(param1), exp_term, 0.5f * x);
        }
        v_endif;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void xielu_init() {
    math::_reset_counters_<p_setrwc::SET_ABD_F>();
    sfpi::vConstFloatPrgm0 = 1.4426950408889634f;   // 1/ln(2)
    sfpi::vConstFloatPrgm1 = -1e-6f;                // eps value
    sfpi::vConstFloatPrgm2 = -0.0000009999995427f;  // expm1(eps)
}

}  // namespace ckernel::sfpu
