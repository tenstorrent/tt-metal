// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_sfpu_converter.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_polyval.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// Softplus via abs(x) symmetry (ported from Blackhole): with f(a) = ln(1+exp(-a)),
// softplus(t) = t + f(t) for t >= 0 and f(-t) for t < 0. is_fp32_dest_acc_en selects a degree-8 poly on
// [0,5] + exp Taylor tail (32-bit Dest) vs the bf16 evaluation in u = exp(-a) (16-bit Dest).
// The bf16 path is one vector per iteration: the two-vector ILP interleave Wormhole and
// Blackhole use has not been applied (or measured) on Quasar.

constexpr float SOFTPLUS_POLY_BOUNDARY = 5.0f;

// FP32 residual polynomial: f(a) = ln(1+exp(-a)) on [0, 5], degree 8
constexpr float SOFTPLUS_POLY_C0 = 6.9310557842e-01f;
constexpr float SOFTPLUS_POLY_C1 = -4.9926245213e-01f;
constexpr float SOFTPLUS_POLY_C2 = 1.2186349183e-01f;
constexpr float SOFTPLUS_POLY_C3 = 5.6753782555e-03f;
constexpr float SOFTPLUS_POLY_C4 = -1.0528374463e-02f;
constexpr float SOFTPLUS_POLY_C5 = 2.7290175203e-03f;
constexpr float SOFTPLUS_POLY_C6 = -3.4358495031e-04f;
constexpr float SOFTPLUS_POLY_C7 = 2.1285692128e-05f;
constexpr float SOFTPLUS_POLY_C8 = -4.8245715334e-07f;

// BF16: same coefficients as the Blackhole/Wormhole kernel. -1/ln2, P1 and H1 live in the
// programmable constant registers (loaded by softplus_init); the rest are fp16-exact.
// EXP_BIAS is the fp32 exponent bias y is offset by so k = trunc(y) is the biased exponent of u.
// p(f) ~ 2^f on [0, 1) is pinned to p(0) = 1 and p(1) = 2 - 2^-16 so it stays in [1, 2).
// h(u) = (ln(1+u)/u - 1)/u on [0, 1], degree 4, h(0) = -1/2 pinned.
constexpr float SOFTPLUS_BF16_NEG_ONE_LN2 = -1.4426950216293334961f;
constexpr float SOFTPLUS_BF16_EXP_BIAS = 127.0f;
constexpr float SOFTPLUS_BF16_P0 = 1.0f;
constexpr float SOFTPLUS_BF16_P1 = 0.6954193115234375f;
constexpr float SOFTPLUS_BF16_P2 = 0.2264404296875f;  // 1855 * 2^-13
constexpr float SOFTPLUS_BF16_P3 = 0.078125f;         // 5 * 2^-6
constexpr float SOFTPLUS_BF16_H0 = -0.5f;
constexpr float SOFTPLUS_BF16_H1 = 0.33147416f;
constexpr float SOFTPLUS_BF16_H2 = -0.229736328125f;    // -941 * 2^-12
constexpr float SOFTPLUS_BF16_H3 = 0.12548828125f;      // 257 * 2^-11
constexpr float SOFTPLUS_BF16_H4 = -0.03411865234375f;  // -559 * 2^-14

// BF16: u = 2^(y-127), y = max(127 - a/ln2, 0), i.e. exp(-a) for a >= 0.
// Quasar has the truncating fp32->uint16 conversion (RoundMode::Zero), same as Blackhole.
// k = trunc(y), f = y - k in [0, 1), p(f) ~ 2^f in [1, 2), u = setexp(p(f), k).
// The max(y, 0) in softplus_bf16_eval is what sends y < 0 to k = 0: the conversion
// takes the magnitude, so an unclamped negative y comes back as a positive k.
// a > 88 clamps to y = 0: k = 0 and p(0) = 1 has a zero mantissa, so setexp yields exactly
// +0; 0 < y < 1 just above gives a denormal the SFPU flushes. Both are the right bf16 answer.
sfpi_inline sfpi::vFloat softplus_exp2_bf16(sfpi::vFloat y) {
    sfpi::vUInt16 k = sfpi::convert<sfpi::vUInt16>(y, sfpi::RoundMode::Zero);
    sfpi::vFloat f = y - sfpi::convert<sfpi::vFloat>(k, sfpi::RoundMode::Nearest);
    sfpi::vFloat p =
        PolynomialEvaluator::eval(f, SOFTPLUS_BF16_P0, sfpi::vConstFloatPrgm1, SOFTPLUS_BF16_P2, SOFTPLUS_BF16_P3);
    return sfpi::setexp(p, sfpi::as<sfpi::vInt>(k));
}

// f(a) = ln(1+u) = u * (1 + u*h(u)), u = exp(-a), a = |t|. No tail branch: the residual
// tends to u as u -> 0, so the negative tail stays accurate down to the bf16 normal floor
// instead of being clamped to 0. Returns beta_reciprocal * softplus(t), rounded to bf16.
sfpi_inline sfpi::vFloat softplus_bf16_eval(sfpi::vFloat t, const float beta_reciprocal) {
    sfpi::vFloat a = sfpi::setsgn(t, 0);
    sfpi::vFloat y = a * sfpi::vConstFloatPrgm0 + SOFTPLUS_BF16_EXP_BIAS;
    sfpi::vFloat sp = sfpi::max(t, 0.0f);
    y = sfpi::max(y, 0.0f);
    sfpi::vFloat u = softplus_exp2_bf16(y);
    sfpi::vFloat h = PolynomialEvaluator::eval(
        u, SOFTPLUS_BF16_H0, sfpi::vConstFloatPrgm2, SOFTPLUS_BF16_H2, SOFTPLUS_BF16_H3, SOFTPLUS_BF16_H4);
    sfpi::vFloat q = u * h + 1.0f;
    sp = u * q + sp;
    // SFPSTORE defaults to truncation.
    return sfpi::convert<sfpi::vFloat16b>(beta_reciprocal * sp, sfpi::RoundMode::Nearest);
}

template <bool is_fp32_dest_acc_en>
sfpi_inline void _calculate_softplus_body_(const float beta, const float beta_reciprocal, const float threshold) {
    sfpi::vFloat val = sfpi::dst_reg[0];
    sfpi::vFloat t = beta * val;

    // `t <= threshold` relies on vConstNeg1/LREG11 == -1.0 (re-established per launch by _init_sfpu_config_reg_).
    v_if(t <= threshold) {
        if constexpr (is_fp32_dest_acc_en) {
            sfpi::vFloat a = sfpi::setsgn(t, 0);
            sfpi::vFloat residual = PolynomialEvaluator::eval(
                a,
                SOFTPLUS_POLY_C0,
                SOFTPLUS_POLY_C1,
                SOFTPLUS_POLY_C2,
                SOFTPLUS_POLY_C3,
                SOFTPLUS_POLY_C4,
                SOFTPLUS_POLY_C5,
                SOFTPLUS_POLY_C6,
                SOFTPLUS_POLY_C7,
                SOFTPLUS_POLY_C8);

            // Tail for a > 5: f(a) ~ exp(-a) via 3-term Taylor ln(1+e) = e*(1 + e*(-1/2 + e/3)).
            v_if(a > SOFTPLUS_POLY_BOUNDARY) {
                sfpi::vFloat e = _sfpu_exp_fp32_accurate_(sfpi::setsgn(a, 1));
                residual = e * (1.0f + e * (-0.5f + e * 0.333333343f));
            }
            v_endif;

            t = sfpi::max(t, 0.0f);
            sfpi::dst_reg[0] = beta_reciprocal * (t + residual);
        } else {
            // Tail-preserving bf16 path. Rounding to bf16 is inside the helper.
            sfpi::dst_reg[0] = softplus_bf16_eval(t, beta_reciprocal);
        }
    }
    v_endif;

    sfpi::dst_reg++;
}

// Loads vConstFloatPrgm0/1/2 for the bf16 path. Wired from softplus_tile_init via
// SFPU_UNARY_INIT_FN, so it runs once per init rather than on every face. The fp32
// path does not read these registers. Call again after another SFPU init overwrites them.
template <bool is_fp32_dest_acc_en>
inline void softplus_init() {
    if constexpr (!is_fp32_dest_acc_en) {
        sfpi::vConstFloatPrgm0 = SOFTPLUS_BF16_NEG_ONE_LN2;
        sfpi::vConstFloatPrgm1 = SOFTPLUS_BF16_P1;
        sfpi::vConstFloatPrgm2 = SOFTPLUS_BF16_H1;
    }
}

/**
 * @brief Compute softplus (1/beta * ln(1 + exp(beta * x))) in-place over a Dest tile.
 *
 * Uses the abs(x) symmetry: with f(a) = ln(1 + exp(-a)) and t = beta * x,
 * softplus(x) = 1/beta * (t + f(t)) for t >= 0 and 1/beta * f(-t) for t < 0. Above `threshold` the op
 * is linear (returns x). is_fp32_dest_acc_en selects a degree-8 polynomial on [0, 5] plus an exp
 * Taylor tail (32-bit Dest) or the bf16 evaluation in u = exp(-|t|) (16-bit Dest), which keeps the
 * negative tail instead of clamping it to 0. APPROXIMATION_MODE is accepted for ABI parity but
 * ignored (softplus is exact).
 *
 * @tparam APPROXIMATION_MODE: Accepted for ABI parity; ignored (softplus has no approximate variant).
 * @tparam is_fp32_dest_acc_en: Select the fp32 (degree-8 + exp tail) vs bf16 (u = exp(-|t|)) path.
 * @tparam ITERATIONS: Number of SFPU loop iterations over the Dest tile.
 * @param beta: Sharpness parameter beta, as an fp32 bit pattern.
 * @param beta_reciprocal: 1/beta, as an fp32 bit pattern.
 * @param threshold: Linear-region threshold on beta*x above which softplus(x) = x, as an fp32 bit pattern.
 * @note The fp32 tail calls @ref _sfpu_exp_fp32_accurate_. The `t <= threshold` compare relies on
 *       vConstNeg1/LREG11 == -1.0, re-established per launch by @ref _init_sfpu_config_reg_. The bf16
 *       path reads vConstFloatPrgm0/1/2 loaded by @ref softplus_init. Call that init first, and again
 *       after another SFPU init overwrites those registers.
 */
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_softplus(std::uint32_t beta, std::uint32_t beta_reciprocal, std::uint32_t threshold) {
    const float beta_f = Converter::as_float(beta);
    const float beta_reciprocal_f = Converter::as_float(beta_reciprocal);
    const float threshold_f = Converter::as_float(threshold);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        _calculate_softplus_body_<is_fp32_dest_acc_en>(beta_f, beta_reciprocal_f, threshold_f);
    }
}

}  // namespace ckernel::sfpu
