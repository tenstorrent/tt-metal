// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpu/ckernel_sfpu_log.h"
#include "ckernel_sfpu_recip.h"

#include "ckernel_sfpu_piecewise_rational.h"
#include "cmath_common.h"

namespace ckernel::sfpu {

// ======================================================================
// LUT-based digamma via piecewise rational P(x)/Q(x)
//
// BF16: n6/d5, 1 segment, range [0.01, 102.0]
// FP32: n10/d9, 2 segment(s), range [0.01, 102.0]
// ======================================================================

#ifdef INP_FLOAT32
constexpr uint32_t DIGAMMA_NUM_DEGREE = 10;
constexpr uint32_t DIGAMMA_DEN_DEGREE = 9;
constexpr uint32_t DIGAMMA_NUM_SEGMENTS = 2;
constexpr uint32_t DIGAMMA_LUT_SIZE = 45;
constexpr std::array<float, 45> DIGAMMA_LUT = {
    {1.0000000000e-02f,  5.1005000000e+01f, 1.0200000000e+02f, -4.2596286535e-01f, -1.2459139824e+00f,
     -7.0374596119e-01f, 3.5418295860e-01f, 3.9686101675e-01f, 1.0760356486e-01f,  1.1159370653e-02f,
     4.5299832709e-04f,  6.4513715188e-06f, 2.3348954770e-08f, 3.3629592167e-12f,  -6.3762914866e-08f,
     4.2596703768e-01f,  1.0000000000e+00f, 8.2736301422e-01f, 3.0081826448e-01f,  4.9985647202e-02f,
     3.7117889151e-03f,  1.1463041301e-04f, 1.2610012732e-06f, 3.3732798776e-09f,  1.3856337913e+05f,
     -2.7915017828e+05f, 5.5189096727e+04f, 2.6230285816e+04f, 2.0102400217e+03f,  5.4774037756e+01f,
     6.1410588531e-01f,  2.8495312547e-03f, 4.9248967852e-06f, 2.2963254970e-09f,  3.8530168912e-14f,
     -1.6724299601e+05f, 1.0000000000e+00f, 6.3677231206e+04f, 1.1013851737e+04f,  5.9383811201e+02f,
     1.2890647910e+01f,  1.2061770104e-01f, 4.7490052009e-04f, 6.9240316615e-07f,  2.5925175628e-10f}};

#else

constexpr uint32_t DIGAMMA_NUM_DEGREE = 6;
constexpr uint32_t DIGAMMA_DEN_DEGREE = 5;
constexpr uint32_t DIGAMMA_NUM_SEGMENTS = 1;
constexpr uint32_t DIGAMMA_LUT_SIZE = 15;
// Layout: [range_lo, range_hi], then num coeffs a0..a6 (ascending degree), then den coeffs b0..b5.
constexpr std::array<float, 15> DIGAMMA_LUT = {
    {1.0000000000e-02f,
     1.0200000000e+02f,
     -3.3767190625e+05f,
     -6.0287050000e+05f,
     2.1532992188e+05f,
     2.1398454688e+05f,
     1.9496867188e+04f,
     2.4847094727e+02f,
     8.7151519954e-02f,
     1.0000000000e+00f,
     3.3760150000e+05f,
     4.0892268750e+05f,
     9.9775718750e+04f,
     5.1246479492e+03f,
     4.1351390839e+01f}};

#endif

// pi*cot(pi*f) = (1 + f^2*P(f^2)) / f on f in [-1/2, 1/2]: P is minimax in the absolute error of f*P
// (Lawson-weighted least squares). c0 is pinned so the fp32 Horner sum gives P(1/4) = -4 exactly, i.e.
// g(1/2) = 0, which positive lanes rely on. Max abs error 4.6e-7 (fp32) / 6.9e-5 (bf16; c1..c3 are
// fp16-exact, so each is one SFPLOADI).
#ifdef INP_FLOAT32
constexpr std::array<float, 6> DIGAMMA_COT_P = {{
    -0x1.a51a46p+1f, -0x1.152264p+1f, -0x1.029f46p+1f, -0x1.1878bp+1f, -0x1.c813aep-1f, -0x1.3e8c84p+2f}};
#else
constexpr std::array<float, 4> DIGAMMA_COT_P = {{-0x1.a50c02p+1f, -0x1.18cp+1f, -0x1.a68p+0f, -0x1.e4p+1f}};
#endif

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_digamma() {
    constexpr uint32_t NUM_COEFFS = DIGAMMA_NUM_DEGREE + 1;
    constexpr uint32_t COEFF_OFFSET = DIGAMMA_NUM_SEGMENTS + 1;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // The rational and the asymptotic below are positive-axis approximations; evaluating the [0.01, 102]
        // rational at a negative argument returned unrelated values (-7.19 for psi(-0.9375) = -15.43). For
        // x < 0 they are evaluated at 1 - x (> 1) and the reflection psi(x) = psi(1 - x) - pi*cot(pi*x) is
        // applied, with pi*cot(pi*x) = g/f, f = x - round(x) in [-1/2, 1/2], g = 1 + f^2*P(f^2).
        // Positive lanes carry f = 1/2, where g = 0 exactly (pi*cot(pi/2) = 0; c0 is pinned so that the fp32
        // Horner sum gives P(1/4) = -4 bit-exactly). Every expression below then reduces exactly to the
        // positive-axis body (the halving of N, D and 1/x cancels exactly), so their results are unchanged and
        // no lane needs a predicate around the cotangent work. Only f is carried across the rational (f and
        // g together exceed the LREG file); g is formed afterwards.
        //
        // Every float below -2^23 is an integer, so x is clamped there. For -2^23 <= x < 0, x - 2^23 lies in
        // [-2^24, -2^23], where the float spacing is exactly 1, so adding the -2^23 bias rounds x to nearest
        // (kernels build with -fno-associative-math, so (x + b) - b is not folded back to x). At a pole (f == 0: the
        // non-positive integers) psi is undefined: the reflected argument is set to NaN, which every path below
        // propagates, as torch.digamma returns NaN there. x = -inf takes the same route.
        sfpi::vFloat f = 0.5f;
        v_if(x < 0.0f) {
            sfpi::vFloat rounding_bias = sfpi::sFloat16b(-0x1p23f);
            sfpi::vFloat xc = sfpi::max(x, -0x1p23f);
            sfpi::vFloat j = xc + rounding_bias;
            j = j - rounding_bias;
            f = xc - j;
            x = 1.0f - x;
            v_if(f == 0.0f) { x = std::numeric_limits<float>::quiet_NaN(); }
            v_endif;
        }
        v_endif;

        // Piecewise-rational LUT, fit on [0.01, 102], with its reciprocal deferred so that one reciprocal
        // serves both terms: psi = N/D - g/f = (N*f - D*g) / (D*f).
        sfpi::vFloat numer = 0.0f, denom = 0.0f;
        piecewise_rational_eval_numer_denom<DIGAMMA_NUM_DEGREE, DIGAMMA_DEN_DEGREE>(
            &DIGAMMA_LUT[COEFF_OFFSET], &DIGAMMA_LUT[COEFF_OFFSET + NUM_COEFFS], x, numer, denom);
        if constexpr (DIGAMMA_NUM_SEGMENTS > 1) {
            piecewise_rational_unroll_segment<
                1,
                DIGAMMA_NUM_DEGREE,
                DIGAMMA_DEN_DEGREE,
                DIGAMMA_NUM_SEGMENTS,
                DIGAMMA_LUT_SIZE>(DIGAMMA_LUT, x, numer, denom);
        }
        sfpi::vFloat s = f * f;
        constexpr uint32_t COT_DEG = DIGAMMA_COT_P.size() - 1;
        sfpi::vFloat p = DIGAMMA_COT_P[COT_DEG];
#pragma GCC unroll 8
        for (int i = COT_DEG - 1; i >= 1; i--) {
            p = p * s + DIGAMMA_COT_P[i];
        }
        p = p * s + sfpi::vConstFloatPrgm1;  // DIGAMMA_COT_P[0], a full fp32 constant held by digamma_init
        sfpi::vFloat g = s * p + 1.0f;
        numer = numer * f - denom * g;
        denom = denom * f;

        // Large-x asymptotic (Bernoulli series). Above the LUT fit range the rational
        // extrapolates poorly (issue #45520: "behaves bad for x>1000"); the asymptotic
        // digamma(x) = ln(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - ... is ~exact for large x
        // (truncation error ~8e-11 at x=102), restoring the (1, inf) support the pre-LUT
        // composite op provided. Crossover at the LUT's upper bound 102 is seamless.
        // NOTE: uses the register-free (bf16-grade) log body, so large-x fp32 is only
        // bf16-accurate. #45520 targets bf16 ULP; an fp32-accurate log here would collide
        // with the reciprocal's vConstFloatPrgm0, so fp32 large-x is intentionally left as-is.
        // Here the shared reciprocal is of x*f, so 1/x = f*r, and numer = g*x makes numer*r = g/f, the
        // reflection term (0 on positive lanes).
        v_if(x > 102.0f) {
            numer = g * x;
            denom = x * f;
        }
        v_endif;
        sfpi::vFloat r = sfpu_reciprocal_iter<2>(denom);
        sfpi::vFloat result = numer * r;
        v_if(x > 102.0f) {
            sfpi::vFloat inv_x = f * r;
            // inv_x*0.5 + g/f: on positive lanes g/f = 0, so this is exactly inv_x*0.5.
            sfpi::vFloat head = inv_x * sfpi::vFloat(0.5f) + result;
            sfpi::vFloat inv_x2 = inv_x * inv_x;
            // ln(x) - inv_x*0.5 - inv_x2*(1/12 - inv_x2/120); 1/12 is held in vConstFloatPrgm2.
            sfpi::vFloat bern = sfpi::vConstFloatPrgm2 - inv_x2 * sfpi::vFloat(0.0083333333f);
            result = _calculate_log_body_no_init_(x) - head - inv_x2 * bern;
            // digamma(+inf) = +inf; the log approximation clamps inf to a finite value, so
            // restore it explicitly (exp field all-ones, zero mantissa => infinity).
            v_if(sfpi::exexp(x) == 128 && sfpi::exman(x) == 0) { result = std::numeric_limits<float>::infinity(); }
            v_endif;
        }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void digamma_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    // On Blackhole sfpu_reciprocal_init programs only vConstFloatPrgm0 (2.0 for the Newton step), so
    // Prgm1/Prgm2 hold two full fp32 constants of the loop, each saving two SFPLOADIs per row.
    sfpu_reciprocal_init();
    sfpi::vConstFloatPrgm1 = DIGAMMA_COT_P[0];
    sfpi::vConstFloatPrgm2 = 0.0833333333f;
}

}  // namespace ckernel::sfpu
