// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * Shared piecewise polynomial P(x) evaluator for LUT-based SFPU activations.
 *
 * Used by ckernel_sfpu_<activation>.h drop-in headers.
 *
 * ALL optimization variants included:
 *   - Horner evaluation with #pragma unroll (any degree)
 *   - Parity x²-Horner (POLY_PARITY_ODD/POLY_PARITY_EVEN)
 *   - Dual-eval ILP: 2 DST rows simultaneously, ~1.27x speedup
 *   - Adaptive per-segment degree (HAS_SEGMENT_DEGREES + SEGMENT_DEGREES[])
 *   - Recursive template unrolling (ALWI)
 *   - Range reduction (exp/trig/tan/log/cbrt via ckernel_sfpu_piecewise_rational.h)
 *
 * Usage:
 *   #include "ckernel_sfpu_piecewise_polynomial.h"
 *   constexpr std::array<float, N> MY_LUT = {{ ... }};
 *   sfpi::vFloat result = ckernel::sfpu::piecewise_polynomial_eval<
 *       POLY_DEGREE, NUM_SEGS, LUT_SIZE>(MY_LUT, x);
 *
 * Dual-eval: #define USE_DUAL_EVAL before including.
 * Parity: #define POLY_PARITY_ODD or POLY_PARITY_EVEN before including.
 */

#include "ckernel.h"
#include "ckernel_defs.h"
#if defined(RANGE_REDUCTION_EXP) || defined(RANGE_REDUCTION_TRIG) || defined(RANGE_REDUCTION_LOG)
#include "ckernel_sfpu_piecewise_rational.h"
#endif

namespace ckernel::sfpu {

// ============================================================================
// Horner polynomial evaluation (any degree, #pragma unroll)
// ============================================================================

template <uint32_t DEGREE>
ALWI sfpi::vFloat piecewise_poly_eval(const float* coeffs, sfpi::vFloat x) {
    sfpi::vFloat result = coeffs[DEGREE];
    if constexpr (DEGREE > 0) {
#pragma unroll
        for (int i = static_cast<int>(DEGREE) - 1; i >= 0; i--) {
            result = result * x + coeffs[i];
        }
    }
    return result;
}

// ============================================================================
// Parity x²-Horner: odd (c0=c2=...=0) or even (c1=c3=...=0) polynomials
// ============================================================================

#if defined(POLY_PARITY_ODD)
template <uint32_t DEGREE>
ALWI sfpi::vFloat piecewise_poly_eval_parity(const float* coeffs, sfpi::vFloat x, sfpi::vFloat x2) {
    constexpr int TOP = (DEGREE % 2 == 1) ? DEGREE : DEGREE - 1;
    constexpr int STEPS = (TOP - 1) / 2;
    sfpi::vFloat result = coeffs[TOP];
#pragma unroll
    for (int k = 1; k <= STEPS; k++) {
        result = result * x2 + coeffs[TOP - 2 * k];
    }
    return result * x;
}
#elif defined(POLY_PARITY_EVEN)
template <uint32_t DEGREE>
ALWI sfpi::vFloat piecewise_poly_eval_parity(const float* coeffs, sfpi::vFloat x, sfpi::vFloat x2) {
    constexpr int TOP = (DEGREE % 2 == 0) ? DEGREE : DEGREE - 1;
    constexpr int STEPS = TOP / 2;
    sfpi::vFloat result = coeffs[TOP];
#pragma unroll
    for (int k = 1; k <= STEPS; k++) {
        result = result * x2 + coeffs[TOP - 2 * k];
    }
    return result;
}
#endif

// ============================================================================
// Dual-eval: process 2 DST rows simultaneously for ILP (~1.27x speedup)
// Coefficient loaded once, used for both chains.
// ============================================================================

#ifdef USE_DUAL_EVAL
template <uint32_t DEGREE>
ALWI void piecewise_poly_eval_dual(
    const float* coeffs, sfpi::vFloat x1, sfpi::vFloat x2, sfpi::vFloat& result1, sfpi::vFloat& result2) {
    {
        sfpi::vFloat c = coeffs[DEGREE];
        result1 = c;
        result2 = c;
    }
    if constexpr (DEGREE > 0) {
#pragma unroll
        for (int i = static_cast<int>(DEGREE) - 1; i >= 0; i--) {
            sfpi::vFloat c = coeffs[i];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
    }
}

#if defined(POLY_PARITY_ODD)
template <uint32_t DEGREE>
ALWI void piecewise_poly_eval_dual_parity(
    const float* coeffs,
    sfpi::vFloat x1,
    sfpi::vFloat x2,
    sfpi::vFloat x1_sq,
    sfpi::vFloat x2_sq,
    sfpi::vFloat& r1,
    sfpi::vFloat& r2) {
    constexpr int TOP = (DEGREE % 2 == 1) ? DEGREE : DEGREE - 1;
    constexpr int STEPS = (TOP - 1) / 2;
    {
        sfpi::vFloat c = coeffs[TOP];
        r1 = c;
        r2 = c;
    }
#pragma unroll
    for (int k = 1; k <= STEPS; k++) {
        sfpi::vFloat c = coeffs[TOP - 2 * k];
        r1 = r1 * x1_sq + c;
        r2 = r2 * x2_sq + c;
    }
    r1 = r1 * x1;
    r2 = r2 * x2;
}
#elif defined(POLY_PARITY_EVEN)
template <uint32_t DEGREE>
ALWI void piecewise_poly_eval_dual_parity(
    const float* coeffs,
    sfpi::vFloat x1,
    sfpi::vFloat x2,
    sfpi::vFloat x1_sq,
    sfpi::vFloat x2_sq,
    sfpi::vFloat& r1,
    sfpi::vFloat& r2) {
    constexpr int TOP = (DEGREE % 2 == 0) ? DEGREE : DEGREE - 1;
    constexpr int STEPS = TOP / 2;
    {
        sfpi::vFloat c = coeffs[TOP];
        r1 = c;
        r2 = c;
    }
#pragma unroll
    for (int k = 1; k <= STEPS; k++) {
        sfpi::vFloat c = coeffs[TOP - 2 * k];
        r1 = r1 * x1_sq + c;
        r2 = r2 * x2_sq + c;
    }
}
#endif
#endif  // USE_DUAL_EVAL

// ============================================================================
// Recursive segment unroller (single-eval)
// ============================================================================

template <uint32_t SEG, uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
ALWI void piecewise_poly_unroll_segment(
    const std::array<float, LUT_SIZE>& lut,
    sfpi::vFloat x,
    sfpi::vFloat& result
#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
    ,
    sfpi::vFloat x2
#endif
) {
    if constexpr (SEG < NUM_SEGMENTS) {
        constexpr uint32_t CPS = POLY_DEGREE + 1;
        constexpr uint32_t CO = NUM_SEGMENTS + 1;
#ifdef HAS_SEGMENT_DEGREES
        constexpr uint32_t DEG = SEGMENT_DEGREES[SEG];
#else
        constexpr uint32_t DEG = POLY_DEGREE;
#endif
        v_if(x >= lut[SEG]) {
#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
            result = piecewise_poly_eval_parity<DEG>(&lut[CO + SEG * CPS], x, x2);
#else
            result = piecewise_poly_eval<DEG>(&lut[CO + SEG * CPS], x);
#endif
        }
        v_endif;
        piecewise_poly_unroll_segment<SEG + 1, POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(
            lut,
            x,
            result
#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
            ,
            x2
#endif
        );
    }
}

// ============================================================================
// Recursive segment unroller (dual-eval)
// ============================================================================

#ifdef USE_DUAL_EVAL
template <uint32_t SEG, uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
ALWI void piecewise_poly_unroll_segment_dual(
    const std::array<float, LUT_SIZE>& lut,
    sfpi::vFloat x1,
    sfpi::vFloat x2,
    sfpi::vFloat& r1,
    sfpi::vFloat& r2
#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
    ,
    sfpi::vFloat x1_sq,
    sfpi::vFloat x2_sq
#endif
) {
    if constexpr (SEG < NUM_SEGMENTS) {
        constexpr uint32_t CPS = POLY_DEGREE + 1;
        constexpr uint32_t CO = NUM_SEGMENTS + 1;
#ifdef HAS_SEGMENT_DEGREES
        constexpr uint32_t DEG = SEGMENT_DEGREES[SEG];
#else
        constexpr uint32_t DEG = POLY_DEGREE;
#endif
        {
            sfpi::vFloat tmp1, tmp2;
#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
            piecewise_poly_eval_dual_parity<DEG>(&lut[CO + SEG * CPS], x1, x2, x1_sq, x2_sq, tmp1, tmp2);
#else
            piecewise_poly_eval_dual<DEG>(&lut[CO + SEG * CPS], x1, x2, tmp1, tmp2);
#endif
            sfpi::vFloat b = lut[SEG];
            v_if(x1 >= b) { r1 = tmp1; }
            v_endif;
            v_if(x2 >= b) { r2 = tmp2; }
            v_endif;
        }
        piecewise_poly_unroll_segment_dual<SEG + 1, POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(
            lut,
            x1,
            x2,
            r1,
            r2
#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
            ,
            x1_sq,
            x2_sq
#endif
        );
    }
}
#endif  // USE_DUAL_EVAL

// ============================================================================
// Public API: evaluate piecewise polynomial LUT for a single sfpi::vFloat x
// ============================================================================

template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
ALWI sfpi::vFloat piecewise_polynomial_eval(const std::array<float, LUT_SIZE>& lut, sfpi::vFloat x) {
    constexpr uint32_t COEFF_OFFSET = NUM_SEGMENTS + 1;

#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
    sfpi::vFloat x2 = x * x;
#endif

    // Segment 0 (unconditional)
#ifdef HAS_SEGMENT_DEGREES
    constexpr uint32_t DEG0 = SEGMENT_DEGREES[0];
#else
    constexpr uint32_t DEG0 = POLY_DEGREE;
#endif

#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
    sfpi::vFloat result = piecewise_poly_eval_parity<DEG0>(&lut[COEFF_OFFSET], x, x2);
#else
    sfpi::vFloat result = piecewise_poly_eval<DEG0>(&lut[COEFF_OFFSET], x);
#endif

    // Segments 1..N-1 via recursive unrolling
    piecewise_poly_unroll_segment<1, POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(
        lut,
        x,
        result
#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
        ,
        x2
#endif
    );

    return result;
}

// ============================================================================
// Full evaluation with range reduction
// Uses helpers from ckernel_sfpu_piecewise_rational.h (same reduce/expand)
// ============================================================================

template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_polynomial_eval_full(const std::array<float, LUT_SIZE>& lut, int d) {
    sfpi::vFloat x_orig = sfpi::dst_reg[d];

#if defined(RANGE_REDUCTION_EXP)
    sfpi::vFloat x;
    sfpi::vInt k_int;
    piecewise_exp_reduce(x_orig, x, k_int);
#elif defined(RANGE_REDUCTION_TRIG)
    sfpi::vFloat x;
    sfpi::vInt q_int;
    piecewise_trig_reduce(x_orig, x, q_int);
#elif defined(RANGE_REDUCTION_LOG)
    sfpi::vFloat x;
    sfpi::vInt e_int;
    piecewise_log_reduce(x_orig, x, e_int);
#else
    sfpi::vFloat x = x_orig;
#endif

    sfpi::vFloat result = piecewise_polynomial_eval<POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut, x);

#if defined(RANGE_REDUCTION_EXP)
    constexpr float EXP_OVERFLOW = 88.5f;
    constexpr float EXP_UNDERFLOW = -88.5f;
    v_if(x_orig > EXP_OVERFLOW) { result = std::numeric_limits<float>::infinity(); }
    v_elseif(x_orig < EXP_UNDERFLOW) { result = 0.0f; }
    v_else { result = piecewise_exp_expand(result, k_int); }
    v_endif;
#elif defined(RANGE_REDUCTION_TRIG)
    result = piecewise_trig_expand(result, q_int);
#elif defined(RANGE_REDUCTION_LOG)
    v_if(x_orig < 0.0f) { result = std::numeric_limits<float>::quiet_NaN(); }
    v_elseif(x_orig == 0.0f) { result = -std::numeric_limits<float>::infinity(); }
    v_else { result = piecewise_log_expand(result, e_int); }
    v_endif;
#endif

    sfpi::dst_reg[d] = result;
}

}  // namespace ckernel::sfpu
