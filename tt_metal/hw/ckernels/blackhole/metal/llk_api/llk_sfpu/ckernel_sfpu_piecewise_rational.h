// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * Shared piecewise rational P(x)/Q(x) evaluator for LUT-based SFPU activations.
 *
 * Stripped from piecewise_rational.cpp + piecewise_rational_specialized.cpp
 * for use in ckernel_sfpu_<activation>.h drop-in headers.
 *
 * ALL optimization variants included:
 *   - Interleaved Horner: back-to-back SFPMAD on numer/denom hides pipeline latency
 *   - Parity x²-Horner: halves FMA count for odd-num/even-den (atanh, erfinv, silu)
 *   - Deferred reciprocal: ONE sfpu_reciprocal for all segments
 *   - Recursive template unrolling: __attribute__((always_inline)) prevents spills
 *   - Range reduction: Cody-Waite for exp/trig, mantissa/exponent for log
 *
 * Usage:
 *   #include "ckernel_sfpu_piecewise_rational.h"
 *   constexpr std::array<float, N> MY_LUT = {{ ... }};
 *   vFloat result = ckernel::sfpu::piecewise_rational_eval<
 *       NUM_DEG, DEN_DEG, NUM_SEGS, LUT_SIZE>(MY_LUT, x);
 *
 * Parity: #define RATIONAL_NUM_PARITY_ODD + RATIONAL_DEN_PARITY_EVEN before including.
 * Range reduction: #define RANGE_REDUCTION_EXP/TRIG/LOG before including, then use
 *   piecewise_rational_eval_full<>() which handles reduce/expand.
 */

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "sfpu/ckernel_sfpu_converter.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// ============================================================================
// Range reduction helpers (Cody-Waite for exp/trig, IEEE754 for log)
// ============================================================================

#ifdef RANGE_REDUCTION_EXP
inline void piecewise_exp_reduce(vFloat x, vFloat& s, vInt& k_int) {
    constexpr float INV_LN2 = 1.4426950408889634f;
    constexpr float NEG_LN2_HI = -0.6931152343750000f;
    constexpr float NEG_LN2_LO = -3.19461832987e-05f;
    const vFloat c231 = Converter::as_float(0x4B400000U);
    vFloat tmp = x * INV_LN2 + c231;
    k_int = reinterpret<vInt>(tmp) - reinterpret<vInt>(c231);
    s = (tmp - c231) * NEG_LN2_LO + ((tmp - c231) * NEG_LN2_HI + x);
}

inline vFloat piecewise_exp_expand(vFloat poly_result, vInt k_int) {
    return setexp(poly_result, exexp_nodebias(poly_result) + k_int);
}
#endif

#ifdef RANGE_REDUCTION_TRIG
inline void piecewise_trig_reduce(vFloat x, vFloat& s, vInt& q_int) {
    constexpr float FRAC_1_PI = 0.31830988618379067f;
    const vFloat c231 = Converter::as_float(0x4B400000U);
    vFloat tmp = x * FRAC_1_PI + c231;
    vFloat q = tmp - c231;
    q_int = reinterpret<vInt>(tmp) - reinterpret<vInt>(c231);
    constexpr float NEG_PI_HI = -3.140625f;
    constexpr float NEG_PI_LO = -0.00096765358979323846f;
    s = q * NEG_PI_LO + (q * NEG_PI_HI + x);
}

inline vFloat piecewise_trig_expand(vFloat poly_result, vInt q_int) {
    v_if(q_int & 1) { poly_result = -poly_result; }
    v_endif;
    return poly_result;
}
#endif

#ifdef RANGE_REDUCTION_LOG
inline void piecewise_log_reduce(vFloat x, vFloat& m, vInt& e_int) {
    e_int = exexp_nodebias(x) - 127;
    m = setexp(x, 127);
}

inline vFloat piecewise_log_expand(vFloat poly_result, vInt e_int) {
    // LOG_EXPAND_CONSTANT may be defined by the including header for non-natural-log bases
    // (e.g., log2: 1.0, log10: log10(2)). Defaults to ln(2) for natural log.
#ifdef LOG_EXPAND_CONSTANT
    constexpr float EXPAND_C = LOG_EXPAND_CONSTANT;
#else
    constexpr float EXPAND_C = 0.6931471805599453f;
#endif
    v_if(e_int < 0) { e_int = setsgn(~e_int + 1, 1); }
    v_endif;
    return int32_to_float(e_int, 0) * EXPAND_C + poly_result;
}
#endif

// ============================================================================
// Interleaved Horner: evaluate P(x) and Q(x) simultaneously
// Back-to-back SFPMADs on independent chains hide pipeline latency.
// ============================================================================

template <uint32_t NUM_DEGREE, uint32_t DEN_DEGREE>
__attribute__((always_inline)) inline void piecewise_rational_eval_numer_denom(
    const float* num_coeffs, const float* den_coeffs, vFloat x, vFloat& out_numer, vFloat& out_denom) {
    constexpr uint32_t MIN_DEG = (NUM_DEGREE < DEN_DEGREE) ? NUM_DEGREE : DEN_DEGREE;

    vFloat numer = num_coeffs[NUM_DEGREE];
    vFloat denom = den_coeffs[DEN_DEGREE];

    if constexpr (NUM_DEGREE > DEN_DEGREE) {
#pragma unroll
        for (int i = NUM_DEGREE - 1; i >= static_cast<int>(DEN_DEGREE); i--) {
            numer = numer * x + num_coeffs[i];
        }
    } else if constexpr (DEN_DEGREE > NUM_DEGREE) {
#pragma unroll
        for (int i = DEN_DEGREE - 1; i >= static_cast<int>(NUM_DEGREE); i--) {
            denom = denom * x + den_coeffs[i];
        }
    }

#pragma unroll
    for (int i = MIN_DEG - 1; i >= 0; i--) {
        numer = numer * x + num_coeffs[i];
        denom = denom * x + den_coeffs[i];
    }

    out_numer = numer;
    out_denom = denom;
}

// ============================================================================
// Parity x²-Horner: odd num / even den → evaluate in x² basis
// Halves FMA count for atanh, erfinv, silu, etc.
// ============================================================================

#if defined(RATIONAL_NUM_PARITY_ODD) && defined(RATIONAL_DEN_PARITY_EVEN)
template <uint32_t NUM_DEGREE, uint32_t DEN_DEGREE>
__attribute__((always_inline)) inline void piecewise_rational_eval_parity_numer_denom(
    const float* num_coeffs, const float* den_coeffs, vFloat x, vFloat x2, vFloat& out_numer, vFloat& out_denom) {
    static_assert(NUM_DEGREE % 2 == 1, "RATIONAL_NUM_PARITY_ODD requires odd NUM_DEGREE");
    static_assert(DEN_DEGREE % 2 == 0, "RATIONAL_DEN_PARITY_EVEN requires even DEN_DEGREE");
    static_assert(NUM_DEGREE >= 1, "NUM_DEGREE must be at least 1 for parity evaluation");
    constexpr int NUM_TOP = NUM_DEGREE;
    constexpr int DEN_TOP = DEN_DEGREE;
    constexpr int NUM_STEPS = (NUM_TOP - 1) / 2;
    constexpr int DEN_STEPS = DEN_TOP / 2;

    vFloat numer = num_coeffs[NUM_TOP];
    vFloat denom = den_coeffs[DEN_TOP];

    if constexpr (NUM_STEPS > DEN_STEPS) {
#pragma unroll
        for (int k = 0; k < NUM_STEPS - DEN_STEPS; k++) {
            numer = numer * x2 + num_coeffs[NUM_TOP - 2 * (k + 1)];
        }
    } else if constexpr (DEN_STEPS > NUM_STEPS) {
#pragma unroll
        for (int k = 0; k < DEN_STEPS - NUM_STEPS; k++) {
            denom = denom * x2 + den_coeffs[DEN_TOP - 2 * (k + 1)];
        }
    }

    constexpr int MIN_STEPS = (NUM_STEPS < DEN_STEPS) ? NUM_STEPS : DEN_STEPS;
    constexpr int NUM_POS = NUM_TOP - 2 * ((NUM_STEPS > DEN_STEPS) ? (NUM_STEPS - DEN_STEPS) : 0);
    constexpr int DEN_POS = DEN_TOP - 2 * ((DEN_STEPS > NUM_STEPS) ? (DEN_STEPS - NUM_STEPS) : 0);

#pragma unroll
    for (int k = 1; k <= MIN_STEPS; k++) {
        numer = numer * x2 + num_coeffs[NUM_POS - 2 * k];
        denom = denom * x2 + den_coeffs[DEN_POS - 2 * k];
    }

    out_numer = numer * x;  // odd parity: P(x) = x * Horner_result
    out_denom = denom;
}
#endif

// ============================================================================
// Unified numer/denom dispatcher: selects parity or interleaved automatically
// ============================================================================

template <uint32_t NUM_DEGREE, uint32_t DEN_DEGREE>
__attribute__((always_inline)) inline void piecewise_rational_dispatch_numer_denom(
    const float* num_coeffs,
    const float* den_coeffs,
    vFloat x,
    vFloat& out_numer,
    vFloat& out_denom
#if defined(RATIONAL_NUM_PARITY_ODD) && defined(RATIONAL_DEN_PARITY_EVEN)
    ,
    vFloat x2
#endif
) {
#if defined(RATIONAL_NUM_PARITY_ODD) && defined(RATIONAL_DEN_PARITY_EVEN)
    piecewise_rational_eval_parity_numer_denom<NUM_DEGREE, DEN_DEGREE>(
        num_coeffs, den_coeffs, x, x2, out_numer, out_denom);
#else
    piecewise_rational_eval_numer_denom<NUM_DEGREE, DEN_DEGREE>(num_coeffs, den_coeffs, x, out_numer, out_denom);
#endif
}

// ============================================================================
// Recursive segment unroller with deferred reciprocal
// ============================================================================

template <uint32_t SEG, uint32_t NUM_DEGREE, uint32_t DEN_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
__attribute__((always_inline)) inline void piecewise_rational_unroll_segment(
    const std::array<float, LUT_SIZE>& lut,
    vFloat x,
    vFloat& numer,
    vFloat& denom
#if defined(RATIONAL_NUM_PARITY_ODD) && defined(RATIONAL_DEN_PARITY_EVEN)
    ,
    vFloat x2
#endif
) {
    if constexpr (SEG < NUM_SEGMENTS) {
        constexpr uint32_t NUM_COEFFS = NUM_DEGREE + 1;
        constexpr uint32_t CPS = NUM_COEFFS + DEN_DEGREE + 1;
        constexpr uint32_t CO = NUM_SEGMENTS + 1;
        v_if(x >= lut[SEG]) {
            piecewise_rational_dispatch_numer_denom<NUM_DEGREE, DEN_DEGREE>(
                &lut[CO + SEG * CPS],
                &lut[CO + SEG * CPS + NUM_COEFFS],
                x,
                numer,
                denom
#if defined(RATIONAL_NUM_PARITY_ODD) && defined(RATIONAL_DEN_PARITY_EVEN)
                ,
                x2
#endif
            );
        }
        v_endif;
        piecewise_rational_unroll_segment<SEG + 1, NUM_DEGREE, DEN_DEGREE, NUM_SEGMENTS, LUT_SIZE>(
            lut,
            x,
            numer,
            denom
#if defined(RATIONAL_NUM_PARITY_ODD) && defined(RATIONAL_DEN_PARITY_EVEN)
            ,
            x2
#endif
        );
    }
}

// ============================================================================
// Public API: evaluate piecewise rational LUT for a single vFloat x
// Automatically dispatches parity when macros are defined.
// ============================================================================

template <uint32_t NUM_DEGREE, uint32_t DEN_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
__attribute__((always_inline)) inline vFloat piecewise_rational_eval(const std::array<float, LUT_SIZE>& lut, vFloat x) {
    constexpr uint32_t NUM_COEFFS = NUM_DEGREE + 1;
    constexpr uint32_t COEFF_OFFSET = NUM_SEGMENTS + 1;

#if defined(RATIONAL_NUM_PARITY_ODD) && defined(RATIONAL_DEN_PARITY_EVEN)
    vFloat x2 = x * x;
#endif

    vFloat numer, denom;
    piecewise_rational_dispatch_numer_denom<NUM_DEGREE, DEN_DEGREE>(
        &lut[COEFF_OFFSET],
        &lut[COEFF_OFFSET + NUM_COEFFS],
        x,
        numer,
        denom
#if defined(RATIONAL_NUM_PARITY_ODD) && defined(RATIONAL_DEN_PARITY_EVEN)
        ,
        x2
#endif
    );

    piecewise_rational_unroll_segment<1, NUM_DEGREE, DEN_DEGREE, NUM_SEGMENTS, LUT_SIZE>(
        lut,
        x,
        numer,
        denom
#if defined(RATIONAL_NUM_PARITY_ODD) && defined(RATIONAL_DEN_PARITY_EVEN)
        ,
        x2
#endif
    );

    return numer * sfpu_reciprocal<false>(denom);
}

// ============================================================================
// Full evaluation with range reduction
// Reads dst_reg, applies reduce → eval → expand, writes back.
// ============================================================================

template <uint32_t NUM_DEGREE, uint32_t DEN_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_rational_eval_full(const std::array<float, LUT_SIZE>& lut, int d) {
    vFloat x_orig = dst_reg[d];

#if defined(RANGE_REDUCTION_EXP)
    vFloat x;
    vInt k_int;
    piecewise_exp_reduce(x_orig, x, k_int);
#elif defined(RANGE_REDUCTION_TRIG)
    vFloat x;
    vInt q_int;
    piecewise_trig_reduce(x_orig, x, q_int);
#elif defined(RANGE_REDUCTION_LOG)
    vFloat x;
    vInt e_int;
    piecewise_log_reduce(x_orig, x, e_int);
#else
    vFloat x = x_orig;
#endif

    vFloat result = piecewise_rational_eval<NUM_DEGREE, DEN_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut, x);

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

    dst_reg[d] = result;
}

}  // namespace sfpu
}  // namespace ckernel
