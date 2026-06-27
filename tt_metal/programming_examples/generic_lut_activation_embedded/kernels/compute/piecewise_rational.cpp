// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/operations/normalization/kernel_util/compute/memory.h"

// Canonical eval_method taxonomy. For the rational base kernel the method is
// EVAL_METHOD_RATIONAL_CASCADE (implied by including this file); REDUCED_POLY
// reduce-then-poly may still be layered on (REDUCE_EXP/TRIG/LOG). The shim maps
// the clean selector to the legacy RANGE_REDUCTION_* the bodies below use.
#include "eval_method.h"

// Include reciprocal function for rational evaluation
#ifdef TRISC_MATH
#include "ckernel_sfpu_recip.h"
#endif

namespace kutil = norm::kernel_util;

/**
 * Piecewise Rational Approximation Kernel
 *
 * Evaluates: y = P(x) / Q(x) where P and Q are polynomials
 *
 * For each segment:
 *   P(x) = n0 + n1*x + n2*x² + ... (numerator polynomial)
 *   Q(x) = d0 + d1*x + d2*x² + ... (denominator polynomial)
 *   y = P(x) / Q(x)
 *
 * LUT Format: [b0, b1, ..., bN, n0_seg0, n1_seg0, ..., d0_seg0, d1_seg0, ..., n0_seg1, ...]
 *
 * Template parameters:
 *   NUM_DEGREE    - Degree of numerator polynomial
 *   DEN_DEGREE    - Degree of denominator polynomial
 *   NUM_SEGMENTS  - Number of piecewise segments
 *   LUT_SIZE      - Total LUT size
 *
 * Supports both modes:
 *   - EMBEDDED_LUT: Coefficients compiled directly into kernel (zero L1 overhead)
 *   - CB mode: Coefficients loaded from L1 circular buffer (runtime flexibility)
 */

#ifdef TRISC_MATH

// Range reduction helpers (Cody-Waite for exp/trig, mantissa/exponent for log)
// Included unconditionally to avoid pre-commit hook reformatting breaking the conditional
#include "sfpu/ckernel_sfpu_converter.h"

namespace sfpi {

inline vFloat apply_output_postcompose(vFloat y, vFloat x_orig) {
#if defined(POSTCOMPOSE_AFFINE_Y)
    y = POSTCOMPOSE_B * y + POSTCOMPOSE_A;
#endif
#if defined(POSTCOMPOSE_AFFINE_Y_TIMES_INPUT)
    y = x_orig * (POSTCOMPOSE_B * y + POSTCOMPOSE_A);
#endif
    return y;
}

inline vFloat rational_reciprocal(vFloat denom) {
#if defined(RATIONAL_RECIPROCAL_ONE_ITER)
    return ckernel::sfpu::sfpu_reciprocal_iter<1>(denom);
#else
    return ckernel::sfpu::sfpu_reciprocal_iter<2>(denom);
#endif
}

#ifdef RANGE_REDUCTION_EXP
// Cody-Waite range reduction: x → (s, k_int) where exp(x) = 2^k * exp(s)
// s ∈ [-ln(2)/2, ln(2)/2] ≈ [-0.347, 0.347]
// Heavily optimized to minimize live SFPU registers - critical for exp with 2-segment rationals
inline void exp_reduce(vFloat x, vFloat& s, vInt& k_int) {
    constexpr float INV_LN2 = 1.4426950408889634f;
    constexpr float NEG_LN2_HI = -0.6931152343750000f;
    constexpr float NEG_LN2_LO = -3.19461832987e-05f;

    // Round-to-nearest-even: z = x/ln(2), k = round(z)
    const vFloat c231 = ckernel::sfpu::Converter::as_float(0x4B400000U);
    vFloat tmp = x * INV_LN2 + c231;  // Fuse multiply and add to reduce live registers

    // Extract k_int directly, minimize temporaries
    k_int = reinterpret<vInt>(tmp) - reinterpret<vInt>(c231);

    // Compute s = x - k*ln(2) with Cody-Waite extended precision
    // Reuse tmp-c231 as k to avoid additional register
    s = (tmp - c231) * NEG_LN2_LO + ((tmp - c231) * NEG_LN2_HI + x);
}

// Inline exp_expand to minimize register pressure - avoid creating intermediate vInt variables
inline vFloat exp_expand(vFloat poly_result, vInt k_int) {
    // Inline the operations to reduce live registers
    return setexp(poly_result, exexp_nodebias(poly_result) + k_int);
}
#endif

#ifdef RANGE_REDUCTION_TRIG
// Trig range reduction: x → (s, q_int) where f(x) = (-1)^q * f(s)
// s ∈ [-π/2, π/2], works for both sin and cos
inline void trig_reduce(vFloat x, vFloat& s, vInt& q_int) {
    constexpr float FRAC_1_PI = 0.31830988618379067f;
    vFloat z = x * FRAC_1_PI;

    // Round z to nearest integer (branch-free magic-number technique)
    const vFloat c231 = ckernel::sfpu::Converter::as_float(0x4B400000U);
    vFloat tmp = z + c231;
    vFloat q = tmp - c231;
    q_int = reinterpret<vInt>(tmp) - reinterpret<vInt>(c231);

    // Cody-Waite: s = x - q*π with extended precision
    constexpr float NEG_PI_HI = -3.140625f;
    constexpr float NEG_PI_LO = -0.00096765358979323846f;
    vFloat s_hi = q * NEG_PI_HI + x;
    s = q * NEG_PI_LO + s_hi;
}

// Reconstruct: sin(x + kπ) = (-1)^k * sin(x), cos(x + kπ) = (-1)^k * cos(x)
inline vFloat trig_expand(vFloat poly_result, vInt q_int) {
    v_if(q_int & 1) { poly_result = -poly_result; }
    v_endif;
    return poly_result;
}
#endif

#if defined(RANGE_REDUCTION_LOG)  // Log range reduction: x → (m, e_int) where x = 2^e * m, m ∈ [1, 2)
// log(x) = e*ln(2) + log(m)
// Uses IEEE 754 bit manipulation: exponent extraction and mantissa normalization
inline void log_reduce(vFloat x, vFloat& m, vInt& e_int) {
    // Extract biased exponent: for x = 2^e * m, biased_exp = e + 127
    vInt biased_exp = exexp_nodebias(x);

    // Unbias: e = biased_exp - 127
    // But we want m in [1, 2), so we set exponent to 127 (bias)
    // This gives m = x / 2^e = x * 2^(-e) = x with exponent = 127
    e_int = biased_exp - 127;

    // Normalize mantissa to [1, 2) by setting exponent bits to 127
    m = setexp(x, 127);
}

// Reconstruct: log(x) = e*ln(2) + log(m)
// poly_result = log(m) where m ∈ [1, 2)
// Final result = e*ln(2) + poly_result
inline vFloat log_expand(vFloat poly_result, vInt e_int) {
    // LOG_EXPAND_CONSTANT is base-specific: ln(2) for log, 1.0 for log2, log10(2) for log10
#ifndef LOG_EXPAND_CONSTANT
#define LOG_EXPAND_CONSTANT 0.6931471805599453f
#endif
    constexpr float EXPAND_C = LOG_EXPAND_CONSTANT;
    // int32_to_float expects SIGN-MAGNITUDE format, not two's complement.
    // Negative exponents must be converted: twos complement → sign-magnitude.
    v_if(e_int < 0) { e_int = setsgn(~e_int + 1, 1); }
    v_endif;
    vFloat e_float = int32_to_float(e_int, RoundMode::Nearest);
    return e_float * EXPAND_C + poly_result;
}
#endif

// Polynomial evaluation using Horner's method - reused for both numerator and denominator
template <uint32_t DEGREE>
inline vFloat eval_poly(const float* coeffs, vFloat x) {
    if constexpr (DEGREE == 0) {
        return coeffs[0];
    } else if constexpr (DEGREE == 1) {
        return coeffs[0] + coeffs[1] * x;
    } else if constexpr (DEGREE == 2) {
        return (coeffs[2] * x + coeffs[1]) * x + coeffs[0];
    } else if constexpr (DEGREE == 3) {
        return ((coeffs[3] * x + coeffs[2]) * x + coeffs[1]) * x + coeffs[0];
    } else if constexpr (DEGREE == 4) {
        vFloat result = coeffs[4];
        result = result * x + coeffs[3];
        result = result * x + coeffs[2];
        result = result * x + coeffs[1];
        result = result * x + coeffs[0];
        return result;
    } else if constexpr (DEGREE == 5) {
        vFloat result = coeffs[5];
        result = result * x + coeffs[4];
        result = result * x + coeffs[3];
        result = result * x + coeffs[2];
        result = result * x + coeffs[1];
        result = result * x + coeffs[0];
        return result;
    } else if constexpr (DEGREE == 6) {
        vFloat result = coeffs[6];
        result = result * x + coeffs[5];
        result = result * x + coeffs[4];
        result = result * x + coeffs[3];
        result = result * x + coeffs[2];
        result = result * x + coeffs[1];
        result = result * x + coeffs[0];
        return result;
    } else if constexpr (DEGREE == 7) {
        vFloat result = coeffs[7];
        result = result * x + coeffs[6];
        result = result * x + coeffs[5];
        result = result * x + coeffs[4];
        result = result * x + coeffs[3];
        result = result * x + coeffs[2];
        result = result * x + coeffs[1];
        result = result * x + coeffs[0];
        return result;
    } else if constexpr (DEGREE == 8) {
        vFloat result = coeffs[8];
        result = result * x + coeffs[7];
        result = result * x + coeffs[6];
        result = result * x + coeffs[5];
        result = result * x + coeffs[4];
        result = result * x + coeffs[3];
        result = result * x + coeffs[2];
        result = result * x + coeffs[1];
        result = result * x + coeffs[0];
        return result;
    } else {
        // General case for higher degrees
        vFloat result = coeffs[DEGREE];
        for (int i = DEGREE - 1; i >= 0; i--) {
            result = result * x + coeffs[i];
        }
        return result;
    }
}

// Interleaved numerator/denominator Horner evaluation for ILP on SFPU pipeline.
// Instead of evaluating P(x) then Q(x) sequentially, we interleave their FMA steps
// so that each SFPMAD's pipeline latency is hidden by the other chain's FMA.
// Register pressure: x, numer, denom, n_coeff, d_coeff = ~5 vFloat (vs dual-eval's 2× everything).
// Works with range reduction (unlike dual-eval which is disabled due to register pressure).
template <uint32_t NUM_DEGREE, uint32_t DEN_DEGREE>
inline vFloat eval_rational_interleaved(const float* num_coeffs, const float* den_coeffs, vFloat x) {
    constexpr uint32_t MIN_DEG = (NUM_DEGREE < DEN_DEGREE) ? NUM_DEGREE : DEN_DEGREE;
    constexpr uint32_t MAX_DEG = (NUM_DEGREE > DEN_DEGREE) ? NUM_DEGREE : DEN_DEGREE;

    // Initialize both Horner chains from highest-degree coefficients
    vFloat numer = num_coeffs[NUM_DEGREE];
    vFloat denom = den_coeffs[DEN_DEGREE];

    // If degrees differ, drain the higher-degree chain's surplus iterations first.
    // This keeps the interleaved loop symmetric.
    if constexpr (NUM_DEGREE > DEN_DEGREE) {
        // Numerator has extra high-degree terms
#pragma GCC unroll 16
        for (int i = NUM_DEGREE - 1; i >= static_cast<int>(DEN_DEGREE); i--) {
            numer = numer * x + num_coeffs[i];
        }
    } else if constexpr (DEN_DEGREE > NUM_DEGREE) {
        // Denominator has extra high-degree terms
#pragma GCC unroll 16
        for (int i = DEN_DEGREE - 1; i >= static_cast<int>(NUM_DEGREE); i--) {
            denom = denom * x + den_coeffs[i];
        }
    }

    // Interleaved Horner: both chains step down in lockstep.
    // Back-to-back SFPMADs on independent chains hide pipeline latency.
#pragma GCC unroll 16
    for (int i = MIN_DEG - 1; i >= 0; i--) {
        numer = numer * x + num_coeffs[i];
        denom = denom * x + den_coeffs[i];
    }

    // Reciprocal + final multiply
    vFloat recip = rational_reciprocal(denom);
    return numer * recip;
}

// Deferred variant: returns P(x) and Q(x) separately, no reciprocal.
// Used by deferred-reciprocal segment dispatch (one division at the end).
template <uint32_t NUM_DEGREE, uint32_t DEN_DEGREE>
inline void eval_rational_interleaved_numer_denom(
    const float* num_coeffs, const float* den_coeffs, vFloat x, vFloat& out_numer, vFloat& out_denom) {
    constexpr uint32_t MIN_DEG = (NUM_DEGREE < DEN_DEGREE) ? NUM_DEGREE : DEN_DEGREE;

    vFloat numer = num_coeffs[NUM_DEGREE];
    vFloat denom = den_coeffs[DEN_DEGREE];

    if constexpr (NUM_DEGREE > DEN_DEGREE) {
#pragma GCC unroll 16
        for (int i = NUM_DEGREE - 1; i >= static_cast<int>(DEN_DEGREE); i--) {
            numer = numer * x + num_coeffs[i];
        }
    } else if constexpr (DEN_DEGREE > NUM_DEGREE) {
#pragma GCC unroll 16
        for (int i = DEN_DEGREE - 1; i >= static_cast<int>(NUM_DEGREE); i--) {
            denom = denom * x + den_coeffs[i];
        }
    }

#pragma GCC unroll 16
    for (int i = MIN_DEG - 1; i >= 0; i--) {
        numer = numer * x + num_coeffs[i];
        denom = denom * x + den_coeffs[i];
    }

    out_numer = numer;
    out_denom = denom;
}

// Parity-aware interleaved Horner: exploits odd/even symmetry for ~2× speedup.
// For odd numerator (n0=n2=...=0): P(x) = x * Horner([n1,n3,n5,...], x²)
// For even denominator (d1=d3=...=0): Q(x) = Horner([d0,d2,d4,...], x²)
// Halves the Horner step count by evaluating in x² basis with stride-2 access.
// LUT format is unchanged — zero coefficients are simply skipped.
//
// Example: atanh n8d8 (odd num, even den)
//   Numerator:  n7*x⁷ + n5*x⁵ + n3*x³ + n1*x = x*(n7*x⁶ + n5*x⁴ + n3*x² + n1)
//              → 3 FMAs in x² (was 8 in x)
//   Denominator: d8*x⁸ + d6*x⁶ + d4*x⁴ + d2*x² + d0
//              → 4 FMAs in x² (was 8 in x)
#if defined(RATIONAL_NUM_PARITY_ODD) && defined(RATIONAL_DEN_PARITY_EVEN)
template <uint32_t NUM_DEGREE, uint32_t DEN_DEGREE>
inline vFloat eval_rational_parity(const float* num_coeffs, const float* den_coeffs, vFloat x) {
    vFloat x2 = x * x;

    // Highest non-zero coefficient index for each parity
    constexpr int NUM_TOP = (NUM_DEGREE % 2 == 1) ? NUM_DEGREE : NUM_DEGREE - 1;
    constexpr int DEN_TOP = (DEN_DEGREE % 2 == 0) ? DEN_DEGREE : DEN_DEGREE - 1;

    // Number of FMA steps in x² basis
    constexpr int NUM_STEPS = (NUM_TOP - 1) / 2;  // walk from NUM_TOP to 1, stride 2
    constexpr int DEN_STEPS = DEN_TOP / 2;        // walk from DEN_TOP to 0, stride 2

    vFloat numer = num_coeffs[NUM_TOP];
    vFloat denom = den_coeffs[DEN_TOP];

    // Drain the chain with more steps (keeps interleaved loop symmetric)
    if constexpr (NUM_STEPS > DEN_STEPS) {
#pragma GCC unroll 16
        for (int k = 0; k < NUM_STEPS - DEN_STEPS; k++) {
            numer = numer * x2 + num_coeffs[NUM_TOP - 2 * (k + 1)];
        }
    } else if constexpr (DEN_STEPS > NUM_STEPS) {
#pragma GCC unroll 16
        for (int k = 0; k < DEN_STEPS - NUM_STEPS; k++) {
            denom = denom * x2 + den_coeffs[DEN_TOP - 2 * (k + 1)];
        }
    }

    // After drain, both chains have MIN_STEPS remaining
    constexpr int MIN_STEPS = (NUM_STEPS < DEN_STEPS) ? NUM_STEPS : DEN_STEPS;
    constexpr int NUM_POS = NUM_TOP - 2 * ((NUM_STEPS > DEN_STEPS) ? (NUM_STEPS - DEN_STEPS) : 0);
    constexpr int DEN_POS = DEN_TOP - 2 * ((DEN_STEPS > NUM_STEPS) ? (DEN_STEPS - NUM_STEPS) : 0);

    // Interleaved Horner in x²: back-to-back SFPMADs on independent chains
#pragma GCC unroll 16
    for (int k = 1; k <= MIN_STEPS; k++) {
        numer = numer * x2 + num_coeffs[NUM_POS - 2 * k];
        denom = denom * x2 + den_coeffs[DEN_POS - 2 * k];
    }

    // Apply odd parity: P(x) = x * Horner_result
    numer = numer * x;

    vFloat recip = rational_reciprocal(denom);
    return numer * recip;
}

// Deferred variant: returns P(x) and Q(x) separately for deferred reciprocal.
template <uint32_t NUM_DEGREE, uint32_t DEN_DEGREE>
inline void eval_rational_parity_numer_denom(
    const float* num_coeffs, const float* den_coeffs, vFloat x, vFloat x2, vFloat& out_numer, vFloat& out_denom) {
    constexpr int NUM_TOP = (NUM_DEGREE % 2 == 1) ? NUM_DEGREE : NUM_DEGREE - 1;
    constexpr int DEN_TOP = (DEN_DEGREE % 2 == 0) ? DEN_DEGREE : DEN_DEGREE - 1;
    constexpr int NUM_STEPS = (NUM_TOP - 1) / 2;
    constexpr int DEN_STEPS = DEN_TOP / 2;

    vFloat numer = num_coeffs[NUM_TOP];
    vFloat denom = den_coeffs[DEN_TOP];

    if constexpr (NUM_STEPS > DEN_STEPS) {
#pragma GCC unroll 16
        for (int k = 0; k < NUM_STEPS - DEN_STEPS; k++) {
            numer = numer * x2 + num_coeffs[NUM_TOP - 2 * (k + 1)];
        }
    } else if constexpr (DEN_STEPS > NUM_STEPS) {
#pragma GCC unroll 16
        for (int k = 0; k < DEN_STEPS - NUM_STEPS; k++) {
            denom = denom * x2 + den_coeffs[DEN_TOP - 2 * (k + 1)];
        }
    }

    constexpr int MIN_STEPS = (NUM_STEPS < DEN_STEPS) ? NUM_STEPS : DEN_STEPS;
    constexpr int NUM_POS = NUM_TOP - 2 * ((NUM_STEPS > DEN_STEPS) ? (NUM_STEPS - DEN_STEPS) : 0);
    constexpr int DEN_POS = DEN_TOP - 2 * ((DEN_STEPS > NUM_STEPS) ? (DEN_STEPS - NUM_STEPS) : 0);

#pragma GCC unroll 16
    for (int k = 1; k <= MIN_STEPS; k++) {
        numer = numer * x2 + num_coeffs[NUM_POS - 2 * k];
        denom = denom * x2 + den_coeffs[DEN_POS - 2 * k];
    }

    out_numer = numer * x;  // odd parity: P(x) = x * Horner_result
    out_denom = denom;
}
#endif

// JIT (Just-In-Time) coefficient loading version for high-degree rationals
// This version loads coefficients one at a time to minimize register pressure
// Uses only 3-4 vFloat registers instead of 15+ for high degrees
template <uint32_t NUM_DEGREE, uint32_t DEN_DEGREE>
inline vFloat eval_rational_jit(const float* num_coeffs, const float* den_coeffs, vFloat x) {
    // === PHASE 1: Evaluate denominator with JIT loading ===
    // Load highest degree coefficient for denominator
    vFloat coeff = den_coeffs[DEN_DEGREE];
    vFloat denom_result = coeff;

    // Horner's method: load one coefficient at a time, reusing the 'coeff' register
    // CRITICAL: Do NOT unroll for low degrees with range reduction - unrolling exposes
    // all loop iterations to register allocator at once, causing spills!
    // Only unroll for high degrees where loop overhead matters more than register pressure
#if defined(RANGE_REDUCTION_EXP) || defined(RANGE_REDUCTION_TRIG) || \
    defined(RANGE_REDUCTION_LOG)  // With range reduction: disable unroll for degrees < 10 to avoid register spills
#if (DEN_DEGREE >= 10)
#pragma GCC unroll 16
#endif
#else
    // Without range reduction: safe to unroll at all degrees
#pragma GCC unroll 16
#endif
    for (int i = DEN_DEGREE - 1; i >= 0; i--) {
        coeff = den_coeffs[i];  // Reuse register - previous value no longer needed
        denom_result = denom_result * x + coeff;
    }

    // Compute reciprocal (denom_result dies after this)
    vFloat recip = rational_reciprocal(denom_result);

    // === PHASE 2: Evaluate numerator with JIT loading ===
    // Reuse the same 'coeff' register for numerator coefficients
    coeff = num_coeffs[NUM_DEGREE];
    vFloat numer_result = coeff;

#if defined(RANGE_REDUCTION_EXP) || defined(RANGE_REDUCTION_TRIG) || \
    defined(RANGE_REDUCTION_LOG)  // With range reduction: disable unroll for degrees < 10 to avoid register spills
#if (NUM_DEGREE >= 10)
#pragma GCC unroll 16
#endif
#else
    // Without range reduction: safe to unroll at all degrees
#pragma GCC unroll 16
#endif
    for (int i = NUM_DEGREE - 1; i >= 0; i--) {
        coeff = num_coeffs[i];  // Reuse register
        numer_result = numer_result * x + coeff;
    }

    // Final multiply: only 2 registers live (numer_result + recip)
    return numer_result * recip;
}

// Evaluate rational function for a single segment: P(x)/Q(x)
// Dispatches to parity-aware, interleaved, or sequential JIT based on context.
template <uint32_t NUM_DEGREE, uint32_t DEN_DEGREE>
inline vFloat eval_rational(const float* num_coeffs, const float* den_coeffs, vFloat x) {
    // Parity-aware x²-Horner: when odd num / even den symmetry is detected,
    // evaluate in x² basis with stride-2 access, halving the Horner step count.
#if defined(RATIONAL_NUM_PARITY_ODD) && defined(RATIONAL_DEN_PARITY_EVEN)
    return eval_rational_parity<NUM_DEGREE, DEN_DEGREE>(num_coeffs, den_coeffs, x);
#else
    // Interleaved evaluation: numerator and denominator Horner chains run in lockstep,
    // hiding SFPMAD pipeline latency. ~5 vFloat registers (x, numer, denom, + coeff loads).
#ifdef EMBEDDED_LUT
    return eval_rational_interleaved<NUM_DEGREE, DEN_DEGREE>(num_coeffs, den_coeffs, x);
#else
#if defined(RANGE_REDUCTION_EXP) || defined(RANGE_REDUCTION_TRIG) || \
    defined(RANGE_REDUCTION_LOG)
    return eval_rational_interleaved<NUM_DEGREE, DEN_DEGREE>(num_coeffs, den_coeffs, x);
#else
    if constexpr (NUM_DEGREE >= 7 || DEN_DEGREE >= 7) {
        return eval_rational_interleaved<NUM_DEGREE, DEN_DEGREE>(num_coeffs, den_coeffs, x);
    } else {
        vFloat numerator = eval_poly<NUM_DEGREE>(num_coeffs, x);
        vFloat denominator = eval_poly<DEN_DEGREE>(den_coeffs, x);
        vFloat recip = rational_reciprocal(denominator);
        return numerator * recip;
    }
#endif
#endif
#endif
}

#if TT_ACT_EVAL_KIND == TT_ACT_EVAL_ABS_DENOMINATOR_RATIONAL || defined(ABS_DENOMINATOR_RATIONAL)
inline void abs_denominator_rational_eval() {
#pragma GCC unroll 8
    for (int d = 0; d < 32; d++) {
        vFloat x = dst_reg[d];
        vFloat den = setsgn(x, 0) + 1.0f;
        vFloat y = x * rational_reciprocal(den);
        y = apply_output_postcompose(y, x);
#ifdef USE_BF16
        y = convert<vFloat16b>(y, RoundMode::Nearest);
#endif
        dst_reg[d] = y;
    }
}
#endif

// Evaluate P(x) and Q(x) separately for deferred-reciprocal dispatch.
// The segment unroller calls this inside v_if, then does ONE reciprocal outside.
template <uint32_t NUM_DEGREE, uint32_t DEN_DEGREE>
inline void eval_rational_numer_denom(
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
    eval_rational_parity_numer_denom<NUM_DEGREE, DEN_DEGREE>(num_coeffs, den_coeffs, x, x2, out_numer, out_denom);
#else
    eval_rational_interleaved_numer_denom<NUM_DEGREE, DEN_DEGREE>(num_coeffs, den_coeffs, x, out_numer, out_denom);
#endif
}

// Include specialized implementations using recursive template unrolling
// Replaces old hand-written piecewise_rational_lut_2/4/8/16/generic functions
#include "piecewise_rational_specialized.cpp"

}  // namespace sfpi
#endif

void kernel_main() {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

#ifdef EMBEDDED_LUT
    // Embedded LUT mode: LUT is compiled directly into the kernel
    // Header must define: NUM_DEGREE, DEN_DEGREE, NUM_SEGMENTS, LUT_SIZE, LUT_DATA
    // This provides zero L1 memory overhead for the LUT
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    // Use embedded LUT data directly from header
    constexpr uint32_t lut_size = LUT_SIZE;
    constexpr uint32_t num_degree = NUM_DEGREE;
    constexpr uint32_t den_degree = DEN_DEGREE;
    constexpr uint32_t num_segments = NUM_SEGMENTS;
    const auto& lut_ref = LUT_DATA;
    auto p_lut = &lut_ref;
#else
    // Generic LUT mode: LUT is loaded from L1 circular buffer
    // This allows runtime LUT generation and sharing across cores
    [[maybe_unused]] float input_min = get_arg_val<float>(1);
    [[maybe_unused]] float input_max = get_arg_val<float>(2);
    constexpr uint32_t lut_size = get_compile_time_arg_val(0);
    constexpr uint32_t num_degree = get_compile_time_arg_val(1);
    constexpr uint32_t den_degree = get_compile_time_arg_val(2);
    constexpr uint32_t num_segments = get_compile_time_arg_val(3);

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr auto cb_lut = tt::CBIndex::c_25;

    // Get LUT array from L1 memory as float array directly
    using lut_t = std::array<float, lut_size>;
    auto p_lut = kutil::compute::memory::get_pointer_to_cb_data<lut_t>(cb_lut, 0);
#endif

    init_sfpu(cb_in, cb_out);

#ifdef TRISC_MATH
    // Initialize reciprocal SFPU function (needed for rational evaluation)
    // Platform-independent init for reciprocal
    ckernel::sfpu::sfpu_reciprocal_init<false>();
#endif

    for (uint32_t tile = 0; tile < n_tiles; tile++) {
        cb_wait_front(cb_in, 1);
        tile_regs_acquire();
        copy_tile(cb_in, 0, 0);

#ifdef TRISC_MATH
        // All degree parameters are constexpr — use them directly as template args.
        // No dispatch table needed; works for ANY (num_degree, den_degree) combination.
#if TT_ACT_EVAL_KIND == TT_ACT_EVAL_ABS_DENOMINATOR_RATIONAL || defined(ABS_DENOMINATOR_RATIONAL)
        (void)p_lut;
        sfpi::abs_denominator_rational_eval();
#else
        sfpi::piecewise_rational_lut_dispatch<num_degree, den_degree, num_segments, lut_size>(*p_lut);
#endif
#endif

        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in, 1);
        tile_regs_release();
    }
}
