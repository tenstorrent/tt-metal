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
// TTI replay lowering framework (shared with piecewise_generic.cpp): the
// TT_REPLAY_* macros, capacity check, and the silicon-derived HAZARD RULES
// every hand-scheduled body must obey.
#include "tti_replay.h"
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

#include "sfpi_min_max.h"
#include "domain_actions.h"

#if defined(TT_RATIONAL_COORDINATE_BOUND)
inline vFloat rational_bound_coordinate(vFloat x) {
    vFloat lo = -TT_RATIONAL_COORDINATE_BOUND;
    vFloat hi = TT_RATIONAL_COORDINATE_BOUND;
    ordered_min_max(lo, x);
    ordered_min_max(x, hi);
    return x;
}

inline void rational_bound_coordinate_tile() {
#pragma GCC unroll 8
    for (int d = 0; d < 32; d++) {
        dst_reg[d] = rational_bound_coordinate(dst_reg[d]);
    }
}
#endif

inline vFloat apply_output_postcompose(vFloat y, vFloat x_orig) {
#if defined(ASYMPTOTIC_FACTOR_QUADRATIC) && defined(ASYMPTOTIC_QUAD_ROOT_A) && \
    defined(ASYMPTOTIC_QUAD_ROOT_B)
    // INTERIOR-ROOT FACTORING for a rational fit: f(x) = (x-A)(x-B) * R(x).
    // Mirrors piecewise_generic_specialized.cpp:528-531, applied ONCE
    // post-cascade because the factor is segment-independent.
    //
    // This is the only construction that makes an interior root EXACT for a
    // rational. A coefficient-space pin cannot: the Hermite-in-monomials
    // residual is ~1e-7 after fp32 rounding, and ulp_bf16(0) = 2^-133 =
    // 9.18e-41, so it scores ~1e33 pure ULP -- worse than not pinning at all.
    // Here fl(A - A) = 0 EXACTLY (equal operands, Sterbenz -- no rounding is
    // possible) and fp32 0*finite = 0, so every rounding inside R -- both
    // Horners, the seeded Newton reciprocal, the bf16 cast -- is annihilated at
    // the root, independent of degree, coefficient magnitude, and arch.
    //
    // Precondition: R must be FINITE at the roots (0 * inf = NaN). The
    // has_pole_in_domain / reciprocal_unsafe CSV invariants already gate that.
    y = y * (x_orig - ASYMPTOTIC_QUAD_ROOT_A) * (x_orig - ASYMPTOTIC_QUAD_ROOT_B);
#if defined(ASYMPTOTIC_SCALE)
    y = y * ASYMPTOTIC_SCALE;
#endif
#endif
#if defined(POSTCOMPOSE_AFFINE_Y)
    y = POSTCOMPOSE_B * y + POSTCOMPOSE_A;
#endif
#if defined(POSTCOMPOSE_AFFINE_Y_TIMES_INPUT)
    y = x_orig * (POSTCOMPOSE_B * y + POSTCOMPOSE_A);
#endif
    return y;
}

// ---------------------------------------------------------------------------
// wh_reciprocal_deferred_scale — WH reciprocal with a shortened `scale` live range.
//
// WH has no SFPARECIP, so tt-metal's WH sfpu_reciprocal_iter builds the seed from
// the inverted exponent bits. It computes `scale` EARLY but consumes it LAST
// (y = y * scale), so `scale` is live across the entire Newton chain: peak live
// {in, negative_x, y, scale, t} = 5 of 8 LREGs. That is why three pinned
// coefficients spill (sfpi ICE "cannot store sfpu register") and why 26
// silu/swish rows are sfpi_jit_reject.
//
// `scale` depends only on `in`, which is already live to the end for copysgn, so
// deferring it past the Newton iterations drops the peak to 4 and frees the
// register the pins need. Same operations on the same values -- accuracy is
// bit-identical, verified on silicon.
//
// It is NOT free: deferring exposes Newton-chain stalls the early scale
// computation used to fill, measured at about +0.45us. So it is opt-in at the
// PINNED call site only, where the freed register is actually spent. Every
// non-pinned config keeps the stock path and is timing-identical to baseline.
// WH_RECIP_DEFER_SCALE_DISABLE is the C-LOW-2 escape.
// ---------------------------------------------------------------------------
#if defined(ARCH_WORMHOLE) && !defined(WH_RECIP_DEFER_SCALE_DISABLE)
#define WH_RECIP_DEFER_SCALE 1
template <int max_iter = 2>
sfpi_inline vFloat wh_reciprocal_deferred_scale(const vFloat in) {
    vFloat negative_x = sfpi::copyman(-1.0f, in);
    vFloat y = sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0 * negative_x;
    y = sfpi::vConstFloatPrgm2 + y * negative_x;
    vFloat t = 1.0f + negative_x * y;
    y = y + y * t;
    if constexpr (max_iter > 1) {
        t = 1.0f + negative_x * y;
        y = y + y * t;
    }
    vUInt scale_bits = ~sfpi::as<vUInt>(in);
    vFloat scale = sfpi::setman(sfpi::as<vFloat>(scale_bits), 0);
    scale *= 0.5f;
    y = y * scale;
    y = sfpi::copysgn(y, in);
    return y;
}
#endif

inline vFloat rational_reciprocal(vFloat denom) {
#if defined(RATIONAL_RECIPROCAL_ZERO_ITER)
    // Raw 8-bit SFPARECIP seed, zero Newton steps — TTNN's own erfc divide.
    return ckernel::sfpu::sfpu_reciprocal_iter<0>(denom);
#elif defined(RATIONAL_RECIPROCAL_ONE_ITER)
    return ckernel::sfpu::sfpu_reciprocal_iter<1>(denom);
#else
    return ckernel::sfpu::sfpu_reciprocal_iter<2>(denom);
#endif
}

// Used ONLY from the pinned rational loop.
inline vFloat rational_reciprocal_pinned(vFloat denom) {
#if defined(WH_RECIP_DEFER_SCALE)
#if defined(RATIONAL_RECIPROCAL_ZERO_ITER)
    return wh_reciprocal_deferred_scale<0>(denom);
#elif defined(RATIONAL_RECIPROCAL_ONE_ITER)
    return wh_reciprocal_deferred_scale<1>(denom);
#else
    return wh_reciprocal_deferred_scale<2>(denom);
#endif
#else
    return rational_reciprocal(denom);
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
    k_int = as<vInt>(tmp) - as<vInt>(c231);

    // Compute s = x - k*ln(2) with Cody-Waite extended precision
    // Reuse tmp-c231 as k to avoid additional register
    s = (tmp - c231) * NEG_LN2_LO + ((tmp - c231) * NEG_LN2_HI + x);
}

// Reconstruct: result = poly_result * 2^k via exponent bit manipulation.
//
// UNDERFLOW GUARD (same defect as piecewise_generic.cpp::asymptotic_exp and
// dst_coeff_store.h::asym_exp_dst): setexp writes the exponent FIELD directly,
// so a biased exponent at or below 0 wraps into a large positive exponent and
// returns ~1e38 where the true value has underflowed to ~0.  The caller's
// EXP_UNDERFLOW floor (-88.5f) is looser than the arithmetic: the wrap begins at
// x = -126*ln2 = -87.33654, so x in [-88.5, -87.33654) reaches here with
// new_exp <= 0 (3 bf16 grid points: -88.5, -88.0, -87.5).
//
// Clamp AFTER setexp: setexp(0.0f, e) rebuilds a nonzero value from the zero
// mantissa.  Register cost is one extra live vInt -- new_exp is held across the
// SFPSETEXP and the incoming poly_result slot is reused (no separate `out`), so
// the live set stays at x_orig + result + new_exp.  Precedent on this same path:
// log_expand below already nests a v_if inside the identical v_else.
inline vFloat exp_expand(vFloat poly_result, vInt k_int) {
    vInt new_exp = exexp(poly_result, ExponentMode::Biased) + k_int;
    poly_result = setexp(poly_result, new_exp);
    v_if(new_exp <= 0) { poly_result = 0.0f; }
    v_endif;
    return poly_result;
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
    q_int = as<vInt>(tmp) - as<vInt>(c231);

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
    vInt biased_exp = exexp(x, ExponentMode::Biased);

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
    v_if(e_int < 0) { e_int = as<vInt>(setsgn(as<vSMag>(~e_int + 1), 1)); }
    v_endif;
    vFloat e_float = convert<vFloat>(as<vSMag>(e_int), RoundMode::Nearest);
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

// ============================================================================
// LREG-pinned interleaved Horner for big single-segment rationals (BH + bf16)
// ============================================================================
// R7/4-class single-segment rationals (n7d4 / n6d4 / n5d5: softplus & friends)
// carry 11-13 Horner constants, none LREG-resident: under EMBEDDED_LUT every
// coefficient rematerializes as an in-body SFPLOADI pair (2 slots each, per
// element). These bodies exceed the 32-slot replay buffer, so the TTI-replay
// lowering cannot help. Keep the persistent coefficient hoist to TWO LREGs:
// this is the repository-wide maximum-safe uniform preload budget under the
// real -O3 -flto allocator (docs/TTMETAL_UPSTREAM_ISSUES.md, Issue 2). A third
// pin leaves no coefficient-independent headroom for reciprocal/finalization
// temporaries and can turn harmless coefficient re-rounding into an LTO spill
// ICE. LReg4/5 are used; LReg7 remains forbidden because BH silicon can clobber
// it mid-loop (see the TRIG_RESIDUAL block in piecewise_generic.cpp).
//
// The l_reg write/read pin + post-loop write-back is the ckernel_sfpu_gelu.h
// idiom: the value enters the named LREG once (pre-loop SFPLOADI), every use
// is an opaque register read the constant folder cannot see through, and the
// post-loop write-back keeps it live-out so the allocator can neither clobber
// nor rematerialize it. Numerically a no-op: the per-chain FMA sequence is
// identical to eval_rational_interleaved_numer_denom (same bits, same ops).
//
// Gate (kept deliberately narrow — everything else keeps the proven path):
//   BH + bf16 + EMBEDDED_LUT + plain rational_cascade (no range reduction,
//   no parity, no pre/postcompose) + single segment + R7/4-class shape
//   (num-heavy, total degree >= 10). RATIONAL_PIN_DISABLE is the A/B escape.
// The two-pin ceiling is architecture-neutral. WH's software reciprocal has a
// larger live set than BH ARECIP, while BH can still reach the same allocator
// frontier when raw-domain finalization is live. The former three-pin scheme
// has produced the same `cannot store sfpu register` LTO ICE on both arches.
#if defined(EMBEDDED_LUT) && defined(USE_BF16) &&                                               \
    (defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)) &&                                      \
    !defined(RATIONAL_PIN_DISABLE) && !defined(RANGE_REDUCTION_EXP) &&                          \
    !defined(RANGE_REDUCTION_TRIG) && !defined(RANGE_REDUCTION_LOG) &&                          \
    !defined(RATIONAL_NUM_PARITY_ODD) && !defined(RATIONAL_DEN_PARITY_EVEN) &&                  \
    !defined(PRECOMPOSE_INPUT_AFFINE) && !defined(POSTCOMPOSE_AFFINE_Y) &&                      \
    !defined(POSTCOMPOSE_AFFINE_Y_TIMES_INPUT)
#define TT_RATIONAL_LREG_PIN_ELIGIBLE 1
#else
#define TT_RATIONAL_LREG_PIN_ELIGIBLE 0
#endif

#if TT_RATIONAL_LREG_PIN_ELIGIBLE
// Persistent coefficient registers are a scarce architectural resource, not
// a function of activation identity or fitted coefficient values. Keep this
// explicit so every eligible rational shape uses the same audited pressure
// budget and coefficient re-rounding cannot change compile viability.
inline constexpr uint32_t kRationalLregPinCount = 2;
static_assert(kRationalLregPinCount == 2, "the audited uniform SFPU hoist budget is two LREGs");

// Compile-time shape gate used by the dispatcher (single segment is checked
// there too; this covers the R7/4-class shape).
template <uint32_t NUM_DEGREE, uint32_t DEN_DEGREE>
inline constexpr bool rational_pin_shape_ok =
    (NUM_DEGREE >= DEN_DEGREE) && (NUM_DEGREE >= 2) && (NUM_DEGREE + DEN_DEGREE >= 10)
#if defined(ARCH_WORMHOLE)
    // WH ONLY: the pin is reachable here only because the deferred-scale
    // reciprocal frees an LREG, and that deferral costs ~0.45us of exposed
    // Newton stalls. A short denominator Horner has too few pinned coefficients
    // to cover it. Measured at 256t, pure ULP bit-identical on every arm:
    //   den>=4 GAIN  logsigmoid R6/4 9.73->9.33, softplus R7/4 9.58->9.19,
    //                mish R9/4 10.76->10.37, i0 R6/6 8.86->8.75
    //   den==2 LOSE  sinh R13/2, erfinv R9/2  (excluded here; they keep the
    //                generic path and stay byte- and timing-identical)
    && (DEN_DEGREE >= 4)
#endif
    ;

// Interleaved P/Q Horner with the top two numerator coefficients supplied as
// LREG-resident pins. The denominator top coefficient is consumed in-body,
// like the unpinned evaluator. Per-chain operation sequence is EXACTLY that of
// eval_rational_interleaved_numer_denom for num-heavy shapes:
//   numer: (((n_top*x + n_next)*x + n[NUM-2]) ... )
//   denom: ((d_top*x + d[DEN-1]) ... )
template <uint32_t NUM_DEGREE, uint32_t DEN_DEGREE>
sfpi_inline void eval_rational_interleaved_numer_denom_pinned(
    const float* num_coeffs,
    const float* den_coeffs,
    vFloat x,
    vFloat pin_n_top,
    vFloat pin_n_next,
    vFloat& out_numer,
    vFloat& out_denom) {
    static_assert(rational_pin_shape_ok<NUM_DEGREE, DEN_DEGREE>, "pinned path expects a num-heavy R7/4-class rational");

    // First numerator step consumes both numerator pins (Horner step i = NUM_DEGREE-1).
    vFloat numer = pin_n_top * x + pin_n_next;
    vFloat denom = den_coeffs[DEN_DEGREE];

    if constexpr (NUM_DEGREE > DEN_DEGREE) {
        // Remaining numerator drain down to DEN_DEGREE (immediates, as before).
#pragma GCC unroll 16
        for (int i = static_cast<int>(NUM_DEGREE) - 2; i >= static_cast<int>(DEN_DEGREE); i--) {
            numer = numer * x + num_coeffs[i];
        }
        // Interleaved lockstep: back-to-back SFPMADs on independent chains.
#pragma GCC unroll 16
        for (int i = static_cast<int>(DEN_DEGREE) - 1; i >= 0; i--) {
            numer = numer * x + num_coeffs[i];
            denom = denom * x + den_coeffs[i];
        }
    } else {
        // Equal degrees (e.g. n5d5): the pinned first numerator step already
        // consumed n[NUM_DEGREE-1], so the denominator does its own first step
        // before the interleave picks up both chains at i = DEN_DEGREE-2.
        denom = denom * x + den_coeffs[DEN_DEGREE - 1];
#pragma GCC unroll 16
        for (int i = static_cast<int>(DEN_DEGREE) - 2; i >= 0; i--) {
            numer = numer * x + num_coeffs[i];
            denom = denom * x + den_coeffs[i];
        }
    }

    out_numer = numer;
    out_denom = denom;
}
#endif  // TT_RATIONAL_LREG_PIN_ELIGIBLE

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
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL)
        vUInt raw_u16 = dst_reg[d].mode<DataLayout::U16>();
#endif
        vFloat x_raw = dst_reg[d];
        vFloat x = prepare_raw_domain_input(x_raw);
        vFloat den = setsgn(x, 0) + 1.0f;
        vFloat y = x * rational_reciprocal(den);
        y = apply_output_postcompose(y, x);
        finalize_raw_domain_actions(x_raw, y);
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL) && \
    !defined(TT_TARGET_BH_BF16_POST_ROUND_RAW_CLASS_REPAIR)
        finalize_encoded_raw_domain_actions(raw_u16, y);
#endif
#ifdef USE_BF16
        y = convert<vFloat16b>(y, RoundMode::Nearest);
#endif
#if defined(TT_TARGET_BH_BF16_POST_ROUND_RAW_CLASS_REPAIR)
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL)
        finalize_post_round_raw_domain_actions(x_raw, raw_u16, y);
#else
        finalize_post_round_raw_domain_actions(x_raw, y);
#endif
#endif
        dst_reg[d] = y;
    }
}
#endif

// Anonymous total-domain replay for the exact x/(1+abs(x)) coefficient form.
// A verifier-derived magnitude clamp folds the symmetric +/-1 finite terminals
// into the selected arithmetic.  `x*(-1)+x` is exactly zero for every finite
// coordinate, but NaN for either decoded infinity; adding that poison after
// the unchanged two-Newton rational makes every exponent-FF ingress narrow to
// the declared +Inf first-pass quotient without a raw load or late finalizer.
// The selected domain is strictly inside the clamp bound, so coefficient bytes,
// operation order, reciprocal iterations, and BF16 rounding are unchanged.
#if defined(EMBEDDED_LUT) && defined(USE_BF16) && defined(ARCH_BLACKHOLE) && \
    defined(ABS_DENOMINATOR_RATIONAL) &&                                  \
    defined(TT_TARGET_BH_BF16_ABS_DENOMINATOR_TTI) &&                     \
    !defined(RATIONAL_TTI_DISABLE) && !defined(TT_DOMAIN_ACTION_PROGRAM) && \
    !defined(PRECOMPOSE_INPUT_AFFINE) && !defined(POSTCOMPOSE_AFFINE_Y) && \
    !defined(POSTCOMPOSE_AFFINE_Y_TIMES_INPUT) &&                         \
    !defined(RATIONAL_RECIPROCAL_ZERO_ITER) &&                            \
    !defined(RATIONAL_RECIPROCAL_ONE_ITER)
#define ABS_DENOMINATOR_TTI_CANDIDATE 1
constexpr bool kAbsDenominatorTtiReplay =
    (TT_TARGET_BH_BF16_ABS_DENOMINATOR_TTI_BODY_SLOTS == 19u) &&
    tti_replay_fits(TT_TARGET_BH_BF16_ABS_DENOMINATOR_TTI_BODY_SLOTS);

// Arm the existing BH load-macro RNE/store tail for this anonymous replay.
// T1 captures the same FP32->BF16 RNE used by the plain body; macro sequence
// zero fires it at issue+1 and the unchanged-format store at issue+3.  The
// template issue is backdoor-only and cannot alter L13's typed clamp bound.
inline void abs_denominator_tti_lm_init() {
    TTI_SFP_STOCH_RND(
        SFPSTOCHRND_RND_EVEN,
        0,
        0,
        ckernel::p_sfpu::LREG0,
        13,
        SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPLOADI(0, SFPLOADI_MOD0_LOWER, 0x0000);  // mad | simple idle
    TTI_SFPLOADI(0, SFPLOADI_MOD0_UPPER, 0x1385);  // store | round
    TTI_SFPCONFIG(0, 4, 0);
    TTI_SFPCONFIG(0x110, 8, 1);
}

inline void abs_denominator_tti_replay_tile() {
    constexpr uint32_t kBodySlots = 19u;
    static_assert(kAbsDenominatorTtiReplay);
    TT_REPLAY_RECORD(kBodySlots);  // record + execute element 0
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_HOLD, 0);  // x
    TTI_SFPNOP;  // conservative replay load-consumer gap
    TTI_SFPSETSGN(
        0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG1,
        1);  // magnitude = |x|
    TTI_SFPSWAP(
        0,
        ckernel::p_sfpu::LREG13,
        ckernel::p_sfpu::LREG1,
        sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);  // magnitude=min(|x|, bound)
    TTI_SFPMAD(
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG7,
        0);  // den = magnitude + 1.0
    TTI_SFPMAD(
        ckernel::p_sfpu::LCONST_neg1,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG5,
        0);  // finite: 0 exactly; decoded +/-Inf: NaN; den hazard gap
    TTI_SFPARECIP(0, 7, 2, 0);  // reciprocal seed -> L2
    TTI_SFPSETSGN(
        0,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG0,
        0);  // restore x sign; reciprocal-seed hazard gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG12,
        ckernel::p_sfpu::LREG3,
        2);  // t = den*r0 - 2.0
    TTI_SFPMAD(
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LREG0,
        0);  // finite identity / exponent-FF poison fills residual hazard gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG4,
        3);  // y1 = r0*-t - 0
    TTI_SFPNOP;
    TTI_SFPSETCC(0, 3, 0, 0);  // CC: first residual < 0
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG12,
        ckernel::p_sfpu::LREG1,
        1);  // t2 = den*y1 - 2.0 (predicated)
    TTI_SFPNOP;
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG2,
        2);  // reciprocal = y1*-t2 - 0 (predicated)
    TTI_SFPENCC(3, 0, 0, 10);  // CC-balanced replay body
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG0,
        0);  // selected y = bounded_x * reciprocal
    // The load-macro itself is the result-MUL hazard gap.  Captured T1 RNE
    // fires at issue+1, then the fixed store advances the same destination
    // row at issue+3. L4 is dead after the second Newton update.
    TTI_SFPLOADMACRO(
        (0 << 2) | (ckernel::p_sfpu::LREG4 & 3),
        0,
        TT_ADDRMOD_DST_ADV,
        (ckernel::p_sfpu::LREG4 >> 2));
    TT_REPLAY_REST(kBodySlots, 31);  // elements 1..31
    TT_REPLAY_TILE_EPILOGUE();
}
#else
#define ABS_DENOMINATOR_TTI_CANDIDATE 0
constexpr bool kAbsDenominatorTtiReplay = false;
#endif

#if TT_ACT_EVAL_KIND == TT_ACT_EVAL_SQUARED_ABS_DENOMINATOR_RATIONAL || \
    defined(SQUARED_ABS_DENOMINATOR_RATIONAL)
inline void squared_abs_denominator_rational_eval() {
#pragma GCC unroll 8
    for (int d = 0; d < 32; d++) {
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL)
        vUInt raw_u16 = dst_reg[d].mode<DataLayout::U16>();
#endif
        vFloat x_raw = dst_reg[d];
        vFloat x = prepare_raw_domain_input(x_raw);
        vFloat den = setsgn(x, 0) + 1.0f;
        vFloat r = rational_reciprocal(den);
        vFloat y = r * r;
        y = apply_output_postcompose(y, x);
        finalize_raw_domain_actions(x_raw, y);
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL)
        finalize_encoded_raw_domain_actions(raw_u16, y);
#endif
#ifdef USE_BF16
        y = convert<vFloat16b>(y, RoundMode::Nearest);
#endif
        dst_reg[d] = y;
    }
}
#endif

// Anonymous total-domain replay for the exact 1/(1+abs(x))**2 coefficient
// form (the derivative of the abs-denominator quotient above).  Unlike the
// linear sibling this body needs NO magnitude clamp and NO special poison:
// the host certificate proves that the unchanged two-Newton reciprocal plus
// one squaring SFPMUL already reproduces every typed finite terminal (the
// exclusive +0 tails AND the inclusive FP32 min-normal singletons at exactly
// +/-2^63) bit-for-bit, and that every exponent-FF ingress word egresses as
// the declared canonical positive zero: the reciprocal seed of an infinite
// denominator is +0 and its NaN first residual fails the predicated Newton
// guard, so the squared seed is exactly +0.  Coefficient bytes, operation
// order, reciprocal iterations, and BF16 rounding are unchanged.
#if defined(EMBEDDED_LUT) && defined(USE_BF16) && defined(ARCH_BLACKHOLE) && \
    defined(SQUARED_ABS_DENOMINATOR_RATIONAL) &&                            \
    defined(TT_TARGET_BH_BF16_SQUARED_ABS_DENOMINATOR_TTI) &&               \
    !defined(RATIONAL_TTI_DISABLE) && !defined(TT_DOMAIN_ACTION_PROGRAM) && \
    !defined(PRECOMPOSE_INPUT_AFFINE) && !defined(POSTCOMPOSE_AFFINE_Y) &&  \
    !defined(POSTCOMPOSE_AFFINE_Y_TIMES_INPUT) &&                           \
    !defined(RATIONAL_RECIPROCAL_ZERO_ITER) &&                              \
    !defined(RATIONAL_RECIPROCAL_ONE_ITER)
#define SQUARED_ABS_DENOMINATOR_TTI_CANDIDATE 1
constexpr bool kSquaredAbsDenominatorTtiReplay =
    (TT_TARGET_BH_BF16_SQUARED_ABS_DENOMINATOR_TTI_BODY_SLOTS == 18u) &&
    tti_replay_fits(TT_TARGET_BH_BF16_SQUARED_ABS_DENOMINATOR_TTI_BODY_SLOTS);

// Arm the existing BH load-macro RNE/store tail for this anonymous replay.
// T1 captures the same FP32->BF16 RNE used by the plain body; macro sequence
// zero fires it at issue+1 and the unchanged-format store at issue+3.  The
// template issue is backdoor-only; no programmable constant beyond the
// reciprocal helper's immutable 2.0 (Prgm0/L12) is consumed.
inline void squared_abs_denominator_tti_lm_init() {
    TTI_SFP_STOCH_RND(
        SFPSTOCHRND_RND_EVEN,
        0,
        0,
        ckernel::p_sfpu::LREG0,
        13,
        SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPLOADI(0, SFPLOADI_MOD0_LOWER, 0x0000);  // mad | simple idle
    TTI_SFPLOADI(0, SFPLOADI_MOD0_UPPER, 0x1385);  // store | round
    TTI_SFPCONFIG(0, 4, 0);
    TTI_SFPCONFIG(0x110, 8, 1);
}

inline void squared_abs_denominator_tti_replay_tile() {
    constexpr uint32_t kBodySlots = 18u;
    static_assert(kSquaredAbsDenominatorTtiReplay);
    TT_REPLAY_RECORD(kBodySlots);  // record + execute element 0
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_HOLD, 0);  // x
    TTI_SFPNOP;  // conservative replay load-consumer gap
    TTI_SFPSETSGN(
        0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG1,
        1);  // magnitude = |x| (no clamp: certificate-proved unnecessary)
    TTI_SFPMAD(
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG7,
        0);  // den = magnitude + 1.0
    TTI_SFPNOP;  // den hazard gap
    TTI_SFPARECIP(0, 7, 2, 0);  // reciprocal seed -> L2
    TTI_SFPNOP;  // reciprocal-seed hazard gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG12,
        ckernel::p_sfpu::LREG3,
        2);  // t = den*r0 - 2.0
    TTI_SFPNOP;  // first-residual hazard gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG4,
        3);  // y1 = r0*-t - 0
    TTI_SFPNOP;
    TTI_SFPSETCC(0, 3, 0, 0);  // CC: first residual < 0 (NaN guard)
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG12,
        ckernel::p_sfpu::LREG1,
        1);  // t2 = den*y1 - 2.0 (predicated)
    TTI_SFPNOP;
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG2,
        2);  // reciprocal = y1*-t2 - 0 (predicated)
    TTI_SFPENCC(3, 0, 0, 10);  // CC-balanced replay body
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG0,
        0);  // y = reciprocal * reciprocal (the derivative's only extra op)
    // The load-macro itself is the result-MUL hazard gap.  Captured T1 RNE
    // fires at issue+1, then the fixed store advances the same destination
    // row at issue+3. L4 is dead after the second Newton update.
    TTI_SFPLOADMACRO(
        (0 << 2) | (ckernel::p_sfpu::LREG4 & 3),
        0,
        TT_ADDRMOD_DST_ADV,
        (ckernel::p_sfpu::LREG4 >> 2));
    TT_REPLAY_REST(kBodySlots, 31);  // elements 1..31
    TT_REPLAY_TILE_EPILOGUE();
}
#else
#define SQUARED_ABS_DENOMINATOR_TTI_CANDIDATE 0
constexpr bool kSquaredAbsDenominatorTtiReplay = false;
#endif

// ----------------------------------------------------------------------------
// Closed biased-log P5/Q3 reconstruction.  The fitter owns the ordinary
// rational CSV and this anonymous form owns only the certified replay schedule:
//
//   a = min(abs(x), terminal); u = scale*a; z = N5(u)/D3(u)
//   f = fract(z); y = as_float(floor(z)<<23) * (1 + f*(e1 + e2*f))
//
// N5=D0=E0=1 are structural gauges. N4 occupies Prgm2/L14 and D3 is
// materialized in-body; fixed DST coefficient offsets are
// forbidden because the replay store advances D-RWC every iteration. The
// remaining coefficients are tile-pinned. The
// existing LOADMACRO mechanism retires RNE/store during the next
// replay head, leaving exactly one 32-issue common loop.
#if defined(EMBEDDED_LUT) && defined(USE_BF16) && defined(ARCH_BLACKHOLE) && \
    defined(EVAL_METHOD_BIASED_LOG_RATIONAL_EXP_1)
#define BIASED_LOG_RATIONAL_EXP_TTI_CANDIDATE 1
static_assert(NUM_DEGREE == 5 && DEN_DEGREE == 3 && NUM_SEGMENTS == 1);
static_assert(TT_BIASED_LOG_BODY_SLOTS == 32u);
static_assert(TT_BIASED_LOG_PEAK_LREGS == 8u);
static_assert(TT_BIASED_LOG_DRAIN_SLOTS == 5u);
static_assert(__builtin_bit_cast(uint32_t, TT_BIASED_LOG_NUM[5]) == 0x3f800000u);
static_assert(__builtin_bit_cast(uint32_t, TT_BIASED_LOG_DEN[0]) == 0x3f800000u);
static_assert((TT_BIASED_LOG_COORD_SCALE_BITS & 0xffffu) == 0u);
static_assert((TT_BIASED_LOG_TERMINAL_BITS & 0xffffu) == 0u);
static_assert((TT_BIASED_LOG_EXP_E1_BITS & 0xffffu) == 0u);
static_assert((TT_BIASED_LOG_EXP_E2_BITS & 0xffffu) == 0u);

inline void biased_log_rational_exp_lm_init() {
    // T1: literal L2 fp32 -> BF16 RNE into VD staging.  The macro load's VD
    // substitution lands on unused Source B, so its raw DST value cannot
    // replace the completed result in L2.  The fixed delayed STORE reads VD
    // and uses the destination row latched by SFPLOADMACRO.
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_EVEN,
        0,
        0,
        ckernel::p_sfpu::LREG2,
        13,
        sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG0,
        sfpi::SFPLOADI_MOD0_UPPER,
        (TT_BIASED_LOG_LM_SEQUENCE_BITS >> 16));
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG0,
        sfpi::SFPLOADI_MOD0_LOWER,
        (TT_BIASED_LOG_LM_SEQUENCE_BITS & 0xffffu));
    TTI_SFPCONFIG(0, 4, 0);
    TTI_SFPCONFIG(TT_BIASED_LOG_LM_MISC_BITS, 8, 1);
    TTI_SFPNOP;
}

inline void biased_log_rational_exp_preload_kernel() {
    TT_REPLAY_PIN_FP32(
        ckernel::p_sfpu::LREG3,
        __builtin_bit_cast(uint32_t, TT_BIASED_LOG_NUM[3]));
    TT_REPLAY_PIN_FP32(
        ckernel::p_sfpu::LREG4,
        __builtin_bit_cast(uint32_t, TT_BIASED_LOG_NUM[2]));
    TT_REPLAY_PIN_FP32(
        ckernel::p_sfpu::LREG5,
        __builtin_bit_cast(uint32_t, TT_BIASED_LOG_NUM[1]));
    TT_REPLAY_PIN_FP32(
        ckernel::p_sfpu::LREG6,
        __builtin_bit_cast(uint32_t, TT_BIASED_LOG_NUM[0]));

}

inline void biased_log_rational_exp_replay_tile() {
    constexpr uint32_t kBodySlots = TT_BIASED_LOG_BODY_SLOTS;
    static_assert(tti_replay_fits(kBodySlots));
    TT_REPLAY_RECORD(kBodySlots);
    TTI_SFPLOAD(
        ckernel::p_sfpu::LREG0,
        0,
        TT_ADDRMOD_DST_HOLD,
#if defined(TT_TARGET_COMPOSITE_NATIVE_EVEN_CLASS_REP_TTI)
        64);  // x from the preserved tile-1 ingress
#else
        0);  // x
#endif
    // This independent exact immediate is the required x-load consumer gap.
    // It does not overwrite the prior iteration's delayed macro carrier L2 or
    // factor L1.
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG7,
        sfpi::SFPLOADI_MOD0_FLOATB,
        (TT_BIASED_LOG_TERMINAL_BITS >> 16));
    TTI_SFPSETSGN(
        0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        1);  // |x|
    TTI_SFPSWAP(
        0,
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG0,
        sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);  // a=min(|x|,92)
    TTI_SFPMULI(
        (TT_BIASED_LOG_COORD_SCALE_BITS >> 16),
        ckernel::p_sfpu::LREG0,
        0);  // u
    constexpr uint32_t kD3Bits =
        __builtin_bit_cast(uint32_t, TT_BIASED_LOG_DEN[3]);
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG2,
        sfpi::SFPLOADI_MOD0_UPPER,
        (kD3Bits >> 16));
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG2,
        sfpi::SFPLOADI_MOD0_LOWER,
        (kD3Bits & 0xffffu));
    TTI_SFPMAD(
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG14,
        ckernel::p_sfpu::LREG1,
        0);  // n=u+n4
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG12,
        ckernel::p_sfpu::LREG2,
        0);  // d=d3*u+d2
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG1,
        0);
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG13,
        ckernel::p_sfpu::LREG2,
        0);
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG1,
        0);
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG2,
        0);
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LREG1,
        0);
    TTI_SFPARECIP(0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG7, 0);
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LREG1,
        0);
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG2,
        1);  // e=1-d*r0
    // E2 is exact BF16, and u is dead after the final numerator rung.  Its
    // one-word load fills the residual/refine dependency window.
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG0,
        sfpi::SFPLOADI_MOD0_FLOATB,
        (TT_BIASED_LOG_EXP_E2_BITS >> 16));
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG7,
        0);  // r=r0+e*r0
    TTI_SFPNOP;
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG2,
        0);  // biased log coordinate
    // E1 is exactly BF16 and its one-word immediate load is the quotient gap.
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG7,
        sfpi::SFPLOADI_MOD0_FLOATB,
        (TT_BIASED_LOG_EXP_E1_BITS >> 16));
    TTI_SFPEXEXP(0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG1, 0);
    TTI_SFPEXMAN(0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG2, 0);
    TTI_SFPSHFT(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG2, 0);
    // Retain the fixed-point z bits in L2: SETEXP below consumes their biased
    // exponent directly, avoiding a separate factor shift and multiply.
    TTI_SFPEXMAN(
        0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG1,
        sfpi::SFPEXMAN_MOD1_PAD9);
    TTI_SFPCAST(ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG1, 0);
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG7,
        0);  // e2*f+e1
    TTI_SFPNOP;
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG7,
        0);  // p=1+f*(e1+e2*f)
    TTI_SFPSETEXP(
        0,
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG2,
        2);  // y=p*2^i -> L2; p is certified in [1,2)
    // L1 is dead after SETEXP and is a safe low-bank macro carrier. Its next
    // iteration numerator write occurs after the delayed RNE/store retire.
    TTI_SFPLOADMACRO(1, 0, TT_ADDRMOD_DST_ADV, 0);  // VD=L1, literal result=L2
    TT_REPLAY_REST(kBodySlots, 31);
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TT_REPLAY_TILE_EPILOGUE();
}

#if defined(TT_TARGET_COMPOSITE_NATIVE_EVEN_POLYVAL_EXTERIOR)
static_assert(TT_TARGET_COMPOSITE_RAW_SHADOW_ROWS == 32u);
static_assert(
    TT_TARGET_COMPOSITE_RAW_SHADOW_ROW_BASE >= 64u &&
    TT_TARGET_COMPOSITE_RAW_SHADOW_ROW_BASE +
            TT_TARGET_COMPOSITE_RAW_SHADOW_ROWS <=
        TT_TARGET_COMPOSITE_DST_CAPACITY);
static_assert(TT_TARGET_COMPOSITE_POLY_DEGREE == 10u);
static_assert(TT_TARGET_COMPOSITE_COEFF_ROWS == 11u);
static_assert(
    TT_TARGET_COMPOSITE_COEFF_ROW_BASE >=
        TT_TARGET_COMPOSITE_RAW_SHADOW_ROW_BASE +
            TT_TARGET_COMPOSITE_RAW_SHADOW_ROWS);
static_assert(
    TT_TARGET_COMPOSITE_COEFF_ROW_BASE + TT_TARGET_COMPOSITE_COEFF_ROWS <=
        TT_TARGET_COMPOSITE_DST_CAPACITY);

template <int I>
inline void target_composite_preload_native_even_polyval_coefficient() {
#if !defined(TT_TARGET_COMPOSITE_DST_COEFF_DISABLE)
    if constexpr (I < 11) {
        dst_reg[TT_TARGET_COMPOSITE_COEFF_ROW_BASE + I] =
            vFloat(TT_TARGET_COMPOSITE_POLY_COEFFS[I]);
        target_composite_preload_native_even_polyval_coefficient<I + 1>();
    }
#endif
}

inline void target_composite_preload_native_even_polyval_coefficients_tile() {
    // One exact FP32 broadcast/store per descriptor coefficient per tile.
    // The subsequent Horner loads are semantically identical to literal
    // materialization but avoid rematerializing 11 FP32 literals on every row.
    target_composite_preload_native_even_polyval_coefficient<0>();
}

template <int I>
sfpi_inline vFloat target_composite_poly_coefficient() {
#if defined(TT_TARGET_COMPOSITE_DST_COEFF_DISABLE)
    return TT_TARGET_COMPOSITE_POLY_COEFFS[I];
#else
    return dst_reg[TT_TARGET_COMPOSITE_COEFF_ROW_BASE + I];
#endif
}

inline void target_composite_prepare_native_even_polyval_exterior_tile() {
#pragma GCC unroll 0
    for (int d = 0; d < 32; d++) {
        vUInt raw_u16 = dst_reg[d].mode<DataLayout::U16>();
        // SFPI's encoded U16 view carries the BF16 exponent in the low byte
        // (the same physical layout consumed by domain_actions.h).
        vUInt raw_exponent = raw_u16 & vUInt(0x00ffu);
        vFloat input = prepare_raw_domain_input(dst_reg[d]);
        vFloat coordinate = input * input;
        vFloat polynomial = target_composite_poly_coefficient<10>();
        polynomial = polynomial * coordinate + target_composite_poly_coefficient<9>();
        polynomial = polynomial * coordinate + target_composite_poly_coefficient<8>();
        polynomial = polynomial * coordinate + target_composite_poly_coefficient<7>();
        polynomial = polynomial * coordinate + target_composite_poly_coefficient<6>();
        polynomial = polynomial * coordinate + target_composite_poly_coefficient<5>();
        polynomial = polynomial * coordinate + target_composite_poly_coefficient<4>();
        polynomial = polynomial * coordinate + target_composite_poly_coefficient<3>();
        polynomial = polynomial * coordinate + target_composite_poly_coefficient<2>();
        polynomial = polynomial * coordinate + target_composite_poly_coefficient<1>();
        polynomial = polynomial * coordinate + target_composite_poly_coefficient<0>();
        vFloat target_result = 1.0f + polynomial * coordinate;
        target_result = convert<vFloat16b>(target_result, RoundMode::Nearest);

        // Carry the actual rounded graph output across the replay. A class
        // marker is insufficient: the selected replay is +Inf on target-
        // finite exterior rows, and an arbitrary finite witness would not be
        // a semantic lowering of the declared composition.
        vFloat overlay = 0.0f;
        vFloat magnitude = setsgn(input, 0);
        v_if(magnitude >= __builtin_bit_cast(float, TT_BIASED_LOG_TERMINAL_BITS)) {
            v_if(raw_exponent != 0xffu) { overlay = target_result; }
            v_endif;
        }
        v_endif;
        dst_reg[TT_TARGET_COMPOSITE_RAW_SHADOW_ROW_BASE + d] = overlay;
    }
}

inline void target_composite_apply_native_even_polyval_exterior_tile() {
#pragma GCC unroll 8
    for (int d = 0; d < 32; d++) {
        vFloat result = dst_reg[d];
        vFloat overlay = dst_reg[TT_TARGET_COMPOSITE_RAW_SHADOW_ROW_BASE + d];
        v_if(!is_zero(overlay)) { result = overlay; }
        v_endif;
        dst_reg[d] = result;
    }
}
#elif defined(TT_TARGET_COMPOSITE_NATIVE_EVEN_CLASS_REP)
static_assert(TT_TARGET_COMPOSITE_RAW_SHADOW_ROW_BASE == 32u);
static_assert(TT_TARGET_COMPOSITE_RAW_SHADOW_ROWS == 32u);
static_assert(TT_TARGET_COMPOSITE_DST_CAPACITY >= 64u);
static_assert(
    TT_TARGET_COMPOSITE_CLASS_REP_BODY_SLOTS == 8u &&
    TT_TARGET_COMPOSITE_CLASS_REP_PEAK_LREGS <= 8u);
#if !defined(TT_TARGET_COMPOSITE_PRESERVES_BIASED_LOG_LREGS)
#error "class-representative scratch must preserve selected replay L3..L6"
#endif
static_assert(TT_TARGET_COMPOSITE_FINITE_FIRST_RAW > 0u);
static_assert(
    TT_TARGET_COMPOSITE_FINITE_FIRST_RAW <=
    TT_TARGET_COMPOSITE_FINITE_LAST_RAW);
static_assert(
    TT_TARGET_COMPOSITE_FINITE_END_RAW ==
    TT_TARGET_COMPOSITE_FINITE_LAST_RAW + 1u);
static_assert(TT_TARGET_COMPOSITE_FINITE_LAST_RAW < 0x7f80u);
static_assert(
    TT_TARGET_COMPOSITE_FINITE_REPRESENTATIVE > 0u &&
    TT_TARGET_COMPOSITE_FINITE_REPRESENTATIVE < 0x7f80u);
static_assert(
    TT_TARGET_COMPOSITE_CLASS_PATTERN_COUNT ==
    2u * (TT_TARGET_COMPOSITE_FINITE_LAST_RAW -
          TT_TARGET_COMPOSITE_FINITE_FIRST_RAW + 1u));

__attribute__((always_inline)) inline void
target_composite_native_even_class_representative_body() {
    TTI_SFPLOAD(
        ckernel::p_sfpu::LREG1,
        0,
        TT_ADDRMOD_DST_HOLD,
        2 * TT_TARGET_COMPOSITE_RAW_SHADOW_ROW_BASE);  // decoded raw input
    TTI_SFPLOAD(
        ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_HOLD, 0);  // selected BF16
    TTI_SFPSETSGN(
        0,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG1,
        1);  // magnitude; selected load is the raw-load consumer gap
    TTI_SFPLE(
        0,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG7,
        1);  // lower <= magnitude
    TTI_SFPGT(
        0,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG2,
        1);  // magnitude < end
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG0,
        sfpi::SFPLOADI_MOD0_FLOATB,
        TT_TARGET_COMPOSITE_FINITE_REPRESENTATIVE);
    TTI_SFPENCC(0, 0, 0, 0);
    TTI_SFPSTORE(
        ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_ADV, 0);
}

inline void target_composite_apply_native_even_class_representative_tile() {
    // The abstract quotient is one sign-mirrored contiguous BF16 interval.
    // Lower it as one fixed body, using the first raw word and the exclusive
    // successor of the last word as exact BF16 comparison boundaries.  The
    // selected result and raw shadow are loaded in the same row body; no
    // high-level predicate tree or observed-value table remains.
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG7,
        sfpi::SFPLOADI_MOD0_FLOATB,
        TT_TARGET_COMPOSITE_FINITE_FIRST_RAW);
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG2,
        sfpi::SFPLOADI_MOD0_FLOATB,
        TT_TARGET_COMPOSITE_FINITE_END_RAW);
#if defined(TARGET_COMPOSITE_CLASS_REP_DISABLE)
#pragma GCC unroll 32
    for (int d = 0; d < 32; ++d) {
        target_composite_native_even_class_representative_body();
    }
#else
    TT_REPLAY_RECORD(TT_TARGET_COMPOSITE_CLASS_REP_BODY_SLOTS);
    target_composite_native_even_class_representative_body();
    TT_REPLAY_REST(TT_TARGET_COMPOSITE_CLASS_REP_BODY_SLOTS, 31);
#endif
    TT_REPLAY_TILE_EPILOGUE();
}
#endif
#else
#define BIASED_LOG_RATIONAL_EXP_TTI_CANDIDATE 0
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

// ============================================================================
// DST-RESIDENT COEFFICIENT STORE (BH + bf16 + fp32-dest, EMBEDDED_LUT)
// ============================================================================
// Multi-segment rational cascades pay 2 SFPLOADI slots per fp32 coefficient
// per element: the immediates rematerialize inside the 32-element dst loop
// (erfc n4d4_s8: 68 non-{0,1} coefficients = 136 of the per-element slots).
// Instead: park each non-{0,1} coefficient ONCE in an unused dst row above
// the data tile (dst_reg[32 + slot], i.e. dst tiles 1..3), and fetch it
// in-body with a single SFPLOAD (immediate row offset, NOINC addr mod) —
// 1 slot per use instead of 2. Coefficients exactly 0.0f / 1.0f keep the
// literal path (GCC folds them onto the hardware constant LREGs, exactly as
// the immediate path does today).
//
// REQUIRES fp32 dest accumulation: in 16-bit dest mode a dest row quantizes
// the stored fp32 coefficient to bf16 (none of erfc's coefficients are
// bf16-exact) and byte-identity dies. The host flips fp32_dest_acc_en for
// bf16 runs under the compiler-derived host ABI bit; the dispatch checks the generated
// constexpr DST_ACCUM_MODE and falls back to the plain sfpi path when the
// host didn't flip the dest mode (mechanism silently off, never wrong).
//
// FLIP BYTE-NEUTRALITY BAND (wave-10, mish root cause): under fp32 dest
// with a bf16 input CB the A2D datacopy writes x + 2^-127 into the dest —
// a constant half-min-normal bias on every element (BH silicon proof via an
// identity rational: every flip-arm output equals bf16(x + 2^-127) with
// denormal results flushing at the pack fp32->bf16). RNE absorbs the bias
// completely once |value| >= 2^-118, so ordinary-magnitude fits are
// bit-identical under the flip; fits whose output sweeps the near-zero band
// diverge (mish, n0 == 0: fixed pure ULP 166.0 on every single-segment
// rational). The CODEGEN therefore only emits DST_COEFF_STORE + the host
// flip when no nonzero modeled output over the deployment bf16 grid enters
// (0, 2^-110) in magnitude (run_csv.sh _dst_flip_band_hits). This kernel
// cannot check that value-range fact at compile time; the define's absence
// keeps the byte-frozen baseline (C-LOW-2). A future variant can lift the
// band restriction by keeping bf16 dest and moving the parked rows with
// MOD0_FMT_FP32 per-instruction overrides, exactly as the generic-path
// (poly) variant in dst_coeff_store.h already does.
//
// DOUBLE-BUFFER: DstSync::SyncHalf alternates dest halves per tile, and all
// SFPU dst addressing is relative to the current half's base
// (DEST_TARGET_REG_CFG_MATH_Offset), so a coefficient row written in one
// half does not exist in the other. DEFAULT = re-store the rows EVERY tile
// (right after the A2D datacopy, inside the same acquire window as the reads
// — no cross-tile persistence assumption; measured cost 3 slots per
// coefficient per tile ≈ +0.01us on erfc n3d4_s3). DST_COEFF_PRELOAD_ONCE
// instead writes only during the first two tiles (one per half): silicon-
// verified byte-identical through the {1,2,8,145,146,256,512} ladder and
// 10x512 determinism, but note the pack thread issues ZEROACC(CLR_HALF) on
// the half after every packed tile (_llk_pack_dest_section_done_,
// llk_pack_common.h) — on BH silicon that does NOT destroy SFPU-readable
// row contents (zero-flag semantics), which preload-once relies on; the
// per-tile refresh default does not. Nothing else touches rows above
// tile 0: the A2D datacopy writes tile 0 only and the packer only reads.
//
// Capacity: fp32-dest SyncHalf on BH = 4 tiles = 128 dst indices; tile 0 is
// data, so up to 96 coefficient rows fit (n4d4_s8 needs 68).
//
// DST_COEFF_DISABLE compiles the whole mechanism out (byte-identity escape).
#if defined(EMBEDDED_LUT) && defined(USE_BF16) && defined(ARCH_BLACKHOLE) &&    \
    defined(DST_COEFF_STORE) && !defined(DST_COEFF_DISABLE) &&                  \
    !defined(RANGE_REDUCTION_EXP) && !defined(RANGE_REDUCTION_TRIG) &&          \
    !defined(RANGE_REDUCTION_LOG) &&                                            \
    !defined(RATIONAL_NUM_PARITY_ODD) && !defined(RATIONAL_DEN_PARITY_EVEN) &&  \
    !defined(PRECOMPOSE_INPUT_AFFINE) && !defined(POSTCOMPOSE_AFFINE_Y) &&      \
    !defined(POSTCOMPOSE_AFFINE_Y_TIMES_INPUT)
#define TT_RATIONAL_DST_COEFF 1
#else
#define TT_RATIONAL_DST_COEFF 0
#endif

#if TT_RATIONAL_DST_COEFF
constexpr uint32_t kDstCoeffLutBase = NUM_SEGMENTS + 1;  // boundary points first
constexpr uint32_t kDstCoeffCount = (NUM_DEGREE + DEN_DEGREE + 2u) * NUM_SEGMENTS;
// dst tile 1 (tile 0 holds the data), or tile 2 when FUSE_GRAD_MUL owns tile 1
// for the incoming grad. Same collision as
// dst_coeff_store.h::kDstCoeffBase -- see the comment there for the measured
// failure mode.
#if defined(FUSE_GRAD_MUL)
constexpr uint32_t kDstCoeffRowBase = 64;
#else
constexpr uint32_t kDstCoeffRowBase = 32;
#endif

// 0.0f / 1.0f ride the hardware constant LREGs (bitwise check: -0.0f must NOT
// match +0.0f — a -0.0 addend changes the sign of an exact-zero FMA result).
constexpr bool dst_coeff_is_special(float v) {
    const uint32_t bits = __builtin_bit_cast(uint32_t, v);
    return bits == 0x00000000u || bits == 0x3F800000u;
}

// Dense dst slot for coefficient j (LUT index kDstCoeffLutBase + j): number of
// non-special coefficients before j.
constexpr uint32_t dst_coeff_slot(uint32_t j) {
    uint32_t s = 0;
    for (uint32_t k = 0; k < j; k++) {
        if (!dst_coeff_is_special(LUT_DATA[kDstCoeffLutBase + k])) {
            s++;
        }
    }
    return s;
}
constexpr uint32_t kDstCoeffRows = dst_coeff_slot(kDstCoeffCount);
static_assert(
    kDstCoeffRowBase + kDstCoeffRows <= 128,
    "DST coefficient rows exceed fp32-dest half capacity (4 tiles)");

// Fetch coefficient j: literal for 0/1, single SFPLOAD from the parked dst row
// for everything else. The row index is a compile-time constant, so GCC emits
// one immediate-offset SFPLOAD (NOINC — the D-RWC walk is untouched).
template <uint32_t J>
sfpi_inline vFloat dst_coeff_fetch() {
    constexpr float v = LUT_DATA[kDstCoeffLutBase + J];
    if constexpr (dst_coeff_is_special(v)) {
        return vFloat(v);
    } else {
        return dst_reg[kDstCoeffRowBase + dst_coeff_slot(J)];
    }
}

// Preload: 2xSFPLOADI + SFPSTORE per non-special coefficient (all lanes hold
// the broadcast value; runs outside any v_if so CC is all-on).
template <uint32_t J>
inline void dst_coeff_preload_from() {
    if constexpr (J < kDstCoeffCount) {
        constexpr float v = LUT_DATA[kDstCoeffLutBase + J];
        if constexpr (!dst_coeff_is_special(v)) {
            dst_reg[kDstCoeffRowBase + dst_coeff_slot(J)] = vFloat(v);
        }
        dst_coeff_preload_from<J + 1>();
    }
}

// Descending Horner rung chain: acc = acc*x + c[I], I down to STOP inclusive.
template <uint32_t JBASE, int I, int STOP>
sfpi_inline void dst_horner_chain(vFloat x, vFloat& acc) {
    if constexpr (I >= STOP) {
        acc = acc * x + dst_coeff_fetch<JBASE + static_cast<uint32_t>(I)>();
        dst_horner_chain<JBASE, I - 1, STOP>(x, acc);
    }
}

// Interleaved lockstep: both chains step down together (SFPMAD latency hiding,
// exactly the eval_rational_interleaved_numer_denom dataflow).
template <uint32_t NB, uint32_t DB, int I>
sfpi_inline void dst_horner_lockstep(vFloat x, vFloat& numer, vFloat& denom) {
    if constexpr (I >= 0) {
        numer = numer * x + dst_coeff_fetch<NB + static_cast<uint32_t>(I)>();
        denom = denom * x + dst_coeff_fetch<DB + static_cast<uint32_t>(I)>();
        dst_horner_lockstep<NB, DB, I - 1>(x, numer, denom);
    }
}

// P(x), Q(x) for segment SEG — per-chain FMA sequence is EXACTLY that of
// eval_rational_interleaved_numer_denom (same values, same operation order,
// same bits); only the coefficient operand source changes.
template <uint32_t SEG>
sfpi_inline void eval_rational_numer_denom_dst(vFloat x, vFloat& out_numer, vFloat& out_denom) {
    constexpr uint32_t NUM_COEFFS = NUM_DEGREE + 1u;
    constexpr uint32_t CPS = NUM_COEFFS + DEN_DEGREE + 1u;
    constexpr uint32_t NB = SEG * CPS;               // numerator coefficient base (j-space)
    constexpr uint32_t DB = NB + NUM_COEFFS;         // denominator coefficient base
    constexpr int MIN_DEG = (NUM_DEGREE < DEN_DEGREE) ? NUM_DEGREE : DEN_DEGREE;

    vFloat numer = dst_coeff_fetch<NB + NUM_DEGREE>();
    vFloat denom = dst_coeff_fetch<DB + DEN_DEGREE>();
    if constexpr (NUM_DEGREE > DEN_DEGREE) {
        dst_horner_chain<NB, static_cast<int>(NUM_DEGREE) - 1, static_cast<int>(DEN_DEGREE)>(x, numer);
    } else if constexpr (DEN_DEGREE > NUM_DEGREE) {
        dst_horner_chain<DB, static_cast<int>(DEN_DEGREE) - 1, static_cast<int>(NUM_DEGREE)>(x, denom);
    }
    dst_horner_lockstep<NB, DB, MIN_DEG - 1>(x, numer, denom);
    out_numer = numer;
    out_denom = denom;
}

// Segment cascade, deferred reciprocal — the unroll_segment_rational_deferred
// twin (same v_if predication, boundary compares stay immediates).
template <uint32_t SEG>
inline void dst_unroll_segments(vFloat x, vFloat& numer, vFloat& denom) {
    if constexpr (SEG < NUM_SEGMENTS) {
        v_if(x >= LUT_DATA[SEG]) { eval_rational_numer_denom_dst<SEG>(x, numer, denom); }
        v_endif;
        dst_unroll_segments<SEG + 1>(x, numer, denom);
    }
}

inline void piecewise_rational_dst_coeff_tile(bool dst_preload) {
#if defined(TT_INLINE_PROGRAM_RATIONAL_LATE_RAW_SCHEDULE)
#error "raw BF16 rational suffix is incompatible with fp32-destination DST coefficients"
#endif
    if (dst_preload) {
        dst_coeff_preload_from<0>();
    }
    for (int d = 0; d < 32; d++) {
        vFloat x_raw = dst_reg[d];
#if defined(TT_SELECTED_CORE_BRIDGE_EXPONENT_REPLAY_TAIL_1)
        vInt selected_core_exponent_delta;
        vFloat x = selected_core_exponent_replay_prepare(
            x_raw, selected_core_exponent_delta);
#else
        vFloat x = prepare_raw_domain_input(x_raw);
#endif
#if defined(TT_RATIONAL_COORDINATE_BOUND)
        x = rational_bound_coordinate(x);
#endif
        vFloat numer, denom;
        eval_rational_numer_denom_dst<0>(x, numer, denom);
        dst_unroll_segments<1>(x, numer, denom);
        // ONE deferred reciprocal, as the baseline path does.
        vFloat result = numer * rational_reciprocal(denom);
        // Gate requires USE_BF16: RNE narrowing before the store (the fp32
        // dest row then holds a bf16-representable value, so the packer's
        // fp32->bf16 conversion is exact — byte-identical to the bf16-dest
        // baseline's SFPSTORE truncation of the same rounded value).
#if defined(TT_SELECTED_CORE_BRIDGE_EXPONENT_REPLAY_TAIL_1)
        selected_core_exponent_replay_finalize(
            x_raw, selected_core_exponent_delta, result);
#else
        finalize_raw_domain_actions(x_raw, result);
#endif
        result = convert<vFloat16b>(result, RoundMode::Nearest);
        dst_reg[d] = result;
    }
}

#endif  // TT_RATIONAL_DST_COEFF

// ----------------------------------------------------------------------------
// TTI REPLAY lowering for the SINGLE-SEGMENT parity rational (odd numerator /
// even denominator x^2-Horner) — the selected tan R5/4 shape (collapses to
// n5d2 under parity), extensible to n7d2 (asin/logit).
//
// PREDICATION PILOT: the body carries the reciprocal's NaN-guard CC block
// in-replay. Sanctioned per the audit iff (a) the body is CC-balanced
// (SFPSETCC ... SFPENCC fully inside the recorded body — it is) and (b) the
// SFPSETCC reading a MAD result gets the rule-1 hazard gap (it gets 3).
//
// Body = 1:1 transcription of the sfpi/GCC BH tan_n5d2 loop (objdump):
//   s = x*x; num = ((n5*s + n3)*s + n1)*x  [odd parity];
//   den = d2*s + d0;
//   recip = sfpu_reciprocal_iter<2>(den)   [ARECIP + 2 guarded Newton steps,
//           exact GCC MAD mod encodings 2/3/1/2 preserved];
//   y = num * recip; STOCH_RND bf16; STORE.
// The sfpi twin materializes every non-unit coefficient in-body (no prgm/pin
// budget is spent), so the replay body does the same via SFPLOADI pairs — they
// double as the MAD-chain gap slots. The rational's common-scale gauge gives
// two structurally equivalent schedules with the SAME slot/register budget:
//   * d0 == 1: d0 rides LCONST_1; the final rung loads arbitrary n1;
//   * n1 == 1: n1 rides LCONST_1; those two LOADI slots load arbitrary d0.
// Emit-time origin normalization intentionally selects the second gauge to
// prevent n1*x underflow. The choice is constexpr, never a per-element branch.
// prgm0 (LREG12) holds 2.0 programmed by sfpu_reciprocal_init and is never
// written here.
// Slots = 23 + 3*NSTEPS plain: n5d2 = 29, n7d2 = 32. Registers: L2 = x
// (-> recip result), L3 = s (-> Newton y1), L0 = num acc, L1 = den acc,
// L4 = coeff scratch (-> Newton t).
//
// SFPLOADMACRO TAIL FOLD (wave 8, "LOADMACRO Pilot B"): a body RESCHEDULE
// frees three slots -> slots = 20 + 3*NSTEPS (n5d2 = 26, n7d2 = 29,
// n9d2 = 32). This ADMITS NSTEPS == 4 (the i1/erfinv n9d2 fits, whose plain
// body at 35 slots does NOT fit the 32-slot replay buffer) — for those
// shapes the LM tail is the replay UNLOCK, and there is no plain-replay
// control arm; the A/B control is the sfpi path via RATIONAL_TTI_DISABLE.
//
// SILICON-PROVEN ENCODING CONSTRAINT (falsified the phase-1 scout's D4-B
// "both sources literal" MAD template on chip 1, 8-tile byte-identity —
// output was exactly 2x): per the ISA sequence-register fields, a macro-fired
// instruction ALWAYS has one source overwritten by the LOADMACRO's VD —
// bit7 (SRCB_OVERRIDE) = 1 -> VD becomes Source B, = 0 -> VD becomes Source C.
// There is NO no-substitution mode, so `y = num * recip` (two live literal
// sources + zero addend) cannot fire from the MAD slot. The redesign folds
// only STOCH_RND + STORE into the macro (STOCH_RND's input is Source C;
// SRCB_OVERRIDE=1 keeps it literal), and pays for the third slot by
// RESCHEDULING: den's MAD moves right after its SFPLOADI pair (s is ready),
// ARECIP (dest now L4) becomes the last-rung -> num-MUL hazard gap, and the
// num MUL doubles as the ARECIP result window. Register remap in the guard:
// r0 = L4, t = L2 (x is dead there), recip = L4; operand ROLES and MAD mod
// encodings are unchanged, so every op computes bit-identical values — the
// fold is a pure reschedule and byte-identity vs the sfpi twin is preserved.
//
// SHIPPING GATE (silicon-measured): the LM tail engages ONLY for NSTEPS == 4
// (n9 unlock). The fold was fully battery-verified on tan/asin (29-slot LM
// bodies, byte-identical + clean) but measured ZERO runtime gain over their
// 32-slot plain bodies (2.00us both at 256x10) — so shapes that fit plain
// keep the wave-6 body and zero LOADMACRO exposure. RAT_LM_DISABLE (env ->
// codegen) vetoes the LM tail entirely: n9 shapes then fail the capacity
// gate and fall back to the sfpi path (the A/B control arm — there is no
// plain-replay arm for 35-slot bodies).
// CC discipline (audit D2): the LM issues two slots AFTER SFPENCC; its
// in-flight ops (max delay 2 -> store fires at issue+3) land in the next
// iteration's CC-free head, ~20 slots before that iteration's SFPSETCC.
// The macro sequence is fully self-contained (Round reads L0 written >= 2
// cycles before the LM; Store reads VD) — safe under either UnitDelayKind
// interpretation (cycle- or instruction-counted delays).
// RATIONAL_TTI_DISABLE (env -> codegen) forces the sfpi path for A/B.
#if defined(EMBEDDED_LUT) && defined(USE_BF16) && defined(ARCH_BLACKHOLE) &&      \
    defined(RATIONAL_NUM_PARITY_ODD) && defined(RATIONAL_DEN_PARITY_EVEN) &&      \
    !defined(RATIONAL_TTI_DISABLE) &&                                             \
    !defined(RANGE_REDUCTION_EXP) && !defined(RANGE_REDUCTION_TRIG) &&            \
    !defined(RANGE_REDUCTION_LOG) && !defined(PRECOMPOSE_INPUT_AFFINE) &&         \
    !defined(POSTCOMPOSE_AFFINE_Y) && !defined(POSTCOMPOSE_AFFINE_Y_TIMES_INPUT) && \
    !defined(RATIONAL_RECIPROCAL_ZERO_ITER) && !defined(RATIONAL_RECIPROCAL_ONE_ITER) && \
    !defined(TT_DOMAIN_ACTION_PROGRAM)
#define RATIONAL_TTI_CANDIDATE 1

constexpr uint32_t kRatTtiCoeffOff = NUM_SEGMENTS + 1;  // boundary points first
constexpr float rat_tti_num_c(uint32_t k) { return LUT_DATA[kRatTtiCoeffOff + k]; }
constexpr float rat_tti_den_c(uint32_t k) { return LUT_DATA[kRatTtiCoeffOff + NUM_DEGREE + 1u + k]; }

constexpr uint32_t kRatTtiNumTop = (NUM_DEGREE % 2u == 1u) ? NUM_DEGREE : NUM_DEGREE - 1u;
constexpr uint32_t kRatTtiNumSteps = (kRatTtiNumTop - 1u) / 2u;  // num rung MADs after init
constexpr uint32_t kRatTtiDenTop = (DEN_DEGREE % 2u == 0u) ? DEN_DEGREE : DEN_DEGREE - 1u;
constexpr bool kRatTtiUnitDenOrigin = (rat_tti_den_c(0) == 1.0f);
constexpr bool kRatTtiUnitNumLinear = (rat_tti_num_c(1) == 1.0f);
// Prefer the byte-frozen d0==1 schedule when both coefficients are unit. The
// numerator-gauge schedule exists precisely for normalized d0!=1 artifacts.
constexpr bool kRatTtiUsesUnitNumeratorGauge =
    !kRatTtiUnitDenOrigin && kRatTtiUnitNumLinear;
// The LM tail engages ONLY where it unlocks capacity (NSTEPS == 4, the n9
// shapes): silicon A/B measured ZERO runtime gain from the -3 fold on shapes
// that already fit plain (tan/asin 29-slot LM = 32-slot plain = 2.00us at
// 256x10 — these bodies are not issue-slot-bound), so smaller shapes keep the
// byte-frozen wave-6 plain body and zero LOADMACRO exposure.
#ifndef RAT_LM_DISABLE
constexpr bool kRatTtiLmTail = (kRatTtiNumSteps >= 4u);
#else
constexpr bool kRatTtiLmTail = false;  // full veto (A/B escape; n9 -> sfpi)
#endif
constexpr uint32_t kRatTtiBodySlots =
    23u + 3u * kRatTtiNumSteps - (kRatTtiLmTail ? 3u : 0u);

// NSTEPS window: the >= 2u lower bound is a VERIFICATION boundary, not a
// capacity one. An NSTEPS == 1 (n3d2) plain body is 23 + 3*1 = 26 slots and
// fits the 32-slot replay buffer; the rung emitter would even produce a
// well-formed num = (n3*s + n1)*x body (rung 0 is unconditional below). The
// window was born as [2, 3] with this template (wave 6) covering exactly the
// objdump-twinned, silicon-verified shapes (tan n5d2/n7d2, asin n7d2) and was
// later raised to 4 by the LOADMACRO n9 unlock; no n3d2 variant was ever
// emitted, objdump-diffed, or A/B-measured, and no selected Table-2 row has
// that shape (git: the bound is unchanged since the body landed; there is no
// "measured slower" receipt). Extending it is a byte-identity + silicon
// verification task, not a slot problem — until then n3d2 stays a census
// fallback (analysis/lowering_census.csv, C-LOW-3).
constexpr bool kRationalTtiReplay = (NUM_SEGMENTS == 1) && (kRatTtiDenTop == 2u) &&
                                    (kRatTtiUnitDenOrigin || kRatTtiUnitNumLinear) &&
                                    (kRatTtiNumSteps >= 2u) &&
                                    (kRatTtiNumSteps <= (kRatTtiLmTail ? 4u : 3u)) &&
                                    tti_replay_fits(kRatTtiBodySlots);
#if defined(TT_INLINE_PROGRAM_OVERLAY)
#if defined(TT_INLINE_PROGRAM_TTI_RATIONAL_SYMMETRIC_SATURATION_OVERLAY)
// Verifier-certified symmetric terminal tails whose target-rounded values
// equal the mature rational at x=+/-B.  The first hardware closure is B=1:
// LCONST_neg1/LCONST_1 let the original input register be clamped in two
// slots, with no raw save, coefficient register, post-select, or new loop.
// Only n5d2 has two free slots in its byte-frozen 29-slot plain body.  Larger
// bodies fail closed; they are never rescheduled onto the LOADMACRO variant.
constexpr uint32_t kRatTtiOverlayBodySlots = kRatTtiBodySlots + 2u;
constexpr bool kRatTtiOverlayReplay =
    kRationalTtiReplay && (kRatTtiNumTop == 5u) && (kRatTtiDenTop == 2u) &&
    !kRatTtiLmTail &&
    (TT_PROGRAM_RATIONAL_TTI_CORE_BOUND_BITS == 0x3f800000u) &&
    tti_replay_fits(kRatTtiOverlayBodySlots);
static_assert(
    !kRationalTtiReplay || kRatTtiOverlayReplay,
    "typed symmetric RATIONAL_TTI overlay exceeds replay/register closure");
#else
static_assert(
    !kRationalTtiReplay,
    "typed core overlay requires route-aware RATIONAL_TTI replay; refusing silent fallback");
#endif
#endif

// One numerator x^2-Horner rung: SFPLOADI the addend n[NTOP-2(J+1)] into L4
// (bit-unit, gap-0 readable; the pair is also the MAD-chain gap), then
// acc = acc*s + addend.
template <uint32_t J>
inline void rat_tti_num_rung() {
    constexpr uint32_t kIndex = kRatTtiNumTop - 2u * (J + 1u);
    if constexpr (kRatTtiUsesUnitNumeratorGauge && kIndex == 1u) {
        TTI_SFPMAD(
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG3,
            ckernel::p_sfpu::LCONST_1,
            ckernel::p_sfpu::LREG0,
            0);  // final nacc = nacc*s + 1.0; the two saved slots load d0
    } else {
        constexpr uint32_t bits = __builtin_bit_cast(uint32_t, rat_tti_num_c(kIndex));
        TTI_SFPLOADI(ckernel::p_sfpu::LREG4, SFPLOADI_MOD0_UPPER, (bits >> 16));
        TTI_SFPLOADI(ckernel::p_sfpu::LREG4, SFPLOADI_MOD0_LOWER, (bits & 0xFFFFu));
        TTI_SFPMAD(
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG3,
            ckernel::p_sfpu::LREG4,
            ckernel::p_sfpu::LREG0,
            0);  // nacc = nacc*s + n[NTOP-2(J+1)]
    }
}

// One-time SFPLOADMACRO arming for the tail fold (macro 0 only). Called from
// kernel_main's TRISC_MATH init AFTER sfpu_reciprocal_init (capture is
// backdoor-only: the template issue does NOT execute and does NOT write L13
// — in-tree precedent ckernel_sfpu_exp.h / _recip.h).
//   T1 (backdoor dest=13 -> instruction register 5, seq mux 5):
//      SFP_STOCH_RND(RND_EVEN, srcB=dontcare, srcC=L0, ->VD, FP32_TO_FP16B)
//      bf16 RNE of y in L0. At fire time SRCB_OVERRIDE=1 puts the LM's VD
//      into Source B (unused by the fp32->bf16 convert) so Source C stays
//      the LITERAL L0 — the mandatory VD source substitution lands on a
//      dead operand. Dest is substituted by VD (L4).
//   Sequence register 0 (config_dest 4) = 0x13850000:
//      simple = 0x00 idle
//      mad    = 0x00 idle
//      round  = 0x85 = 0b1_0_000_101: T1 (mux 5), delay 0 -> fires issue+1
//               (2 cycles after the stream's `MUL y` — the plain body's gap),
//               SRCB_OVERRIDE=1, dest = VD
//      store  = 0x13 = 0b0_0_010_011: fixed STORE (mux 3), delay 2 -> fires
//               issue+3, source = VD (no staging) — same byte exp.h seq-1 uses
//   Misc (config_dest 8) = 0x110: StoreMod0 = 0,
//      UsesLoadMod0ForStore[macro0] = 1 (store inherits the LM's DEFAULT
//      mod0 — byte-identical format to the plain SFPSTORE it replaces),
//      UnitDelayKind[macro0] = 1 (WaitForElapsedInstructions: delays count
//      issued SFPU instructions, so the SFPNOP drain is exact under any
//      frontend hiccup; the sequence is also self-contained, so it is safe
//      under the per-unit cycle-counted reading as well).
inline void rational_tti_lm_init() {
    TTI_SFP_STOCH_RND(
        SFPSTOCHRND_RND_EVEN,
        0,
        0,
        ckernel::p_sfpu::LREG0,
        13,
        SFPSTOCHRND_MOD1_FP32_TO_FP16B);  // capture T1 (does not execute)
    TTI_SFPLOADI(0, SFPLOADI_MOD0_LOWER, 0x0000);  // mad | simple (idle)
    TTI_SFPLOADI(0, SFPLOADI_MOD0_UPPER, 0x1385);  // store | round
    TTI_SFPCONFIG(0, 4, 0);      // -> Macro Sequence Register 0
    TTI_SFPCONFIG(0x110, 8, 1);  // -> LoadMacroConfig.Misc
}

inline void rational_tti_replay_tile() {
    constexpr uint32_t kNumTopBits = __builtin_bit_cast(uint32_t, rat_tti_num_c(kRatTtiNumTop));
    constexpr uint32_t kDenTopBits = __builtin_bit_cast(uint32_t, rat_tti_den_c(2));
    constexpr uint32_t kDenOriginBits = __builtin_bit_cast(uint32_t, rat_tti_den_c(0));
#if defined(TT_INLINE_PROGRAM_TTI_RATIONAL_SYMMETRIC_SATURATION_OVERLAY)
    constexpr uint32_t kRatTtiRecordedBodySlots = kRatTtiOverlayBodySlots;
#else
    constexpr uint32_t kRatTtiRecordedBodySlots = kRatTtiBodySlots;
#endif
    TT_REPLAY_RECORD(kRatTtiRecordedBodySlots);  // record + execute element 0
    if constexpr (kRatTtiLmTail) {
        // RESCHEDULED body, 20 + 3*NSTEPS slots (see block comment above).
        // Identical op DAG / operand roles / mod encodings as the plain body
        // -> bit-identical values; only the schedule and the guard's register
        // assignment (r0 = L4, t = L2, recip = L4) differ.
        TTI_SFPLOAD(ckernel::p_sfpu::LREG2, 0, TT_ADDRMOD_DST_HOLD, 0);  // x
        TTI_SFPMUL(
            ckernel::p_sfpu::LREG2,
            ckernel::p_sfpu::LREG2,
            ckernel::p_sfpu::LCONST_0,
            ckernel::p_sfpu::LREG3,
            0);  // s = x*x
        TTI_SFPLOADI(ckernel::p_sfpu::LREG1, SFPLOADI_MOD0_UPPER, (kDenTopBits >> 16));
        TTI_SFPLOADI(ckernel::p_sfpu::LREG1, SFPLOADI_MOD0_LOWER, (kDenTopBits & 0xFFFFu));  // d2
        if constexpr (kRatTtiUnitDenOrigin) {
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG1,
                ckernel::p_sfpu::LREG3,
                ckernel::p_sfpu::LCONST_1,
                ckernel::p_sfpu::LREG1,
                0);  // den = d2*s + 1.0 (byte-frozen legacy schedule)
        }
        TTI_SFPLOADI(ckernel::p_sfpu::LREG0, SFPLOADI_MOD0_UPPER, (kNumTopBits >> 16));
        TTI_SFPLOADI(ckernel::p_sfpu::LREG0, SFPLOADI_MOD0_LOWER, (kNumTopBits & 0xFFFFu));  // nacc
        rat_tti_num_rung<0>();
        if constexpr (kRatTtiNumSteps >= 2) {
            rat_tti_num_rung<1>();
        }
        if constexpr (kRatTtiNumSteps >= 3) {
            rat_tti_num_rung<2>();
        }
        if constexpr (kRatTtiUsesUnitNumeratorGauge) {
            // Reuse the final n1 LOADI pair for d0. Placing the denominator
            // MAD between the prior and final numerator rungs preserves both
            // chains' hazard gaps without changing the 32-slot n9 body.
            TTI_SFPLOADI(ckernel::p_sfpu::LREG4, SFPLOADI_MOD0_UPPER, (kDenOriginBits >> 16));
            TTI_SFPLOADI(ckernel::p_sfpu::LREG4, SFPLOADI_MOD0_LOWER, (kDenOriginBits & 0xFFFFu));
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG1,
                ckernel::p_sfpu::LREG3,
                ckernel::p_sfpu::LREG4,
                ckernel::p_sfpu::LREG1,
                0);  // den = d2*s + d0
        }
        if constexpr (kRatTtiNumSteps >= 4) {
            rat_tti_num_rung<3>();  // n9 shapes (i1/erfinv) — LM tail only
        }
        TTI_SFPARECIP(0, 1, 4, 0);  // r0 = arecip(den) -> L4 (last-rung gap)
        TTI_SFPMUL(
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG2,
            ckernel::p_sfpu::LCONST_0,
            ckernel::p_sfpu::LREG0,
            0);  // num = nacc*x (odd parity; doubles as the ARECIP window)
        TTI_SFPMAD(
            ckernel::p_sfpu::LREG1,
            ckernel::p_sfpu::LREG4,
            ckernel::p_sfpu::LREG12,
            ckernel::p_sfpu::LREG2,
            2);  // t = den*r0 - 2.0 -> L2 (x is dead: num already formed)
        TTI_SFPNOP;
        TTI_SFPMAD(
            ckernel::p_sfpu::LREG2,
            ckernel::p_sfpu::LREG4,
            ckernel::p_sfpu::LCONST_0,
            ckernel::p_sfpu::LREG3,
            3);  // y1 = r0*-t - 0 (s dead)
        TTI_SFPNOP;
        TTI_SFPSETCC(0, 2, 0, 0);  // CC: t < 0 (MAD result, 3-slot gap)
        TTI_SFPMAD(
            ckernel::p_sfpu::LREG1,
            ckernel::p_sfpu::LREG3,
            ckernel::p_sfpu::LREG12,
            ckernel::p_sfpu::LREG1,
            1);  // t2 = den*y1 - 2.0 (pred; exact GCC mod encoding)
        TTI_SFPNOP;
        TTI_SFPMAD(
            ckernel::p_sfpu::LREG3,
            ckernel::p_sfpu::LREG1,
            ckernel::p_sfpu::LCONST_0,
            ckernel::p_sfpu::LREG4,
            2);  // recip = y1*-t2 - 0 (pred) -> L4 (r0 dead)
        TTI_SFPENCC(3, 0, 0, 10);  // all lanes on — CC-balanced body
        TTI_SFPMUL(
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG4,
            ckernel::p_sfpu::LCONST_0,
            ckernel::p_sfpu::LREG0,
            0);  // y = num * recip -> L0
        // Tail fold: ONE SFPLOADMACRO (macro 0, VD = L4 — dead: recip was
        // consumed by the MUL at issue). Fires T1 = STOCH_RND at issue+1
        // (Round unit; SRCB_OVERRIDE keeps its input the LITERAL L0 = y,
        // exactly 2 cycles after the MUL — the plain body's gap) and the
        // fixed STORE at issue+3 (Store unit, source VD, LATCHED
        // pre-increment dst row). ADDR_MOD_6 rides the LM, so the
        // per-element D-RWC walk (dest += 2, once, last slot) is IDENTICAL
        // to the plain SFPSTORE it replaces. The LM's own load reads the
        // current input row into L4 — dead value, overwritten by T1.
        // In-flight ops land in the next iteration's CC-free LOAD/LOADI
        // head; the body has NO other Round/Store-unit instructions, so
        // there are zero unit collisions under any issue jitter, ~20 slots
        // before the next SFPSETCC (CC discipline). Operand packing per
        // ckernel_sfpu_typecast.h: lreg_ind = (macro_select<<2)|(VD&3),
        // dest_reg_addr LSB = VD>>2.
        TTI_SFPLOADMACRO(
            (0 << 2) | (ckernel::p_sfpu::LREG4 & 3),
            0,
            TT_ADDRMOD_DST_ADV,
            (ckernel::p_sfpu::LREG4 >> 2));
    } else {
        // Wave-6 plain body, 23 + 3*NSTEPS slots — byte-frozen control arm.
        TTI_SFPLOAD(ckernel::p_sfpu::LREG2, 0, TT_ADDRMOD_DST_HOLD, 0);  // x
#if defined(TT_INLINE_PROGRAM_TTI_RATIONAL_SYMMETRIC_SATURATION_OVERLAY)
        // Pre-evaluation route fold.  Mod 9 leaves max(-1,x) in VD; the
        // ordinary vec-min/max mod then leaves min(x,+1) in the same L2.
        // The numerator/denominator/Newton/RNE/store subsequence below is
        // unchanged from the direct mature replay body.
        TTI_SFPSWAP(
            0,
            ckernel::p_sfpu::LCONST_neg1,
            ckernel::p_sfpu::LREG2,
            9);
        TTI_SFPSWAP(
            0,
            ckernel::p_sfpu::LCONST_1,
            ckernel::p_sfpu::LREG2,
            sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
#endif
        TTI_SFPMUL(
            ckernel::p_sfpu::LREG2,
            ckernel::p_sfpu::LREG2,
            ckernel::p_sfpu::LCONST_0,
            ckernel::p_sfpu::LREG3,
            0);  // s = x*x
        TTI_SFPLOADI(ckernel::p_sfpu::LREG0, SFPLOADI_MOD0_UPPER, (kNumTopBits >> 16));
        TTI_SFPLOADI(ckernel::p_sfpu::LREG0, SFPLOADI_MOD0_LOWER, (kNumTopBits & 0xFFFFu));  // nacc = n[NTOP]
        TTI_SFPLOADI(ckernel::p_sfpu::LREG1, SFPLOADI_MOD0_UPPER, (kDenTopBits >> 16));
        TTI_SFPLOADI(ckernel::p_sfpu::LREG1, SFPLOADI_MOD0_LOWER, (kDenTopBits & 0xFFFFu));  // dacc = d2
        rat_tti_num_rung<0>();
        if constexpr (kRatTtiUnitDenOrigin) {
            if constexpr (kRatTtiNumSteps >= 2) {
                rat_tti_num_rung<1>();
            }
            if constexpr (kRatTtiNumSteps >= 3) {
                rat_tti_num_rung<2>();
            }
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG1,
                ckernel::p_sfpu::LREG3,
                ckernel::p_sfpu::LCONST_1,
                ckernel::p_sfpu::LREG1,
                0);  // den = d2*s + 1.0 (byte-frozen legacy schedule)
        } else {
            // For n1==1, run every non-final numerator rung first. The d0
            // LOADI pair then supplies the numerator chain gap formerly
            // supplied by n1's pair; final n1 uses LCONST_1.
            if constexpr (kRatTtiNumSteps >= 3) {
                rat_tti_num_rung<1>();
            }
            TTI_SFPLOADI(ckernel::p_sfpu::LREG4, SFPLOADI_MOD0_UPPER, (kDenOriginBits >> 16));
            TTI_SFPLOADI(ckernel::p_sfpu::LREG4, SFPLOADI_MOD0_LOWER, (kDenOriginBits & 0xFFFFu));
            if constexpr (kRatTtiNumSteps == 2) {
                rat_tti_num_rung<1>();
            } else if constexpr (kRatTtiNumSteps >= 3) {
                rat_tti_num_rung<2>();
            }
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG1,
                ckernel::p_sfpu::LREG3,
                ckernel::p_sfpu::LREG4,
                ckernel::p_sfpu::LREG1,
                0);  // den = d2*s + d0
        }
        TTI_SFPMUL(
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG2,
            ckernel::p_sfpu::LCONST_0,
            ckernel::p_sfpu::LREG0,
            0);  // num = nacc*x (odd parity)
        TTI_SFPARECIP(0, 1, 2, 0);  // r0 = arecip(den) -> L2 (den-gap: the MUL above)
        TTI_SFPNOP;                 // (ARECIP window)
        TTI_SFPMAD(
            ckernel::p_sfpu::LREG1,
            ckernel::p_sfpu::LREG2,
            ckernel::p_sfpu::LREG12,
            ckernel::p_sfpu::LREG4,
            2);  // t = den*r0 - 2.0
        TTI_SFPNOP;
        TTI_SFPMAD(
            ckernel::p_sfpu::LREG4,
            ckernel::p_sfpu::LREG2,
            ckernel::p_sfpu::LCONST_0,
            ckernel::p_sfpu::LREG3,
            3);  // y1 = r0*-t - 0 (s dead)
        TTI_SFPNOP;
        TTI_SFPSETCC(0, 4, 0, 0);  // CC: t < 0 (MAD result, 3-slot gap — rule (b))
        TTI_SFPMAD(
            ckernel::p_sfpu::LREG1,
            ckernel::p_sfpu::LREG3,
            ckernel::p_sfpu::LREG12,
            ckernel::p_sfpu::LREG1,
            1);  // t2 = den*y1 - 2.0 (pred; exact GCC mod encoding)
        TTI_SFPNOP;
        TTI_SFPMAD(
            ckernel::p_sfpu::LREG3,
            ckernel::p_sfpu::LREG1,
            ckernel::p_sfpu::LCONST_0,
            ckernel::p_sfpu::LREG2,
            2);  // y = y1*-t2 - 0 (pred) -> L2
        TTI_SFPENCC(3, 0, 0, 10);  // all lanes on — CC-balanced body
        TTI_SFPMUL(
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG2,
            ckernel::p_sfpu::LCONST_0,
            ckernel::p_sfpu::LREG0,
            0);  // y = num * recip
        TTI_SFPNOP;
        TTI_SFP_STOCH_RND(
            SFPSTOCHRND_RND_EVEN,
            0,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG0,
            SFPSTOCHRND_MOD1_FP32_TO_FP16B);  // fp32 -> bf16 RNE
        TTI_SFPSTORE(ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_ADV, 0);  // store, dest += 2
    }
    TT_REPLAY_REST(kRatTtiRecordedBodySlots, 31);  // elements 1..31
    if constexpr (kRatTtiLmTail) {
        // Drain (audit L5, computed): the final iteration's LM has residuals
        // at issue+1 (Round) and issue+3 (Store) -> required NOPs = 3; five
        // are emitted (2 margin, ~0.15 ns/element). UnitDelayKind =
        // WaitForElapsedInstructions counts SFPNOPs as instructions, so the
        // store retires BEFORE the SETRWC SET_D epilogue re-bases the D-RWC
        // (audit L10) and before tile_regs_commit exposes this half to the
        // packer (audit F4/F5).
        TTI_SFPNOP;
        TTI_SFPNOP;
        TTI_SFPNOP;
        TTI_SFPNOP;
        TTI_SFPNOP;
    }
    TT_REPLAY_TILE_EPILOGUE();
}
#else
#define RATIONAL_TTI_CANDIDATE 0
#endif

// Include specialized implementations using recursive template unrolling
// Replaces old hand-written piecewise_rational_lut_2/4/8/16/generic functions
#include "piecewise_rational_specialized.cpp"

#ifdef FUSE_GRAD_MUL
// Fused unary-backward multiply: y = grad * f'(x). Verbatim twin of
// piecewise_generic.cpp::fuse_grad_mul — see the comment block there for the
// tile-loop-epilogue placement rationale and the byte-identity oracle
// (grad == 1 fused dump must equal the unfused f' dump). This arm was MISSING
// from the rational kernel until 2026-08-18: the host auto-enables fusion for
// every `_bw` op, so the reader streamed cb_grad tiles the compute never
// consumed and the first rational `_bw` config (softsign_bw) deadlocked the
// pipeline. The DST coefficient store above already reserved row base 64 for
// exactly this arm.
inline void fuse_grad_mul() {
#pragma GCC unroll 8
    for (int d = 0; d < 32; d++) {
        vFloat y = dst_reg[d] * dst_reg[32 + d];
#ifdef USE_BF16
        y = convert<vFloat16b>(y, RoundMode::Nearest);
#endif
        dst_reg[d] = y;
    }
}
#endif

}  // namespace sfpi
#endif
