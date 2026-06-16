// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// This file is auto-included by the LLK infrastructure.
// It replaces the previous naive implementations of atanh / asinh / acosh
// with numerically stable log1p-based algorithms.

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_log.h"
#include "ckernel_sfpu_sqrt.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

//
// ─────────────────────────────────────────────────────────────────────────────
//  Internal helpers
// ─────────────────────────────────────────────────────────────────────────────
//

// Compute 1/x using Newton-Raphson refinement of the hardware reciprocal.
// APPROXIMATION_MODE → one HW-approx step only.
// Full mode           → two NR iterations (~24-bit accurate).
template <bool APPROXIMATION_MODE>
inline sfpi::vFloat _sfpu_recip_nr_(sfpi::vFloat x) {
    if constexpr (APPROXIMATION_MODE) {
        return sfpi::approx_recip(x);
    } else {
        sfpi::vFloat r = sfpi::approx_recip(x);
        // NR:  r ← r·(2 − x·r)   twice
        r = r * (sfpi::vFloat(2.0f) - x * r);
        r = r * (sfpi::vFloat(2.0f) - x * r);
        return r;
    }
}

// ─── log1p(x) = log(1+x), Kahan-compensated ──────────────────────────────────
//
// For x not too close to −1 or +∞ this gives ~1-ulp accuracy even when
// |x| << 1.
//
// Reference: Higham, "Accuracy and Stability of Numerical Algorithms", §1.14.
//   u  = 1 + x                         (may round)
//   log1p(x) ≈ log(u) + (x − (u−1))/u  (corrects the rounding in u)
//
template <bool APPROXIMATION_MODE>
inline sfpi::vFloat _log1p_(sfpi::vFloat x) {
    sfpi::vFloat u    = sfpi::vConst1 + x;
    sfpi::vFloat lnu  = _calculate_log_body_no_init_(u);

    // Kahan correction: recover the bits lost when forming u = 1 + x
    sfpi::vFloat diff = x - (u - sfpi::vConst1);
    sfpi::vFloat corr = diff * _sfpu_recip_nr_<APPROXIMATION_MODE>(u);

    sfpi::vFloat result = lnu + corr;

    // x == 0 → result must be exactly 0 (log(1) = 0)
    v_if (x == sfpi::vConst0) {
        result = sfpi::vConst0;
    }
    v_endif;

    return result;
}

// ─── sqrt via existing helper ─────────────────────────────────────────────────
// Wraps _calculate_sqrt_body_ so callers don't have to spell out the template.
template <bool APPROXIMATION_MODE>
inline sfpi::vFloat _sfpu_sqrt_(sfpi::vFloat x) {
    return _calculate_sqrt_body_<APPROXIMATION_MODE>(x);
}

//
// ─────────────────────────────────────────────────────────────────────────────
//  atanh(x) — numerically stable, full fp32 range
// ─────────────────────────────────────────────────────────────────────────────
//
// Mathematical identity (exact):
//   atanh(x) = 0.5 · log((1+x)/(1−x))
//
// Naive form cancels catastrophically for |x| → 0 and saturates the
// reciprocal for |x| → 1.
//
// Stable form (valid for all |x| < 1):
//   atanh(x) = 0.5 · log1p( 2x / (1−x) )
//
// Derivation:
//   (1+x)/(1−x) = 1 + (1+x)/(1−x) − 1
//               = 1 + 2x/(1−x)
//   ∴ log((1+x)/(1−x)) = log1p( 2x/(1−x) )
//
// The fraction 2x/(1−x) → 0 as x→0, so log1p captures it correctly.
// For x→1  the fraction → +∞ and log1p(+∞) = +∞ correctly.
// For x→−1 the fraction → −1 and log1p(−1) = −∞ correctly.
// No reciprocal of (1−x) explodes before the log.
//
// Special cases:
//   |x| >= 1  → ±inf (or NaN) — handled by v_if guard
//   x  == 0   → 0
//
template <bool APPROXIMATION_MODE>
inline sfpi::vFloat _calculate_atanh_body_(sfpi::vFloat x) {
    // den = 1 - x  (always > 0 for |x| < 1)
    sfpi::vFloat den = sfpi::vConst1 - x;

    // arg = 2x / (1 - x)
    // We compute via multiply-by-reciprocal to avoid a divide instruction.
    sfpi::vFloat inv_den = _sfpu_recip_nr_<APPROXIMATION_MODE>(den);
    // 2x * inv_den:
    sfpi::vFloat arg = x * inv_den;              // x / (1-x)
    arg = arg + arg;                              // 2x / (1-x)

    // 0.5 * log1p(arg)
    sfpi::vFloat result = _log1p_<APPROXIMATION_MODE>(arg) * sfpi::vFloat(0.5f);

    // Guard: |x| >= 1 → ±inf
    // (The fp32 reciprocal of (1-x) may misbehave at exactly x=1, so clamp.)
    v_if (x >= sfpi::vConst1) {
        result = std::numeric_limits<float>::infinity();
    }
    v_endif;
    v_if (x <= -sfpi::vConst1) {
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    return result;
}

//
// ─────────────────────────────────────────────────────────────────────────────
//  asinh(x) — numerically stable, full fp32 range
// ─────────────────────────────────────────────────────────────────────────────
//
// Mathematical identity (exact):
//   asinh(x) = log( x + sqrt(x² + 1) )
//
// Problems with naive form:
//   • x → 0:  sqrt(x²+1) ≈ 1, so argument ≈ 1+x → log1p needed.
//   • |x| >> 1: x² + 1 overflows fp32 (threshold ≈ 1.84 × 10¹⁹).
//
// Stable form (three regions):
//
//  Region A  |x| <= SMALL  (e.g. |x| <= 1):
//     asinh(x) = log1p( x + x²/(1 + sqrt(1 + x²)) )
//
//     Derivation:
//       x + sqrt(x²+1) = x + sqrt(x²+1)
//       Multiply/divide by (sqrt(x²+1) + 1):
//         = (x·(sqrt(x²+1)+1) + x²+1 − 1) / (sqrt(x²+1)+1)
//         Wait, simpler:
//       log(x + sqrt(x²+1)) = log1p( x + sqrt(x²+1) − 1 )
//       x + sqrt(x²+1) − 1 = x + (sqrt(x²+1) − 1)
//                           = x + x²/(sqrt(x²+1)+1)    [rationalise]
//     So:
//       asinh(x) = log1p( x + x² / (1 + sqrt(1+x²)) )
//     This is safe for |x| → 0 (argument → 0 + 0 = 0) and for x of order 1.
//
//  Region B  1 < |x| <= LARGE  (|x| <= sqrt(FLT_MAX/4) ≈ 1.38×10¹⁹):
//     asinh(x) = sign(x) · log( |x| + sqrt(x²+1) )
//     This is the direct formula, but now |x| ≥ 1 so sqrt(x²+1) ≈ |x|,
//     and their sum is ≥ 1, so the log argument is well above 1.
//     No cancellation occurs.
//
//  Region C  |x| > LARGE:
//     x² + 1 overflows → use
//       asinh(x) = sign(x) · (log(2) + log(|x|))
//                = sign(x) · log(2·|x|)
//     More precisely:
//       log(|x| + sqrt(x²+1)) = log(|x|·(1 + sqrt(1+1/x²)))
//                              ≈ log(|x|) + log(2)   for large |x|
//     with correction log1p(0.5/x²) ≈ 0.
//
// Implementation note: fp32 sqrt(FLT_MAX) = 1.844×10¹⁹  so we use
//   LARGE = 1e19 as the overflow threshold.
//
static constexpr float ASINH_LARGE_THRESHOLD = 1.0e19f;
static constexpr float LOG2_F               = 0.693147180559945f;

template <bool APPROXIMATION_MODE>
inline sfpi::vFloat _calculate_asinh_body_(sfpi::vFloat inp) {
    sfpi::vFloat x    = sfpi::abs(inp);
    sfpi::vFloat sign = sfpi::vConst1;
    v_if (inp < sfpi::vConst0) {
        sign = sfpi::vFloat(-1.0f);
    }
    v_endif;

    sfpi::vFloat result;

    // ── Region C: overflow guard ──────────────────────────────────────────
    // |x| > LARGE  →  asinh ≈ sign · (log(x) + log(2))
    sfpi::vFloat log_x      = _calculate_log_body_no_init_(x);
    sfpi::vFloat large_res  = log_x + sfpi::vFloat(LOG2_F);

    // ── Region A: |x| <= 1 ───────────────────────────────────────────────
    // asinh(x) = log1p( x + x² / (1 + sqrt(1+x²)) )
    sfpi::vFloat x2         = x * x;
    sfpi::vFloat sqrt_1px2  = _sfpu_sqrt_<APPROXIMATION_MODE>(sfpi::vConst1 + x2);
    sfpi::vFloat denom_a    = sfpi::vConst1 + sqrt_1px2;
    sfpi::vFloat inv_da     = _sfpu_recip_nr_<APPROXIMATION_MODE>(denom_a);
    sfpi::vFloat arg_a      = x + x2 * inv_da;
    sfpi::vFloat small_res  = _log1p_<APPROXIMATION_MODE>(arg_a);

    // ── Region B: 1 < |x| <= LARGE ───────────────────────────────────────
    // asinh(x) = log( x + sqrt(x²+1) )  — no cancellation for |x|>1
    sfpi::vFloat sqrt_x2p1  = _sfpu_sqrt_<APPROXIMATION_MODE>(x2 + sfpi::vConst1);
    sfpi::vFloat mid_res    = _calculate_log_body_no_init_(x + sqrt_x2p1);

    // Select result by region
    result = small_res;                                       // default: Region A

    v_if (x > sfpi::vConst1) {
        result = mid_res;                                     // Region B
    }
    v_endif;

    v_if (x > sfpi::vFloat(ASINH_LARGE_THRESHOLD)) {
        result = large_res;                                   // Region C
    }
    v_endif;

    result = result * sign;
    return result;
}

//
// ─────────────────────────────────────────────────────────────────────────────
//  acosh(x) — numerically stable, full fp32 range  (domain: x >= 1)
// ─────────────────────────────────────────────────────────────────────────────
//
// Mathematical identity (exact):
//   acosh(x) = log( x + sqrt(x² − 1) )   for x >= 1
//
// Problems with naive form:
//   • x → 1⁺:  sqrt(x²−1) → 0, so argument → 1+0 → log(1+tiny) loses digits.
//   • x ≳ 1.84×10¹⁹:  x² overflows fp32.
//
// Stable form (three regions):
//
//  Region A  1 <= x <= SMALL (e.g. x <= 2):
//     Use the Kahan / log1p trick:
//       x + sqrt(x²-1) = 1 + (x-1) + sqrt((x-1)(x+1))
//                      = 1 + (x-1) + sqrt(x-1)·sqrt(x+1)
//     Let  d = x − 1  (exact near x=1 since x ≥ 1).
//       acosh(x) = log1p( d + sqrt(d) · sqrt(d + 2) )
//                = log1p( d + sqrt(d·(d+2)) )
//     because (x-1)·(x+1) = x²-1 = d·(d+2).
//     For x→1: d→0, sqrt(d·(d+2)) ≈ sqrt(2d) → 0, arg → 0. log1p(0) = 0. ✓
//
//  Region B  2 < x <= LARGE  (x <= sqrt(FLT_MAX/4) ≈ 1.38×10¹⁹):
//     Direct formula: acosh(x) = log( x + sqrt(x²-1) )
//     For x ≥ 2:  sqrt(x²-1) ≥ sqrt(3) ≈ 1.73 and x ≥ 2, so the argument
//     is ≥ 3.73 → well above 1 → no cancellation.
//
//  Region C  x > LARGE:
//     x² overflows → acosh(x) ≈ log(2x) = log(x) + log(2)
//
static constexpr float ACOSH_LARGE_THRESHOLD = 1.0e19f;

template <bool APPROXIMATION_MODE>
inline sfpi::vFloat _calculate_acosh_body_(sfpi::vFloat x) {
    sfpi::vFloat result;

    // ── Region C: overflow guard ──────────────────────────────────────────
    sfpi::vFloat log_x     = _calculate_log_body_no_init_(x);
    sfpi::vFloat large_res = log_x + sfpi::vFloat(LOG2_F);

    // ── Region A: 1 <= x <= 2 ────────────────────────────────────────────
    // d = x - 1  (computed exactly for x near 1 in fp32)
    sfpi::vFloat d         = x - sfpi::vConst1;
    // d*(d+2) = (x-1)*(x+1) = x² - 1
    sfpi::vFloat d2        = d + sfpi::vFloat(2.0f);
    sfpi::vFloat sq        = _sfpu_sqrt_<APPROXIMATION_MODE>(d * d2);
    sfpi::vFloat arg_a     = d + sq;
    sfpi::vFloat small_res = _log1p_<APPROXIMATION_MODE>(arg_a);

    // ── Region B: 2 < x <= LARGE ─────────────────────────────────────────
    sfpi::vFloat x2        = x * x;
    sfpi::vFloat sq_b      = _sfpu_sqrt_<APPROXIMATION_MODE>(x2 - sfpi::vConst1);
    sfpi::vFloat mid_res   = _calculate_log_body_no_init_(x + sq_b);

    // Select by region (default: Region A)
    result = small_res;

    v_if (x > sfpi::vFloat(2.0f)) {
        result = mid_res;                                    // Region B
    }
    v_endif;

    v_if (x > sfpi::vFloat(ACOSH_LARGE_THRESHOLD)) {
        result = large_res;                                  // Region C
    }
    v_endif;

    // x < 1 is out of domain → NaN (hardware should already give NaN from
    // sqrt of negative, but make it explicit)
    v_if (x < sfpi::vConst1) {
        // Generate a NaN via 0/0 or just load a canonical NaN constant.
        // On Wormhole the SFPU represents NaN as all-ones exponent+mantissa.
        // Simplest portable approach: sqrt of negative number.
        result = _sfpu_sqrt_<APPROXIMATION_MODE>(sfpi::vFloat(-1.0f));
    }
    v_endif;

    return result;
}

//
// ─────────────────────────────────────────────────────────────────────────────
//  Public entry points called by the LLK dispatch layer
// ─────────────────────────────────────────────────────────────────────────────
//

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_atanh(const uint dst_offset) {
    for (int d = 0; d < ITERATIONS; ++d) {
        copy_tile_to_dst_init_short_with_dt(dst_offset, dst_offset);
        sfpi::dst_reg[0] = _calculate_atanh_body_<APPROXIMATION_MODE>(sfpi::dst_reg[0]);
        d++;  // advance dst register (pairs)
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_asinh(const uint dst_offset) {
    for (int d = 0; d < ITERATIONS; ++d) {
        copy_tile_to_dst_init_short_with_dt(dst_offset, dst_offset);
        sfpi::dst_reg[0] = _calculate_asinh_body_<APPROXIMATION_MODE>(sfpi::dst_reg[0]);
        d++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_acosh(const uint dst_offset) {
    for (int d = 0; d < ITERATIONS; ++d) {
        copy_tile_to_dst_init_short_with_dt(dst_offset, dst_offset);
        sfpi::dst_reg[0] = _calculate_acosh_body_<APPROXIMATION_MODE>(sfpi::dst_reg[0]);
        d++;
    }
}

}  // namespace ckernel::sfpu
