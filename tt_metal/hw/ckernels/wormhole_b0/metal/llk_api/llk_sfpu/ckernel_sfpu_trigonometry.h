// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

//
// log1p(x) = log(1 + x), computed accurately for all x > -1.
//
// Strategy:
//   - For |x| < 0.5: use the identity
//       log1p(x) = x - x^2/2 + x^3/3 - ...
//     but more efficiently via:
//       log1p(x) = log(1 + x)  using the SFPU log after argument reduction.
//     We use: if |x| <= 0.5, compute as log((1+x)), but rewrite as:
//       u = x / (2 + x),  log1p(x) = 2 * atanh(u)  -- this avoids cancellation.
//     Actually the cleanest SFPU-friendly approach: split on magnitude.
//
//   For |x| < 0.5:
//     u = x / (x + 2)          (exact, no cancellation since x+2 > 1.5)
//     log1p(x) = 2 * u * (1 + u^2/3 + u^4/5 + u^6/7 + u^8/9 + u^10/11)
//     This is the standard Cephes / glibc log1p polynomial in u = x/(x+2).
//
//   For x >= 0.5 (or x <= -0.5):
//     log1p(x) = log(1 + x)  via the normal log path (no cancellation risk
//     since 1+x >= 0.5 keeps mantissa bits intact when x is not tiny).
//
// NOTE: caller must ensure x > -1.
//
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _calculate_log1p_body_(sfpi::vFloat x) {
    // We use the identity: for u = x/(x+2),
    //   log1p(x) = 2*u*(1 + u^2/3 + u^4/5 + u^6/7 + u^8/9 + u^10/11)
    // which is accurate for all x > -1 without cancellation.
    //
    // For large x (|x| > threshold), fall back to normal log(1+x) since
    // 1+x is representable accurately and log won't lose bits.

    // Threshold: if x > 1.0 or x < -0.5, use log(1+x) directly
    // (when x is large, x/(x+2) -> 1 and the series converges slowly;
    //  also when 1+x >= 0.5 there's no catastrophic cancellation in log)
    //
    // For the SFPU we'll use a hybrid: always compute both and select.

    sfpi::vFloat one_plus_x = x + sfpi::vConst1;

    // For |x| < threshold, use the u-substitution series
    // u = x / (x + 2)
    sfpi::vFloat denom = x + 2.0f;
    sfpi::vFloat u = x * _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(denom);
    sfpi::vFloat u2 = u * u;

    sfpi::vFloat series;
    if constexpr (APPROXIMATION_MODE) {
        // 4-term series: 1 + u^2/3 + u^4/5 + u^6/7
        series = u2 * (1.0f / 7.0f);
        series = series + (1.0f / 5.0f);
        series = series * u2;
        series = series + (1.0f / 3.0f);
        series = series * u2;
        series = series + 1.0f;
    } else {
        // 6-term series: 1 + u^2/3 + u^4/5 + u^6/7 + u^8/9 + u^10/11
        series = u2 * (1.0f / 11.0f);
        series = series + (1.0f / 9.0f);
        series = series * u2;
        series = series + (1.0f / 7.0f);
        series = series * u2;
        series = series + (1.0f / 5.0f);
        series = series * u2;
        series = series + (1.0f / 3.0f);
        series = series * u2;
        series = series + 1.0f;
    }
    sfpi::vFloat log1p_small = 2.0f * u * series;

    // For large |x|, use standard log
    sfpi::vFloat log1p_large = _calculate_log_body_no_init_(one_plus_x);

    // Select: use series result when |x| < 1.0, otherwise use log(1+x)
    // (when x >= 1, 1+x >= 2 so no precision loss in log; series less efficient)
    sfpi::vFloat result = log1p_small;
    v_if(sfpi::abs(x) >= 1.0f) {
        result = log1p_large;
    }
    v_endif;

    return result;
}

//
// atanh(x) = 0.5 * log1p(2x / (1 - x))
//
// This avoids catastrophic cancellation near x=0 (the naive (1+x)/(1-x) form
// suffers from log(1 + small) when x is tiny).
//
// Edge cases:
//   x = +1 -> +inf
//   x = -1 -> -inf
//   |x| > 1 -> NaN (domain error, return NaN)
//   x = 0  -> 0
//
template <bool APPROXIMATION_MODE>
sfpi_inline void _calculate_atanh_(sfpi::vFloat inp) {
    // atanh(x) = 0.5 * log1p(2x / (1 - x))
    // Argument to log1p: t = 2x / (1 - x)
    // When x -> 0: t -> 2x, log1p(2x) -> 2x, result -> x.  Accurate.
    // When x -> 1: t -> +inf, log1p(inf) = inf, result -> +inf.  Correct.
    // When x -> -1: t -> -inf... actually t = 2x/(1-x), x=-1 -> t = -2/2 = -1,
    //   log1p(-1) = -inf. Correct.

    sfpi::vFloat two_x = inp * 2.0f;
    sfpi::vFloat one_minus_x = sfpi::vConst1 - inp;

    // Compute t = 2x / (1 - x)
    sfpi::vFloat t = two_x * _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(one_minus_x);

    // log1p(t)
    sfpi::vFloat log1p_t = _calculate_log1p_body_<APPROXIMATION_MODE>(t);

    sfpi::vFloat res = 0.5f * log1p_t;

    // Handle x = 0 exactly
    v_if(inp == 0.0f) {
        res = 0.0f;
    }
    v_endif;

    sfpi::dst_reg[0] = res;
}

//
// asinh(x) = sign(x) * log1p(|x| + (x^2 / (1 + sqrt(x^2 + 1))))
//
// This is the Kahan-stable form:
//   asinh(x) = log1p(x + x^2 / (1 + sqrt(x^2 + 1)))   for x >= 0
//
// Derivation: multiply log(x + sqrt(x^2+1)) numerator/denominator by
//   (sqrt(x^2+1) - x) / (sqrt(x^2+1) - x):
//   = log((x^2+1 - x^2) / (sqrt(x^2+1) - x)) ... no, let's use:
//
//   asinh(x) = log(x + sqrt(x^2+1))
//            = log1p(x + sqrt(x^2+1) - 1)
//            = log1p(x + (x^2+1-1)/(sqrt(x^2+1)+1))    [rationalise]
//            = log1p(x + x^2 / (sqrt(x^2+1) + 1))
//
// This form has no catastrophic cancellation for any x.
//
// For large |x| (|x| > sqrt(FLT_MAX/2) ~ 1.3e19): avoid x^2 overflow.
//   Use: asinh(x) = sign(x) * (log(2) + log(|x|))  for very large x
//   More precisely: asinh(x) = sign(x) * log(2|x|) * (1 + 1/(4x^2) + ...)
//   We use: sign(x) * (log(|x|) + log(2))  when |x| > 2^63 (basically).
//   Practically for fp32, overflow happens at |x| > ~1.84e19.
//   FLT_MAX ~ 3.4e38, sqrt(FLT_MAX) ~ 1.84e19.
//
template <bool APPROXIMATION_MODE>
sfpi_inline void _calculate_asinh_(sfpi::vFloat inp) {
    sfpi::vFloat abs_x = sfpi::abs(inp);
    sfpi::vFloat x2 = inp * inp;

    // Large-x threshold: sqrt(FLT_MAX) ~ 1.844e19
    // We use 1e19 as a safe threshold in fp32
    constexpr float LARGE_THRESH = 1.844e19f;
    constexpr float LOG2 = 0.6931471805599453f;

    // Normal path: log1p(|x| + x^2 / (1 + sqrt(x^2 + 1)))
    sfpi::vFloat x2_plus_1 = x2 + sfpi::vConst1;
    sfpi::vFloat sqrt_x2p1 = _calculate_sqrt_body_<APPROXIMATION_MODE>(x2_plus_1);
    sfpi::vFloat denom = sqrt_x2p1 + sfpi::vConst1;
    sfpi::vFloat correction = x2 * _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(denom);
    sfpi::vFloat arg = abs_x + correction;
    sfpi::vFloat result_normal = _calculate_log1p_body_<APPROXIMATION_MODE>(arg);

    // Large-x path: log(2) + log(|x|)
    sfpi::vFloat result_large = _calculate_log_body_no_init_(abs_x) + LOG2;

    // Select based on magnitude
    sfpi::vFloat result = result_normal;
    v_if(abs_x >= LARGE_THRESH) {
        result = result_large;
    }
    v_endif;

    // Restore sign
    v_if(inp < 0.0f) {
        result = -result;
    }
    v_endif;

    // Handle x = 0 exactly
    v_if(inp == 0.0f) {
        result = 0.0f;
    }
    v_endif;

    sfpi::dst_reg[0] = result;
}

//
// acosh(x) = log1p((x - 1) + sqrt((x-1)*(x+1)))
//          = log1p((x-1) + sqrt(x^2 - 1))
//          = log1p((x-1) * (1 + sqrt((x+1)/(x-1))))   [alternative form]
//
// The second form avoids cancellation near x=1 where sqrt(x^2-1) -> 0:
//   acosh(x) = log1p(d + sqrt(d * (d + 2)))   where d = x - 1
//
// When d -> 0: d*(d+2) -> 2d, sqrt(2d) -> sqrt(2d), log1p(d + sqrt(2d))
//   For very small d: ~ log1p(sqrt(2d)) ~ sqrt(2d)  (matches Taylor: acosh(1+eps) ~ sqrt(2eps))
//   No cancellation since we only add positive quantities.
//
// For large x (x > sqrt(FLT_MAX) ~ 1.84e19): x^2 overflows.
//   Use: acosh(x) = log(2x) = log(2) + log(x)
//   More precisely: acosh(x) = log(x + sqrt(x^2-1)) ~ log(2x) for large x.
//
// Domain: x >= 1. For x < 1, return NaN.
//
template <bool APPROXIMATION_MODE>
sfpi_inline void _calculate_acosh_(sfpi::vFloat inp) {
    constexpr float LARGE_THRESH = 1.844e19f;
    constexpr float LOG2 = 0.6931471805599453f;

    // d = x - 1  (>= 0 in valid domain)
    sfpi::vFloat d = inp - sfpi::vConst1;

    // Normal path: log1p(d + sqrt(d * (d + 2)))
    // d*(d+2) = d^2 + 2d = x^2 - 1  (algebraically)
    sfpi::vFloat d_plus_2 = d + 2.0f;
    sfpi::vFloat radicand = d * d_plus_2;  // = (x-1)(x+1) = x^2-1, no overflow near x=1
    sfpi::vFloat sqrt_rad = _calculate_sqrt_body_<APPROXIMATION_MODE>(radicand);
    sfpi::vFloat arg = d + sqrt_rad;
    sfpi::vFloat result_normal = _calculate_log1p_body_<APPROXIMATION_MODE>(arg);

    // Large-x path: log(2) + log(x)
    sfpi::vFloat result_large = _calculate_log_body_no_init_(inp) + LOG2;

    sfpi::vFloat result = result_normal;
    v_if(inp >= LARGE_THRESH) {
        result = result_large;
    }
    v_endif;

    // x = 1 -> acosh(1) = 0 exactly
    v_if(inp == sfpi::vConst1) {
        result = 0.0f;
    }
    v_endif;

    // x < 1: domain error -> NaN
    // (hardware will naturally produce NaN from sqrt of negative, but let's be explicit)
    // We leave the NaN from sqrt(negative radicand) to propagate naturally for x < 1.

    sfpi::dst_reg[0] = result;
}

template <bool APPROXIMATION_MODE>
inline void calculate_atanh(uint iterations) {
    _init_sfpu_log_();
    for (int d = 0; d < iterations; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        _calculate_atanh_<APPROXIMATION_MODE>(val);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void calculate_asinh(uint iterations) {
    _init_sfpu_log_();
    for (int d = 0; d < iterations; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        _calculate_asinh_<APPROXIMATION_MODE>(val);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void calculate_acosh(uint iterations) {
    _init_sfpu_log_();
    for (int d = 0; d < iterations; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        _calculate_acosh_<APPROXIMATION_MODE>(val);
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
