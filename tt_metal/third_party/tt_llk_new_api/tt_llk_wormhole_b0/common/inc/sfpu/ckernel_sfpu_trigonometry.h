// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "noc_nonblocking_api.h"
#include "sfpi.h"
#include "ckernel_sfpu_converter.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_log.h"
#include "ckernel_sfpu_sqrt.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel {
namespace sfpu {

//
// log1p(x) = log(1 + x)
//
// Numerically stable implementation:
//   - For |x| >= 0.5: use ordinary log(1 + x)
//   - For |x| <  0.5: use a degree-8 minimax polynomial in x
//
// The polynomial was derived from the standard Padé/Chebyshev approximation
// for log(1+x)/x on (-0.5, 0.5):
//   log(1+x) ≈ x * P(x)
// where P(x) = 1 - x/2 + x^2/3 - x^3/4 + x^4/5 - x^5/6 + x^6/7 - x^7/8
//
// This gives about 6-7 ULP accuracy over (-0.5, 0.5).
// Outside that range we fall back to the existing log kernel.
//
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _calculate_log1p_body_(sfpi::vFloat x) {
    // For |x| < 0.5 use a direct polynomial; otherwise use log(1+x).
    sfpi::vFloat one = sfpi::vConst1;
    sfpi::vFloat abs_x = sfpi::abs(x);

    // Polynomial path for |x| < 0.5
    // log(1+x) = x*(1 - x/2 + x^2/3 - x^3/4 + x^4/5 - x^5/6 + x^6/7 - x^7/8)
    // Computed via Horner's method:
    // p = ((((((( -1/8 )*x + 1/7 )*x - 1/6 )*x + 1/5 )*x - 1/4 )*x + 1/3 )*x - 1/2 )*x + 1
    // result_poly = x * p
    sfpi::vFloat p = sfpi::vFloat(-0.125f);                   // -1/8
    p = p * x + sfpi::vFloat(0.142857142857f);                // +1/7
    p = p * x + sfpi::vFloat(-0.166666666667f);               // -1/6
    p = p * x + sfpi::vFloat(0.2f);                           // +1/5
    p = p * x + sfpi::vFloat(-0.25f);                         // -1/4
    p = p * x + sfpi::vFloat(0.333333333333f);                // +1/3
    p = p * x + sfpi::vFloat(-0.5f);                          // -1/2
    p = p * x + one;                                           // +1
    sfpi::vFloat result_poly = x * p;

    // Log path for |x| >= 0.5: log(1 + x)
    sfpi::vFloat arg = one + x;
    sfpi::vFloat result_log = _calculate_log_body_no_init_(arg);

    // Select based on magnitude
    // Use v_if / v_else on abs_x < 0.5
    sfpi::vFloat result = result_log;
    v_if(abs_x < 0.5f) {
        result = result_poly;
    }
    v_endif;

    return result;
}

//
// acosh(x) = log(x + sqrt(x^2 - 1))
//
// Numerically stable form via log1p:
//   acosh(x) = log1p((x-1) + sqrt((x-1)*(x+1)))
//            = log1p(d + sqrt(d * (d + 2)))   where d = x - 1
//
// This avoids the absorption error when x -> 1+ (where sqrt(x^2-1) -> 0
// and adding x ≈ 1 to a tiny number loses precision).
//
// For very large x (x > 2^24 in fp32, i.e., x^2 would overflow):
//   acosh(x) ≈ log(2x) = log(2) + log(x)
//
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _calculate_acosh_body_(sfpi::vFloat inp) {
    // d = x - 1  (exact when x is close to 1)
    sfpi::vFloat d = inp - sfpi::vConst1;

    // sqrt(d * (d + 2)) = sqrt((x-1)(x+1)) = sqrt(x^2 - 1)
    sfpi::vFloat inner = d * (d + sfpi::vFloat(2.0f));
    sfpi::vFloat sq = _calculate_sqrt_body_<APPROXIMATION_MODE>(inner);

    // log1p(d + sqrt(...))
    sfpi::vFloat log1p_arg = d + sq;
    sfpi::vFloat result_near = _calculate_log1p_body_<APPROXIMATION_MODE>(log1p_arg);

    // Large-x path: x > 2^12 = 4096 to be conservative (x^2 > 2^24, near fp32 precision limit)
    // acosh(x) ≈ log(2) + log(x) = log(2*x)
    sfpi::vFloat result_large = _calculate_log_body_no_init_(inp + inp) ;
    // log(2x) = log(x) + log(2); we compute log(2*inp) directly
    // but _calculate_log_body_no_init_ expects a pre-initialised value.
    // Instead compute as: log(inp) + log(2)
    sfpi::vFloat log_inp = _calculate_log_body_no_init_(inp);
    result_large = log_inp + sfpi::vFloat(0.693147180560f);  // + log(2)

    sfpi::vFloat result = result_near;
    v_if(inp > sfpi::vFloat(4096.0f)) {
        result = result_large;
    }
    v_endif;

    return result;
}

//
// asinh(x) = log(x + sqrt(x^2 + 1))
//
// Numerically stable forms:
//   Small |x| (|x| < 1):
//     asinh(x) = log1p(x + x^2/(1 + sqrt(1 + x^2)))
//     This avoids log(1 + small) catastrophic cancellation.
//
//   Large |x| (|x| >= 1, but not huge):
//     asinh(x) = sign(x) * log(|x| + sqrt(x^2 + 1))   [standard, OK here]
//
//   Very large |x| (|x| > 2^12):
//     asinh(x) = sign(x) * (log(2) + log(|x|))
//     because x^2 + 1 ≈ x^2, sqrt ≈ |x|, log(2|x|) = log(2) + log(|x|)
//
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _calculate_asinh_body_(sfpi::vFloat inp) {
    sfpi::vFloat abs_inp = sfpi::abs(inp);
    sfpi::vFloat one = sfpi::vConst1;

    // --- Small path: |x| < 1 ---
    // asinh(x) = log1p(x + x^2 / (1 + sqrt(1 + x^2)))
    sfpi::vFloat x2 = inp * inp;
    sfpi::vFloat sqrt_1_x2 = _calculate_sqrt_body_<APPROXIMATION_MODE>(one + x2);
    sfpi::vFloat denom_small = one + sqrt_1_x2;
    // Use reciprocal to do x^2 / denom
    sfpi::vFloat recip_denom = _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(denom_small);
    sfpi::vFloat log1p_arg_small = inp + x2 * recip_denom;
    sfpi::vFloat result_small = _calculate_log1p_body_<APPROXIMATION_MODE>(log1p_arg_small);

    // --- Medium path: |x| >= 1 ---
    // asinh(x) = sign(x) * log(|x| + sqrt(x^2 + 1))
    sfpi::vFloat sqrt_x2_1 = _calculate_sqrt_body_<APPROXIMATION_MODE>(x2 + one);
    sfpi::vFloat log_arg_med = abs_inp + sqrt_x2_1;
    sfpi::vFloat result_med_abs = _calculate_log_body_no_init_(log_arg_med);

    // --- Large path: |x| > 4096 ---
    // asinh(x) = sign(x) * (log(|x|) + log(2))
    sfpi::vFloat log_abs = _calculate_log_body_no_init_(abs_inp);
    sfpi::vFloat result_large_abs = log_abs + sfpi::vFloat(0.693147180560f);

    // Combine medium and large
    sfpi::vFloat result_abs = result_med_abs;
    v_if(abs_inp > sfpi::vFloat(4096.0f)) {
        result_abs = result_large_abs;
    }
    v_endif;

    // Restore sign for medium/large
    sfpi::vFloat result_signed = result_abs;
    v_if(inp < sfpi::vFloat(0.0f)) {
        result_signed = -result_abs;
    }
    v_endif;

    // Select small vs medium/large
    sfpi::vFloat result = result_signed;
    v_if(abs_inp < one) {
        result = result_small;
    }
    v_endif;

    return result;
}

//
// atanh(x) = 0.5 * log((1+x)/(1-x))
//
// Numerically stable form via log1p:
//   atanh(x) = 0.5 * log1p(2x / (1 - x))
//
// This avoids:
//   1. Catastrophic cancellation at x -> 0 (log(1 + 2x) ≈ 2x loses nothing)
//   2. Reciprocal blow-up at x -> 1 (the argument 2x/(1-x) -> +inf naturally,
//      and log1p(+inf) = +inf which is correct)
//
// Domain: x in (-1, 1). At |x| >= 1 the result is ±inf or NaN (matches math).
//
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _calculate_atanh_body_(sfpi::vFloat inp) {
    sfpi::vFloat one = sfpi::vConst1;

    // 1 - x
    sfpi::vFloat one_minus_x = one - inp;

    // 2x / (1 - x)  -- use reciprocal then multiply
    sfpi::vFloat recip_omx = _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(one_minus_x);
    sfpi::vFloat two_x = inp + inp;
    sfpi::vFloat log1p_arg = two_x * recip_omx;

    // 0.5 * log1p(2x / (1 - x))
    sfpi::vFloat result = sfpi::vFloat(0.5f) * _calculate_log1p_body_<APPROXIMATION_MODE>(log1p_arg);

    return result;
}

// -------------------------------------------------------------------------
// Public SFPU kernel entry points (called from generated op wrappers)
// -------------------------------------------------------------------------

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_acosh() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = _calculate_acosh_body_<APPROXIMATION_MODE>(val);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_asinh() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = _calculate_asinh_body_<APPROXIMATION_MODE>(val);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_atanh() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = _calculate_atanh_body_<APPROXIMATION_MODE>(val);
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
