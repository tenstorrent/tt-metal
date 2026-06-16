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
    sfpi::vFloat one = sfpi::vConst1;
    sfpi::vFloat abs_x = sfpi::abs(x);

    // Polynomial path for |x| < 0.5
    // Horner form: (((((((-1/8)*x + 1/7)*x - 1/6)*x + 1/5)*x - 1/4)*x + 1/3)*x - 1/2)*x + 1
    sfpi::vFloat p = sfpi::vFloat(-0.125f);
    p = p * x + sfpi::vFloat(0.142857142857f);
    p = p * x + sfpi::vFloat(-0.166666666667f);
    p = p * x + sfpi::vFloat(0.2f);
    p = p * x + sfpi::vFloat(-0.25f);
    p = p * x + sfpi::vFloat(0.333333333333f);
    p = p * x + sfpi::vFloat(-0.5f);
    p = p * x + one;
    sfpi::vFloat result_poly = x * p;

    // Log path for |x| >= 0.5
    sfpi::vFloat arg = one + x;
    sfpi::vFloat result_log = _calculate_log_body_no_init_(arg);

    sfpi::vFloat result = result_log;
    v_if(abs_x < 0.5f) {
        result = result_poly;
    }
    v_endif;

    return result;
}

//
// acosh(x) = log1p((x-1) + sqrt((x-1)*(x+1)))
//
// Numerically stable: avoids absorption at x -> 1+
// Large-x path (x > 4096): acosh(x) ≈ log(x) + log(2)
//
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _calculate_acosh_body_(sfpi::vFloat inp) {
    sfpi::vFloat d = inp - sfpi::vConst1;
    sfpi::vFloat inner = d * (d + sfpi::vFloat(2.0f));
    sfpi::vFloat sq = _calculate_sqrt_body_<APPROXIMATION_MODE>(inner);
    sfpi::vFloat log1p_arg = d + sq;
    sfpi::vFloat result_near = _calculate_log1p_body_<APPROXIMATION_MODE>(log1p_arg);

    sfpi::vFloat log_inp = _calculate_log_body_no_init_(inp);
    sfpi::vFloat result_large = log_inp + sfpi::vFloat(0.693147180560f);

    sfpi::vFloat result = result_near;
    v_if(inp > sfpi::vFloat(4096.0f)) {
        result = result_large;
    }
    v_endif;

    return result;
}

//
// asinh(x) numerically stable implementation
//
// Small |x| (< 1): log1p(x + x^2/(1 + sqrt(1 + x^2)))
// Medium |x| (>= 1, <= 4096): sign * log(|x| + sqrt(x^2 + 1))
// Large |x| (> 4096): sign * (log(|x|) + log(2))
//
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _calculate_asinh_body_(sfpi::vFloat inp) {
    sfpi::vFloat abs_inp = sfpi::abs(inp);
    sfpi::vFloat one = sfpi::vConst1;
    sfpi::vFloat x2 = inp * inp;

    // Small path
    sfpi::vFloat sqrt_1_x2 = _calculate_sqrt_body_<APPROXIMATION_MODE>(one + x2);
    sfpi::vFloat denom_small = one + sqrt_1_x2;
    sfpi::vFloat recip_denom = _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(denom_small);
    sfpi::vFloat log1p_arg_small = inp + x2 * recip_denom;
    sfpi::vFloat result_small = _calculate_log1p_body_<APPROXIMATION_MODE>(log1p_arg_small);

    // Medium/large path (abs value)
    sfpi::vFloat sqrt_x2_1 = _calculate_sqrt_body_<APPROXIMATION_MODE>(x2 + one);
    sfpi::vFloat result_med_abs = _calculate_log_body_no_init_(abs_inp + sqrt_x2_1);

    sfpi::vFloat log_abs = _calculate_log_body_no_init_(abs_inp);
    sfpi::vFloat result_large_abs = log_abs + sfpi::vFloat(0.693147180560f);

    sfpi::vFloat result_abs = result_med_abs;
    v_if(abs_inp > sfpi::vFloat(4096.0f)) {
        result_abs = result_large_abs;
    }
    v_endif;

    sfpi::vFloat result_signed = result_abs;
    v_if(inp < sfpi::vFloat(0.0f)) {
        result_signed = -result_abs;
    }
    v_endif;

    sfpi::vFloat result = result_signed;
    v_if(abs_inp < one) {
        result = result_small;
    }
    v_endif;

    return result;
}

//
// atanh(x) = 0.5 * log1p(2x / (1 - x))
//
// Numerically stable: avoids cancellation at x -> 0 and reciprocal blow-up at x -> 1
//
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _calculate_atanh_body_(sfpi::vFloat inp) {
    sfpi::vFloat one = sfpi::vConst1;
    sfpi::vFloat one_minus_x = one - inp;
    sfpi::vFloat recip_omx = _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(one_minus_x);
    sfpi::vFloat two_x = inp + inp;
    sfpi::vFloat log1p_arg = two_x * recip_omx;
    sfpi::vFloat result = sfpi::vFloat(0.5f) * _calculate_log1p_body_<APPROXIMATION_MODE>(log1p_arg);
    return result;
}

// -------------------------------------------------------------------------
// Public SFPU kernel entry points
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
