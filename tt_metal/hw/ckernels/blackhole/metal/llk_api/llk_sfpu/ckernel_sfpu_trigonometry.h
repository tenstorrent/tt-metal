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
// Uses the identity: u = x/(x+2), log1p(x) = 2*u*P(u^2)
// where P(t) = 1 + t/3 + t^2/5 + t^3/7 + ...
// This is the standard Cephes/glibc approach and avoids all cancellation.
//
// For |x| >= 1.0, falls back to log(1+x) which is safe since 1+x >= 0
// and no significant cancellation occurs.
//
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _calculate_log1p_body_(sfpi::vFloat x) {
    sfpi::vFloat one_plus_x = x + sfpi::vConst1;

    // u = x / (x + 2)
    sfpi::vFloat denom = x + 2.0f;
    sfpi::vFloat u = x * _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(denom);
    sfpi::vFloat u2 = u * u;

    sfpi::vFloat series;
    if constexpr (APPROXIMATION_MODE) {
        // 4-term series
        series = u2 * (1.0f / 7.0f);
        series = series + (1.0f / 5.0f);
        series = series * u2;
        series = series + (1.0f / 3.0f);
        series = series * u2;
        series = series + 1.0f;
    } else {
        // 6-term series
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

    sfpi::vFloat log1p_large = _calculate_log_body_no_init_(one_plus_x);

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
// Numerically stable for all |x| < 1. Avoids catastrophic cancellation
// that occurs in the naive 0.5*log((1+x)/(1-x)) form near x=0.
//
template <bool APPROXIMATION_MODE>
sfpi_inline void _calculate_atanh_(sfpi::vFloat inp) {
    sfpi::vFloat two_x = inp * 2.0f;
    sfpi::vFloat one_minus_x = sfpi::vConst1 - inp;

    sfpi::vFloat t = two_x * _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(one_minus_x);

    sfpi::vFloat log1p_t = _calculate_log1p_body_<APPROXIMATION_MODE>(t);

    sfpi::vFloat res = 0.5f * log1p_t;

    v_if(inp == 0.0f) {
        res = 0.0f;
    }
    v_endif;

    sfpi::dst_reg[0] = res;
}

//
// asinh(x) = sign(x) * log1p(|x| + x^2 / (1 + sqrt(x^2 + 1)))
//
// This is the Kahan-stable formula. For x -> 0, reduces to log1p(0) = 0.
// For large x, uses log(2) + log(|x|) to avoid x^2 overflow.
//
template <bool APPROXIMATION_MODE>
sfpi_inline void _calculate_asinh_(sfpi::vFloat inp) {
    sfpi::vFloat abs_x = sfpi::abs(inp);
    sfpi::vFloat x2 = inp * inp;

    constexpr float LARGE_THRESH = 1.844e19f;
    constexpr float LOG2 = 0.6931471805599453f;

    sfpi::vFloat x2_plus_1 = x2 + sfpi::vConst1;
    sfpi::vFloat sqrt_x2p1 = _calculate_sqrt_body_<APPROXIMATION_MODE>(x2_plus_1);
    sfpi::vFloat denom = sqrt_x2p1 + sfpi::vConst1;
    sfpi::vFloat correction = x2 * _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(denom);
    sfpi::vFloat arg = abs_x + correction;
    sfpi::vFloat result_normal = _calculate_log1p_body_<APPROXIMATION_MODE>(arg);

    sfpi::vFloat result_large = _calculate_log_body_no_init_(abs_x) + LOG2;

    sfpi::vFloat result = result_normal;
    v_if(abs_x >= LARGE_THRESH) {
        result = result_large;
    }
    v_endif;

    v_if(inp < 0.0f) {
        result = -result;
    }
    v_endif;

    v_if(inp == 0.0f) {
        result = 0.0f;
    }
    v_endif;

    sfpi::dst_reg[0] = result;
}

//
// acosh(x) = log1p(d + sqrt(d*(d+2)))   where d = x - 1
//
// This avoids x^2 overflow for x near 1 and prevents the absorption error
// log(x + tiny) ~ log(x) that the naive form suffers near x=1^+.
// For large x, uses log(2) + log(x).
//
template <bool APPROXIMATION_MODE>
sfpi_inline void _calculate_acosh_(sfpi::vFloat inp) {
    constexpr float LARGE_THRESH = 1.844e19f;
    constexpr float LOG2 = 0.6931471805599453f;

    sfpi::vFloat d = inp - sfpi::vConst1;

    sfpi::vFloat d_plus_2 = d + 2.0f;
    sfpi::vFloat radicand = d * d_plus_2;
    sfpi::vFloat sqrt_rad = _calculate_sqrt_body_<APPROXIMATION_MODE>(radicand);
    sfpi::vFloat arg = d + sqrt_rad;
    sfpi::vFloat result_normal = _calculate_log1p_body_<APPROXIMATION_MODE>(arg);

    sfpi::vFloat result_large = _calculate_log_body_no_init_(inp) + LOG2;

    sfpi::vFloat result = result_normal;
    v_if(inp >= LARGE_THRESH) {
        result = result_large;
    }
    v_endif;

    v_if(inp == sfpi::vConst1) {
        result = 0.0f;
    }
    v_endif;

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
