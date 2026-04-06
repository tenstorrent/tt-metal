// SPDX-FileCopyrightText: (c) 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// Compute exp(val) for POSITIVE val using Horner polynomial + repeated squaring.
// The polynomial approximates exp(x) for x in [0.5, 1.0).
// For val >= 1.0, range-reduce by clamping exponent to -1 then squaring back up.
// For val in (0, 1), the polynomial is applied directly.
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _selu_exp_positive_(sfpi::vFloat val) {
    sfpi::vInt exp = sfpi::exexp(val);
    v_if(exp >= 0) {
        val = sfpi::setexp(val, 126);  // map to [0.5, 1.0)
    }
    v_endif;

    // Horner polynomial: exp(x) ~ 0.8373*x^2 + 0.863281*x + 1.0
    sfpi::vFloat tmp = val * sfpi::vConst0p8373 + sfpi::s2vFloat16b(0.863281f);
    val = val * tmp + sfpi::vConst1;

    // Repeated squaring: exp(x) = exp(x/2^n)^(2^n)
    v_if(exp >= 0) {
        val = val * val;
        for (int s_iter = 0; s_iter < 7; s_iter++) {
            exp = exp - 1;
            v_and(exp >= 0);
            val = val * val;
        }
    }
    v_endif;

    return val;
}

// Newton-Raphson reciprocal: compute 1/in with 2 iterations (float32 precision).
// Adapted from ckernel_sfpu_recip.h. Uses programmable constants vConstFloatPrgm0/1/2
// loaded in selu_init() for the quadratic initial estimate.
template <int max_iter = 2>
sfpi_inline sfpi::vFloat _selu_reciprocal_(const sfpi::vFloat in) {
    // Scale input to [1.0, 2.0) and negate
    sfpi::vFloat negative_x = sfpi::setman(sfpi::vConstNeg1, sfpi::reinterpret<sfpi::vInt>(in));

    // Quadratic initial estimate: y = k2 + k1*(-x) + k0*(-x)^2
    sfpi::vFloat y = sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0 * negative_x;

    // Scale factor: 2^(255-in.Exp) via bitwise NOT
    sfpi::vInt scale_bits = ~sfpi::reinterpret<sfpi::vUInt>(in);

    y = sfpi::vConstFloatPrgm2 + y * negative_x;

    sfpi::vFloat scale = sfpi::setman(sfpi::reinterpret<sfpi::vFloat>(scale_bits), 0);

    // First Newton-Raphson iteration
    sfpi::vFloat t = sfpi::vConst1 + negative_x * y;

    // Adjust scale: 255-in.Exp -> 254-in.Exp
    scale *= 0.5f;

    y = y + y * t;

    if constexpr (max_iter > 1) {
        // Second Newton-Raphson iteration for float32 precision
        t = sfpi::vConst1 + negative_x * y;
        y = y + y * t;
    }

    // Apply scaling and restore sign
    y = y * scale;
    y = sfpi::setsgn(y, in);

    return y;
}

// SELU: selu(x) = scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
// Simplifies to:
//   x >= 0: selu(x) = scale * x
//   x <  0: selu(x) = scale * alpha * (exp(x) - 1)
//
// Constants:
//   scale = 1.0507009873554804934193349852946
//   alpha = 1.6732632423543772848170429916717
//   scale * alpha = 1.7580993408473768599402175208123
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_selu() {
    constexpr float scale = 1.0507009873554804934193349852946f;
    constexpr float scale_alpha = 1.7580993408473768599402175208123f;

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];

        v_if(v >= 0.0f) {
            // Positive branch: selu(x) = scale * x
            v = v * scale;
        }
        v_else {
            // Negative branch: selu(x) = scale_alpha * (exp(x) - 1)
            // Compute exp(x) for x < 0 as 1/exp(|x|)
            sfpi::vFloat abs_v = sfpi::setsgn(v, 0);  // |v|
            sfpi::vFloat exp_abs = _selu_exp_positive_<APPROXIMATION_MODE>(abs_v);
            sfpi::vFloat exp_v = _selu_reciprocal_<2>(exp_abs);  // exp(v) = 1/exp(|v|)
            v = (exp_v - 1.0f) * scale_alpha;
        }
        v_endif;

        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void selu_init() {
    // Load Sollya-optimized polynomial coefficients for reciprocal initial estimate
    // These approximate 1/x over [1, 2) using y = k2 + k1*(-x) + k0*(-x)^2
    sfpi::vConstFloatPrgm0 = 0.3232325017452239990234375f;  // k0
    sfpi::vConstFloatPrgm1 = 1.4545459747314453125f;        // k1
    sfpi::vConstFloatPrgm2 = 2.121212482452392578125f;      // k2
}

}  // namespace sfpu
}  // namespace ckernel
