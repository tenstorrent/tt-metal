// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_converter.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

namespace cosh_internal {

// Inline exponential using polynomial approximation after range reduction.
// Computes exp(x) using: exp(x) = poly(r) * 2^k, where k = round(x/ln2), r = x - k*ln2.
// Polynomial is degree-2 Horner evaluation matching the coefficients from SDPA compute_common.hpp.
template <bool APPROXIMATION_MODE>
sfpi::vFloat inline_exp(sfpi::vFloat val) {
    using namespace sfpi;

    constexpr float LN2_RECIP = 1.44269504088896340736f;  // 1/ln(2)
    constexpr float NEG_LN2 = -0.69314718055994530942f;   // -ln(2)

    // Degree 2 polynomial coefficients for exp(r) where |r| <= ln(2)/2
    constexpr float c0 = 0.999848792924395313327307061545061386175496934006f;
    constexpr float c1 = 1.01508760098521056684783640695492761469306929535975f;
    constexpr float c2 = 0.50628367056745568861842335616023694454759126020461f;

    // k = round(x / ln(2)), clamped to int8 range [-128, 127]
    vFloat scaled = val * LN2_RECIP;
    vUInt k_uint = float_to_int8(scaled);
    vInt k_int = reinterpret<vInt>(k_uint);
    vFloat k_float = int32_to_float(k_int, 0);

    // r = x - k * ln(2) -- fractional part in [-ln2/2, ln2/2]
    vFloat r = val + k_float * NEG_LN2;

    // Evaluate degree-2 polynomial: exp(r) ~ (c2*r + c1)*r + c0
    vFloat poly = r * c2 + c1;
    poly = poly * r + c0;

    // Reconstruct exp(x) = poly * 2^k by constructing 2^k via exponent manipulation.
    // IEEE754: biased_exponent = k + 127. Set exponent field of 1.0f to (k + 127).
    vInt biased_exp = k_int + vInt(127);
    vFloat two_to_k = setexp(vFloat(1.0f), reinterpret<vUInt>(biased_exp));
    vFloat result = poly * two_to_k;

    // Handle underflow: for very negative x, exp(x) -> 0
    v_if(val < -80.0f) { result = 0.0f; }
    v_endif;

    return result;
}

}  // namespace cosh_internal

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_cosh() {
    using namespace sfpi;

    // cosh(x) = (exp(x) + exp(-x)) / 2
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x = dst_reg[0];

        // Compute exp(x) and exp(-x) using inline polynomial exp
        vFloat exp_pos = cosh_internal::inline_exp<APPROXIMATION_MODE>(x);
        vFloat exp_neg = cosh_internal::inline_exp<APPROXIMATION_MODE>(-x);

        // cosh = (exp(x) + exp(-x)) / 2
        vFloat result = (exp_pos + exp_neg) * 0.5f;

        dst_reg[0] = result;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
