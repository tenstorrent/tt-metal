// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "ckernel_sfpu_log.h"

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat calculate_sqrt_custom(sfpi::vFloat in) {
    sfpi::vFloat val = in;
    sfpi::vFloat out;
    v_if(val != 0.0f) {
        // Magic number 0x5f37 is used as an approximation constant in the fast inverse square root algorithm.
        // See: https://en.wikipedia.org/wiki/Fast_inverse_square_root
        sfpi::vUInt magic = sfpi::reinterpret<sfpi::vUInt>(sfpi::vFloat(sfpi::s2vFloat16b(0x5f37)));
        sfpi::vFloat approx = sfpi::reinterpret<sfpi::vFloat>(magic - (sfpi::reinterpret<sfpi::vUInt>(val) >> 1));
        sfpi::vFloat neg_half_val = val * -0.5f;
        approx = ((approx * approx) * neg_half_val + 1.5f) * approx;
        approx = ((approx * approx) * neg_half_val + 1.5f) * approx;
        out = approx * val;
    }
    v_else { out = val; }
    v_endif;
    return out;
}

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat calculate_erfinv_body(sfpi::vFloat in) {
    // Algorithm based on "A handy approximation for the error function and its inverse" by Sergei Winitzki (2008)
    // This approximation defines erfinv(x) as:
    // erfinv(x) = sqrt( - 2/(pi*a) - log(1 - x^2)/2 + sqrt( ( 2/(pi*a) + log(1 - x^2)) ^2 - 1/a log(1 - x^2)) )
    // Where a is a polynomial coefficient used in the approximation of the error function (and re-used in inverse error
    // function)

    // Compute log(1 - x^2)
    sfpi::vFloat log_value = in * in;
    log_value = 1 - log_value;
    log_value = calculate_log_body<false, false, true>(log_value, 0);  // use fp32 to avoid intermediate rounding

    sfpi::vFloat temp = log_value * 0.5;

    // Paper sets a constant a = 0.147.
    // This constant is used to compute two constant expressions:
    constexpr float TwoPiA = 4.330746750799873f;   // 2 / (pi * a)
    constexpr float OneDivA = 6.802721088435375f;  // 1/a

    // tmp = -2 / (pi * a) - log(1 - x^2)/2
    temp = TwoPiA + temp;
    temp = -temp;

    // calculated_value = temp + sqrt( temp^2 - log_value / a)
    sfpi::vFloat calculated_value = (temp * temp) - (log_value * OneDivA);
    sfpi::vFloat intermediate_result = calculate_sqrt_custom<false>(calculated_value);
    calculated_value = temp + intermediate_result;

    // result = sqrt(calculated_value)
    sfpi::vFloat result = calculate_sqrt_custom<false>(calculated_value);

    return result;
}

template <bool APPROXIMATION_MODE>
inline void calculate_erfinv() {
    // SFPU microcode
    constexpr int ITERATIONS = 8;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat result;

        // Since erfinv(-x) = -erfinv(x), we can compute the result for the absolute value of the input.
        // This reduces the number of edge cases.
        sfpi::vFloat abs_v = sfpi::abs(v);

        v_if(abs_v == 1.0f) { result = std::numeric_limits<float>::infinity(); }
        v_elseif(abs_v > 1.0f) {  // Nan not supported
            result = std::numeric_limits<float>::quiet_NaN();
        }
        v_else { result = calculate_erfinv_body<true>(abs_v); }
        v_endif;

        result = sfpi::setsgn(result, v);  // restore sign

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void erfinv_init() {
    log_init<false, false, false>();
}

}  // namespace sfpu
}  // namespace ckernel
