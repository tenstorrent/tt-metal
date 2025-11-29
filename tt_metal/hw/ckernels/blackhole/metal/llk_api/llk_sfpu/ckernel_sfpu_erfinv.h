// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
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
        for (int r = 0; r < 2; r++) {
            approx = ((approx * approx) * (val * -0.5f) + 1.5f) * approx;
        }
        out = approx * val;
    }
    v_else { out = val; }
    v_endif;
    return out;
}

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat calculate_erfinv_body(sfpi::vFloat in) {
    sfpi::vFloat log_value = in * in;
    log_value = 1 - log_value;
    log_value = calculate_log_body<false, false, true>(log_value, 0);  // use fp32 to avoid intermediate rounding
    sfpi::vFloat temp = log_value * 0.5;
    temp = 4.5469 + temp;
    temp = -temp;

    sfpi::vFloat calculated_value = (temp * temp) - (log_value * 7.1427);
    sfpi::vFloat intermediate_result = calculate_sqrt_custom<false>(calculated_value);
    calculated_value = temp + intermediate_result;
    log_value = calculate_sqrt_custom<false>(calculated_value);
    return log_value;
}

template <bool APPROXIMATION_MODE>
inline void calculate_erfinv() {
    // SFPU microcode
    constexpr int ITERATIONS = 8;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat result;
        v_if(v == 1.0f) { result = std::numeric_limits<float>::infinity(); }
        v_elseif(v == -1.0f) { result = -std::numeric_limits<float>::infinity(); }
        v_elseif((v < -1.0f) || (v > 1.0f)) {  // Nan not supported
            result = std::numeric_limits<float>::quiet_NaN();
        }
        v_elseif(v < 0.0f) {
            result = calculate_erfinv_body<true>(v);
            result = -result;
        }
        v_else { result = calculate_erfinv_body<true>(v); }
        v_endif;

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
