// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

template <bool APPROXIMATION_MODE>
sfpi_inline vFloat _sfpu_sine_maclaurin_series_(vFloat val) {
    // Good for [-pi:pi]
    // Mclauren series = x - x^3/3! + x^5/5! - x^7/7! + x^9/9! - x^11/11!
    vFloat tmp = val;
    // x
    vFloat output = tmp;
    // x^3/3!
    tmp = tmp * val * val;
    output += -0.166666666 * tmp;
    // x^5/5!
    tmp = tmp * val * val;
    output += 0.0083333333 * tmp;
    // x^7/7!
    tmp = tmp * val * val;
    output += -0.0001984126 * tmp;
    if constexpr (not APPROXIMATION_MODE) {
        // x^9/9!
        tmp = tmp * val * val;
        output += 0.0000027557 * tmp;
        // x^11/11!
        tmp = tmp * val * val;
        output += -0.00000002505 * tmp;
    }

    // Write out output
    return output;
}

template <bool APPROXIMATION_MODE>
sfpi_inline vFloat _sfpu_cosine_maclaurin_series_(vFloat val) {
    // Good for [-pi:pi]
    // Mclauren series = 1 - x^2/2! + x^4/4! - x^6/6! + x^8/8! - x^10/10! + x^12/12!
    // 1
    vFloat output = 1.0f;
    // x^2/2!
    vFloat tmp = val * val;
    output += -0.5 * tmp;
    // x^4/4!
    tmp = tmp * val * val;
    output += 0.0416666666 * tmp;
    // x^6/6!
    tmp = tmp * val * val;
    output += -0.0013888888 * tmp;
    if constexpr (not APPROXIMATION_MODE) {
        // x^8/8!
        tmp = tmp * val * val;
        output += 0.0000248015 * tmp;
        // x^10/10!
        tmp = tmp * val * val;
        output += -0.0000002755 * tmp;
    }

    // Write out output
    return output;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_sine_() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];

        // Assume v is bound [0:2pi]
        // phase shift [0:2pi] to [-pi:pi] and multiply result by -1
        v = v - 3.14159264f;
        v = _sfpu_sine_maclaurin_series_<APPROXIMATION_MODE>(v);

        // Use symmetrical properties of trig
        v *= -1;

        // Write Output
        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_cosine_() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];

        // Assume v is bound [0:2pi]
        // phase shift [0:2pi] to [-pi:pi] and multiply result by -1
        v = v - 3.14159264f;
        v = _sfpu_cosine_maclaurin_series_<APPROXIMATION_MODE>(v);

        // Use symmetrical properties of trig
        v *= -1;

        // Write Output
        dst_reg[0] = v;
        dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
