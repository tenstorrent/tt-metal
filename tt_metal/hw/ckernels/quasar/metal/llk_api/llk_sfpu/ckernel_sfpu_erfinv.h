// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * The log(x) code is derived from code by Norbert Juffa.
 *
 * Copyright (c) 2015-2023, Norbert Juffa
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <limits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "ckernel_sfpu_sqrt_custom.h"

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// BF16 log polynomial from Blackhole; constants are installed by erfinv_init.
sfpi_inline sfpi::vFloat _erfinv_log_body_(sfpi::vFloat a) {
    sfpi::vFloat three_quarters = 0.75f;
    sfpi::vInt e = sfpi::as<sfpi::vInt>(a) - sfpi::as<sfpi::vInt>(three_quarters);
    a = a * 1.0f + 0.0f;
    e = sfpi::as<sfpi::vInt>(sfpi::setman(sfpi::as<sfpi::vFloat>(e), 0));
    sfpi::vFloat m = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(a) - e);
    sfpi::vFloat result = std::numeric_limits<float>::quiet_NaN();
    m -= 1.0f;
    v_if(a >= 0.0f) {
        sfpi::vFloat s = m * m;
        sfpi::vMag abs_e = sfpi::abs(e);
        sfpi::vFloat r = -0.25f * m + sfpi::vConstFloatPrgm1;
        sfpi::vFloat e_float = sfpi::convert<sfpi::vFloat>(abs_e, sfpi::RoundMode::Nearest);
        r = r * m + sfpi::vConstFloatPrgm2;
        a = sfpi::addexp(a, -1);
        r = r * s + m;
        e_float = sfpi::copysgn(e_float, sfpi::as<sfpi::vFloat>(e));
        result = e_float * sfpi::vConstFloatPrgm0 + r;
        v_if(sfpi::exexp(a, sfpi::ExponentMode::Biased) - 255 >= 0) { result *= a; }
        v_endif;
    }
    v_endif;
    return result;
}

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat calculate_erfinv_body(sfpi::vFloat x) {
    // Algorithm based on "A handy approximation for the error function and its inverse" by Sergei Winitzki (2008)
    // This approximation defines erfinv(x) as:
    // erfinv(x) = sqrt( - 2/(pi*a) - log(1 - x^2)/2 + sqrt( ( 2/(pi*a) + log(1 - x^2)) ^2 - 1/a log(1 - x^2)) )
    // Where a is a polynomial coefficient used in the approximation of the error function (and reused in inverse error
    // function)

    // Compute log(1 - x^2)
    sfpi::vFloat log_value = _erfinv_log_body_(1.0f - x * x);

    // Paper sets a constant a = 0.147.
    // This constant is used to compute two constant expressions:
    constexpr float TwoPiA = -4.330746750799873f;   // -2 / (pi * a)
    constexpr float OneDivA = 6.802721088435375f;  // 1/a

    // tmp = -2 / (pi * a) - log(1 - x^2)/2
    sfpi::vFloat tmp = TwoPiA + -0.5f * log_value;

    // calculated_value = temp + sqrt( temp^2 - log_value / a)
    sfpi::vFloat calculated_value = tmp * tmp - log_value * OneDivA;
    sfpi::vFloat intermediate_result = sfpu_sqrt_custom<false>(calculated_value);
    calculated_value = tmp + intermediate_result;

    // result = sqrt(calculated_value)
    sfpi::vFloat result = sfpu_sqrt_custom<false>(calculated_value);

    return result;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_erfinv() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat result = calculate_erfinv_body<false>(in);
        in = sfpi::dst_reg[0];  // reload due to register pressure
        sfpi::dst_reg[0] = sfpi::copysgn(result, in);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void erfinv_init() {
    math::_reset_counters_<p_setrwc::SET_ABD_F>();
    sfpi::vConstFloatPrgm0 = 0.693147182f * 1.19209290e-7f;
    sfpi::vConstFloatPrgm1 = 0x1.744p-2f;
    sfpi::vConstFloatPrgm2 = -0x1.008p-1f;
}

}  // namespace sfpu
}  // namespace ckernel
