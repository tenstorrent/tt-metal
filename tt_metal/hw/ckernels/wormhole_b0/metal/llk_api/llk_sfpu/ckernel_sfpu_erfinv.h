// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "ckernel_sfpu_log.h"
#include "ckernel_sfpu_sqrt_custom.h"

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

sfpi_inline sfpi::vFloat calculate_erfinv_series(sfpi::vFloat x) {
    // Maclaurin series for erfinv(x) = (sqrt(pi)/2) * (x + pi/12 x^3 + 7 pi^2/480 x^5 + ...),
    // evaluated as a Horner polynomial in t = x^2. Used near x = 0, where the Winitzki
    // approximation below cancels two nearly-equal float32 values and collapses to 0
    // for |x| below ~3.9e-4 (see issue #51660).
    constexpr float SqrtPiOver2 = 0.8862269254527579f;
    constexpr float A1 = 0.2617993877991494f;   // pi/12
    constexpr float A2 = 0.14393173084921979f;  // 7*pi^2/480
    constexpr float A3 = 0.09766361950392055f;  // 127*pi^3/40320
    constexpr float A4 = 0.07329907936638086f;  // 4369*pi^4/5806080
    constexpr float A5 = 0.05837250087858452f;  // 34807*pi^5/182476800

    sfpi::vFloat t = x * x;
    sfpi::vFloat series = A5;
    series = series * t + A4;
    series = series * t + A3;
    series = series * t + A2;
    series = series * t + A1;
    series = series * t + 1.0f;
    return SqrtPiOver2 * x * series;
}

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat calculate_erfinv_body(sfpi::vFloat x) {
    // Algorithm based on "A handy approximation for the error function and its inverse" by Sergei Winitzki (2008)
    // This approximation defines erfinv(x) as:
    // erfinv(x) = sqrt( - 2/(pi*a) - log(1 - x^2)/2 + sqrt( ( 2/(pi*a) + log(1 - x^2)) ^2 - 1/a log(1 - x^2)) )
    // Where a is a polynomial coefficient used in the approximation of the error function (and reused in inverse error
    // function)

    // Compute log(1 - x^2)
    sfpi::vFloat log_value = calculate_log_body<false, false, false>(1.0f - x * x, 0);

    // Paper sets a constant a = 0.147.
    // This constant is used to compute two constant expressions:
    constexpr float TwoPiA = -4.330746750799873f;  // -2 / (pi * a)
    constexpr float OneDivA = 6.802721088435375f;  // 1/a

    // tmp = -2 / (pi * a) - log(1 - x^2)/2
    sfpi::vFloat tmp = TwoPiA + -0.5f * log_value;

    // calculated_value = temp + sqrt( temp^2 - log_value / a)
    sfpi::vFloat calculated_value = tmp * tmp - log_value * OneDivA;
    sfpi::vFloat intermediate_result = sfpu_sqrt_custom<false, 2>(calculated_value);
    calculated_value = tmp + intermediate_result;

    // result = sqrt(calculated_value)
    sfpi::vFloat result = sfpu_sqrt_custom<false, 2>(calculated_value);

    // The Winitzki form above subtracts two nearly-equal float32 values as x -> 0 and
    // rounds to exactly 0 below |x| ~ 3.9e-4 (see issue #51660). Below that threshold,
    // erfinv is well-conditioned, so use the Maclaurin series instead. The threshold is
    // insensitive anywhere in [0.05, 0.7]; 0.1 keeps a comfortable margin above the collapse
    // point while staying inside the series' high-accuracy region.
    sfpi::vFloat abs_x = sfpi::abs(x);
    v_if(abs_x < 0.1f) { result = calculate_erfinv_series(x); }
    v_endif;

    return result;
}

template <bool APPROXIMATION_MODE>
inline void calculate_erfinv() {
    constexpr int ITERATIONS = 8;
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
    math::reset_counters(p_setrwc::SET_ABD_F);
    log_init<false, false, false>();
}

}  // namespace sfpu
}  // namespace ckernel
