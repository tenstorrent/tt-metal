// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "ckernel_sfpu_log.h"

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
inline void calculate_erfinv() {
    constexpr float TwoPiA = -4.330746750799873f;
    constexpr float OneDivA = 6.802721088435375f;
    constexpr float SqrtPiDiv2 = 0.886226925452758f;
    constexpr int ITERATIONS = 8;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat x2 = x * x;
        sfpi::vFloat one_minus_x2 = 1.0f - x2;
        sfpi::vFloat rounding_error = (1.0f - one_minus_x2) - x2;
        sfpi::vFloat log_value = calculate_log_body<true, false, false>(one_minus_x2, 0) + rounding_error;
        sfpi::vFloat tmp = TwoPiA + -0.5f * log_value;
        sfpi::vFloat neg_log_div_a = log_value * -OneDivA;
        sfpi::vFloat discriminant = tmp * tmp + neg_log_div_a;
        sfpi::vUInt magic = sfpi::as<sfpi::vUInt>(sfpi::vFloat(sfpi::sFloat16b(0x5f37)));
        sfpi::vFloat y = sfpi::as<sfpi::vFloat>(magic - (sfpi::as<sfpi::vUInt>(discriminant) >> 1));
        sfpi::vFloat neg_half_discriminant = discriminant * -0.5f;
        sfpi::vFloat t = y * y;
        t = t * neg_half_discriminant + 1.5f;
        sfpi::vFloat in = sfpi::dst_reg[0];
        y = t * y;
        t = y * y;
        t = t * neg_half_discriminant + 1.5f;
        sfpi::vFloat result = in * SqrtPiDiv2 + neg_log_div_a;
        y = t * y;
        sfpi::vFloat p = discriminant * y - tmp;
        sfpi::vUInt magic_p = sfpi::as<sfpi::vUInt>(sfpi::vFloat(sfpi::sFloat16b(0x5f37)));
        p = p * neg_log_div_a;
        v_if(p != 0.0f && sfpi::is_finite(p)) {
            sfpi::vFloat z = sfpi::as<sfpi::vFloat>(magic_p - (sfpi::as<sfpi::vUInt>(p) >> 1));
            sfpi::vFloat neg_half_p = p * -0.5f;
            z = ((z * z) * neg_half_p + 1.5f) * z;
            z = ((z * z) * neg_half_p + 1.5f) * z;
            result = neg_log_div_a * z;
        }
        v_endif;
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
