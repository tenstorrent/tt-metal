// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
inline void init_fmod(const uint value, const uint recip) {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstFloatPrgm0 = Converter::as_float(value);
    sfpi::vConstFloatPrgm1 = Converter::as_float(recip);
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_fmod() {
    // SFPU microcode
    sfpi::vFloat s = sfpi::abs(sfpi::vConstFloatPrgm0);
    sfpi::vFloat recip_val = sfpi::abs(sfpi::vConstFloatPrgm1);
    // Keep one fractional bit in the quotient estimate before correcting the residual.
    constexpr int residual_correction_exp_limit = 23;

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat v = sfpi::abs(val);

        sfpi::vFloat quotient = v * recip_val;
        sfpi::vInt exp = sfpi::exexp(quotient);
        v_if(exp < 0) { quotient = 0.0f; }
        // Since fp32 has 23 mantissa bits, the LSB represents the fractional part when exp < 23.
        // We effectively round off the fractional bits to zero by right shifting using (exp - 23) and then left
        // shifting it back using (0 - (exp - 23)).
        v_elseif(exp < 23) {
            quotient =
                sfpi::as<sfpi::vFloat>(shft((shft(sfpi::as<sfpi::vUInt>(quotient), (exp - 23))), (0 - (exp - 23))));
        }
        v_endif;

        v_if(exp < 23 && quotient * s > v) {
            quotient = quotient - 1;
        }
        v_endif;
        v = v - quotient * s;
        v_if(exp < residual_correction_exp_limit) {
            // The floored quotient times s can exceed abs(val), leaving a negative residual.
            v_if(v < 0) { v = v + s; }
            v_endif;

            quotient = quotient + 1;
            quotient = sfpi::abs(val) - quotient * s;
            v_if(quotient >= 0) {
                // Accept exact zero for either sign; preserve the positive tail's nonzero residual.
                v_if(quotient == 0) { v = quotient; }
                v_elseif(val < 0) { v = quotient; }
                v_endif;
            }
            v_endif;
        }
        v_endif;

        v = sfpi::copysgn(v, val);

        v_if(s == 0) { v = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;

        constexpr auto iter = 10;
        for (int l = 0; l < iter; l++) {
            v_if(v >= s) { v = v - s; }
            v_endif;
        }
        v_if(sfpi::abs(v) - s == 0.0f) { v = sfpi::copysgn(sfpi::vFloat(0.0f), val); }
        v_endif;
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
