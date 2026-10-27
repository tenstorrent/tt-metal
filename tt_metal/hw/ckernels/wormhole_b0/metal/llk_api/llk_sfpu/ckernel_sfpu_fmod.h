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
    // 2^23 - 1: rounding constant of the residual correction in the calculate function below.
    sfpi::vConstFloatPrgm2 = 8388607.0f;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_fmod() {
    // SFPU microcode
    sfpi::vFloat s = sfpi::abs(sfpi::vConstFloatPrgm0);
    sfpi::vFloat recip_val = sfpi::abs(sfpi::vConstFloatPrgm1);

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat v = sfpi::abs(val);

        vFloat quotient;
        vInt exp = sfpi::exexp(v * recip_val);
        v_if(exp < 0) { quotient = 0.0f; }
        // Since fp32 has 23 mantissa bits, the LSB represents the fractional part when exp < 23.
        // We effectively round off the fractional bits to zero by right shifting using (exp - 23) and then left
        // shifting it back using (0 - (exp - 23)).
        v_elseif(exp < 23) {
            quotient = sfpi::as<sfpi::vFloat>(
                shft((shft(sfpi::as<sfpi::vUInt>(v * recip_val), (exp - 23))), (0 - (exp - 23))));
        }
        v_else { quotient = v * recip_val; }
        v_endif

        v_if(exp < 23 && quotient * s > v) {
            quotient = quotient - 1;
        }
        v_endif;
        v = v - quotient * s;

        v = sfpi::copysgn(v, val);

        v_if(s == 0) { v = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;

        // Residual correction. `quotient` is trunc(fl(v * recip)), which is off by up to ~v/s * 2^-23 units for
        // large quotients, so v may still hold several multiples of s. One more reciprocal step removes them:
        // k = round(v * recip - 1) is floor(v/s) - 1 or floor(v/s) (the 2^23 - 1 constant does the rounding for
        // v * recip < 2^22, and the -1 keeps k from overshooting), so the fused v - k * s lands in [0, 2s) and a
        // single conditional subtract finishes it. A non-finite t (v * recip overflowed, or v is NaN) skips the
        // step, as the subtraction loop it replaces left those lanes alone too.
        sfpi::vFloat t = v * recip_val + sfpi::vConstFloatPrgm2;
        v_if(v >= s && sfpi::exexp(t) < 128) {
            sfpi::vFloat k = t - 8388608.0f;
            v = k * sfpi::setsgn(s, 1) + v;
        }
        v_endif;
        v_if(v >= s) { v = v - s; }
        v_endif;
        v_if(sfpi::abs(v) - s == 0.0f) { v = 0.0f; }
        v_endif;
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
