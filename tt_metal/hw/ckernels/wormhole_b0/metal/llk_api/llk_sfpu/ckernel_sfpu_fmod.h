// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "ckernel_sfpu_converter.h"
#include "ckernel_sfpu_recip.h"

using namespace sfpi;
namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
inline void _init_fmod_(const uint value, const uint recip) {
    // load vConstFloatPrgm0 = value
    _sfpu_load_config32_(0xC, (value >> 16) & 0xFFFF, value & 0xFFFF);
    // load vConstFloatPrgm1 = recip
    _sfpu_load_config32_(0xD, (recip >> 16) & 0xFFFF, recip & 0xFFFF);
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_fmod(const uint value, const uint recip) {
    // SFPU microcode
    vFloat s = vConstFloatPrgm0;
    vFloat recip_val = vConstFloatPrgm1;
    s = sfpi::abs(s);
    recip_val = sfpi::abs(recip_val);

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat val = dst_reg[0];
        vFloat v = sfpi::abs(val);

        vFloat quotient = v * recip_val;

        vFloat newquotient;
        vInt exp = exexp(quotient);
        v_if(exp < 0) { quotient = vConst0; }
        v_elseif(exp < 23) {
            newquotient = reinterpret<vFloat>(shft((shft(reinterpret<vUInt>(quotient), (exp - 23))), (0 - (exp - 23))));
        }
        v_else { newquotient = quotient; }
        v_endif

        v_if(newquotient > quotient) {
            newquotient = newquotient - 1;
        }
        v_endif;

        v = v - newquotient * s;
        v = setsgn(v, val);

        v_if(s == 0) { v = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;

        constexpr auto iter = 10;
        for (int l = 0; l < iter; l++) {
            v_if(v >= s) { v = s - v; }
            v_endif;
        }
        v_if(sfpi::abs(v) - s == 0.0f) { v = 0.0f; }
        v_endif;
        dst_reg[0] = v;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
