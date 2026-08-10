// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cmath_common.h"  // math::reset_counters, p_setrwc
#include "ckernel_sfpu_sigmoid.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_silu() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // silu(x) = x * sigmoid(x)
        // _sfpu_sigmoid_'s parameter selects the accurate exp plus a 2-iteration reciprocal; false
        // selects exp_21f (~1 ULP on bfloat16) plus a single iteration. Take the accurate path only
        // when DST is fp32 and the caller has not asked for approximate math.
        constexpr bool accurate_sigmoid = is_fp32_dest_acc_en && !APPROXIMATION_MODE;
        sfpi::vFloat result = x * _sfpu_sigmoid_<accurate_sigmoid>(x);

        // Round to bfloat16 if not in fp32 accumulation mode
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void silu_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    // Both of calculate_silu's paths go through _sfpu_sigmoid_, never the approximate sigmoid LUT, and
    // they differ only in the reciprocal iteration count. sigmoid_init<false> sets up that reciprocal,
    // so it is the correct init for either value of APPROXIMATION_MODE.
    sigmoid_init<false>();
}

}  // namespace ckernel::sfpu
