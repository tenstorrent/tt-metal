// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cmath_common.h"  // math::reset_counters, p_setrwc
#include "ckernel_sfpu_sigmoid.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_silu() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // silu(x) = x * sigmoid(x)
        sfpi::vFloat result = x * _sfpu_sigmoid_<is_fp32_dest_acc_en>(x);
        // Below -80, sigmoid(x) equals exp(x) to fp32 precision, but 1 / (1 + exp(-x)) flushes to zero near -87.3
        // and silu(x) came out exactly 0 (#55711). x * exp(x) is still normal down to about -103; evaluate it as
        // (x * exp(x/2)) * exp(x/2) so no intermediate leaves the normal range.
        v_if(x < -80.0f) {
            sfpi::vFloat half_exp;
            if constexpr (is_fp32_dest_acc_en) {
                half_exp = _sfpu_exp_accurate_<true>(0.5f * x);
            } else {
                half_exp = _sfpu_exp_21f_bf16_<true>(0.5f * x);
            }
            result = (x * half_exp) * half_exp;
        }
        v_endif;

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
    // calculate_silu uses the non-approx sigmoid path via _sfpu_sigmoid_, so we must use non-approx sigmoid_init
    sigmoid_init<false>();
}

}  // namespace ckernel::sfpu
