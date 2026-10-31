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

        // silu(x) = x * sigmoid(x). For x <= -88.5f (including -inf), exp(-x)
        // overflows so sigmoid(x) underflows to 0 and the product would be
        // (-inf) * 0.0f = NaN.
        //
        // Clamping the input is enough, because the NaN needs an *infinite*
        // x: with x finite, x * 0.0f is -0.0f, not NaN. For x = -88.5f,
        // sigmoid is ~3.7e-39 and the product is ~-3.3e-37, which rounds to
        // zero in every destination format — and if exp underflows all the
        // way to 0 the product is -0.0f, which is equally correct.
        //
        // This is branchless on purpose. A v_if/v_else here measured ~190
        // cycles/tile slower on Wormhole and ~130 on Blackhole, and it also
        // pushes the fp32 arm out of LRegs (the "too few lregs to hold live
        // values" build failure). sfpi::max is the same guard other kernels
        // already use: expm1_cw_clamped clamps to -87.0f for the same reason.
        const sfpi::vFloat xc = sfpi::max(x, -88.5f);
        sfpi::vFloat result = xc * _sfpu_sigmoid_<is_fp32_dest_acc_en>(xc);

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
    // calculate_silu always uses the non-approx sigmoid path via _sfpu_sigmoid_, so we must
    // use non-approx sigmoid_init regardless of APPROXIMATION_MODE.
    sigmoid_init<false>();
}

}  // namespace ckernel::sfpu
