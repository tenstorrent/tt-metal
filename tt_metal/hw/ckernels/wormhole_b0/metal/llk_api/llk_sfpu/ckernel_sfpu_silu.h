// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cmath_common.h"  // math::reset_counters, p_setrwc
#include "ckernel_sfpu_sigmoid.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

bool bf16_dest_silu();
template <int ITERATIONS>
void calculate_silu_bf16();
// Whether BF16 DEST runs the generated silu kernel as one call over the whole tile.
inline constexpr bool silu_bf16_whole_tile = true;

template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_silu() {
    if constexpr (!is_fp32_dest_acc_en && ITERATIONS == 32) {
        if (bf16_dest_silu()) {
            calculate_silu_bf16<ITERATIONS>();
            return;
        }
    }
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // silu(x) = x * sigmoid(x)
        sfpi::vFloat result = x * _sfpu_sigmoid_<is_fp32_dest_acc_en>(x);

        // Round to bfloat16 if not in fp32 accumulation mode
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

void init_silu_bf16();

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = true>
inline void silu_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    // calculate_silu always uses the non-approx sigmoid path via _sfpu_sigmoid_, so we must
    // use non-approx sigmoid_init regardless of APPROXIMATION_MODE.
    sigmoid_init<false>();
    if constexpr (!is_fp32_dest_acc_en) {
        init_silu_bf16();
    }
}

}  // namespace ckernel::sfpu

#include "ckernel_sfpu_silu_bf16.h"
