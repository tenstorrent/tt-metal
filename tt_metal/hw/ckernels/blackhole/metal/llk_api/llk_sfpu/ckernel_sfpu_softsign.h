// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_recip.h"
#include "cmath_common.h"

#include "ckernel_sfpu_softsign_bf16.h"

namespace ckernel::sfpu {

// Whether BF16 DEST runs the generated softsign kernel as one call over the whole tile.
inline constexpr bool softsign_bf16_whole_tile = true;
template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en = true>
inline void calculate_softsign() {
    if constexpr (!is_fp32_dest_acc_en && !APPROXIMATION_MODE && ITERATIONS == 32) {
        if (bf16_dest_softsign()) {
            calculate_softsign_bf16<ITERATIONS>();
            return;
        }
    }
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat tmp = sfpi::abs(v) + 1.0f;
        tmp = sfpu_reciprocal<APPROXIMATION_MODE>(tmp);
        sfpi::dst_reg[0] = v * tmp;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = true>
void init_softsign() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
    if constexpr (!is_fp32_dest_acc_en && !APPROXIMATION_MODE) {
        init_softsign_bf16();
    }
}

}  // namespace ckernel::sfpu
