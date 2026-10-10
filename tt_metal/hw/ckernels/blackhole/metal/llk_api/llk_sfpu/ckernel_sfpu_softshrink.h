// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

inline void softshrink_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

bool bf16_dest_softshrink();
template <int ITERATIONS>
void calculate_softshrink_bf16();
void init_softshrink_bf16();
// Whether BF16 DEST runs the generated softshrink kernel as one call over the whole tile.
inline constexpr bool softshrink_bf16_whole_tile = true;

template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en = true>
inline void calculate_softshrink(uint32_t param0) {
    if constexpr (!is_fp32_dest_acc_en && ITERATIONS == 32) {
        if (bf16_dest_softshrink() && param0 == 0x3f000000u) {
            init_softshrink_bf16();
            calculate_softshrink_bf16<ITERATIONS>();
            return;
        }
    }
    // Softshrink(x) = x - λ if x > λ, x + λ if x < -λ, else 0
    // SFPU microcode
    sfpi::vFloat lambda = Converter::as_float(param0);
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = 0.0f;
        v_if(v > lambda) { sfpi::dst_reg[0] = v - lambda; }
        v_elseif(v < (-lambda)) { sfpi::dst_reg[0] = v + lambda; }
        v_endif;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu

#include "ckernel_sfpu_softshrink_bf16.h"
