// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_softshrink(std::uint32_t dst_index_in, std::uint32_t dst_index_out, uint32_t param0) {
    // Softshrink(x) = x - λ if x > λ, x + λ if x < -λ, else 0
    // SFPU microcode
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
    sfpi::vFloat lambda = Converter::as_float(param0);
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[dst_index_in * SFP_DST_TILE_ROWS];
        sfpi::dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = sfpi::vConst0;
        v_if(v > lambda) { sfpi::dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = v - lambda; }
        v_elseif(v < (-lambda)) { sfpi::dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = v + lambda; }
        v_endif;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
