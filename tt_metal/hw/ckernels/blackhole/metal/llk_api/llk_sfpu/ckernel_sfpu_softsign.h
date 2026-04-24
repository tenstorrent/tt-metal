// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_softsign(std::uint32_t dst_index_in, std::uint32_t dst_index_out) {
    // SFPU microcode
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[dst_index_in * SFP_DST_TILE_ROWS];
        sfpi::vFloat tmp = sfpi::abs(v) + sfpi::vConst1;
        tmp = sfpu_reciprocal<APPROXIMATION_MODE>(tmp);
        sfpi::dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = v * tmp;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void init_softsign() {
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
