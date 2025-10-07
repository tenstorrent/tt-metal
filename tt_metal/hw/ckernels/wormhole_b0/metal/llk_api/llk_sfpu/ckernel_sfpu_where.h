// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_where_int32(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_in2, const uint dst_index_out) {
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;

    sfpi::vInt output_tensor = 0;
    sfpi::vInt true_tensor = 0;
    sfpi::vInt false_tensor = 0;
    sfpi::vInt cond = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];

    for (int i = 0; i < ITERATIONS; i++) {
        cond = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        true_tensor = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        false_tensor = sfpi::dst_reg[dst_index_in2 * dst_tile_size_sfpi];
        output_tensor = false_tensor;

        v_if(cond != 0) { output_tensor = true_tensor; }
        v_endif;

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = output_tensor;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
