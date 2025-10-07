// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
void copy_dest_value(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out /* unused */) {
    for (int d = 0; d < ITERATIONS; d++) {
        // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
        constexpr uint dst_tile_size_sfpi = 32;
        dst_reg[dst_index_in0 * dst_tile_size_sfpi] = dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        dst_reg++;
    }
}

void copy_dest_value_init() {
    // No initialization required
}

}  // namespace sfpu
}  // namespace ckernel
