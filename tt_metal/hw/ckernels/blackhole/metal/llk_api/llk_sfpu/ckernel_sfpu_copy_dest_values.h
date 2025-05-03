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

template <bool APPROXIMATION_MODE, bool IDST0_BIGGER, int ITERATIONS = 8>
void copy_dest_value(const uint dst_offset) {
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size = 32;
        if constexpr (IDST0_BIGGER) {
            dst_reg[dst_offset * dst_tile_size] = dst_reg[0];
        } else {
            dst_reg[0] = dst_reg[dst_offset * dst_tile_size];
        }
        dst_reg++;
    }
}

void copy_dest_value_init() {
    // No initialization required
}

}  // namespace sfpu
}  // namespace ckernel
