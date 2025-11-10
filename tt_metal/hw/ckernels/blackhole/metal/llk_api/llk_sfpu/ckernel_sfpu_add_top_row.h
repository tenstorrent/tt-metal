// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_add_top_row.h"
#include "ckernel_instr_params.h"

namespace ckernel::sfpu {

template <DataFormat format>
inline void calculate_add_top_row(const uint tile_idx_0 = 0, const uint tile_idx_1 = 0, const uint tile_idx_dst = 0) {
    // Use the specified format and tile indices for the add_top_row operation
    _calculate_add_top_row_<format>(tile_idx_0, tile_idx_1, tile_idx_dst);
}

inline void init_add_top_row() {
    // Initialize add_top_row operation
    _init_add_top_row_();
}

}  // namespace ckernel::sfpu

