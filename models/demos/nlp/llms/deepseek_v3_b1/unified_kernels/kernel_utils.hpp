// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#if defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#endif

// Firmware-set logical coordinates (defined in brisc.cc, ncrisc.cc, trisc.cc)
extern uint8_t my_logical_x_;
extern uint8_t my_logical_y_;

namespace unified_kernels {

// ============================================================================
// Grid coordinate utilities
// ============================================================================

// Compute linear index within a grid defined by start/end coordinates
// RowMajor=true:  index = rel_y * grid_width + rel_x  (iterate x first, then y)
// RowMajor=false: index = rel_x * grid_height + rel_y (iterate y first, then x)
template <bool RowMajor>
uint32_t linear_id_in_grid(uint32_t grid_start_x, uint32_t grid_start_y, uint32_t grid_end_x, uint32_t grid_end_y) {
    uint32_t rel_x = my_logical_x_ - grid_start_x;
    uint32_t rel_y = my_logical_y_ - grid_start_y;
    if constexpr (RowMajor) {
        uint32_t grid_width = grid_end_x - grid_start_x + 1;
        return rel_y * grid_width + rel_x;
    } else {
        uint32_t grid_height = grid_end_y - grid_start_y + 1;
        return rel_x * grid_height + rel_y;
    }
}

// ============================================================================
// Sharded persistent buffer setup utilities
// ============================================================================

#if defined(COMPILE_FOR_NCRISC)

// Setup a sharded persistent buffer by reserving and pushing tiles
// This makes the buffer available for compute to read from
FORCE_INLINE void setup_sharded_buffer(uint32_t cb_id, uint32_t num_tiles) {
    cb_reserve_back(cb_id, num_tiles);
    cb_push_back(cb_id, num_tiles);
}

#endif

}  // namespace unified_kernels
