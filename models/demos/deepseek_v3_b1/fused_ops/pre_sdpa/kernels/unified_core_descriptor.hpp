// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "compile_time_args.h"

// Firmware-set logical coordinates (defined in brisc.cc, ncrisc.cc, trisc.cc)
extern uint8_t my_logical_x_;
extern uint8_t my_logical_y_;

namespace pre_sdpa {

// ============================================================================
// UnifiedCoreDescriptor: Uses named compile-time args from UnifiedKernelDescriptor
// Provides compile-time role flags for dead code elimination via if constexpr
// ============================================================================

struct UnifiedCoreDescriptor {
    // Compile-time role flags from UnifiedCompileTimeCoreDescriptor
    static constexpr bool is_input_core = get_named_compile_time_arg_val("is_input_core") == 1;
    static constexpr bool is_matmul_core = get_named_compile_time_arg_val("is_matmul_core") == 1;

    // Runtime: core position (set by firmware)
    uint8_t x;
    uint8_t y;

    // Constructor: initialize from firmware-set logical coordinates (available for all RISCs)
    UnifiedCoreDescriptor() : x(my_logical_x_), y(my_logical_y_) {}

    // Convenience: compute linear core ID (absolute)
    uint32_t linear_id(uint32_t grid_width) const { return y * grid_width + x; }

    // Compute linear index within a grid defined by start/end coordinates
    // RowMajor=true:  index = rel_y * grid_width + rel_x  (iterate x first, then y)
    // RowMajor=false: index = rel_x * grid_height + rel_y (iterate y first, then x)
    template <bool RowMajor = true>
    uint32_t linear_id_in_grid(
        uint32_t grid_start_x, uint32_t grid_start_y, uint32_t grid_end_x, uint32_t grid_end_y) const {
        uint32_t rel_x = x - grid_start_x;
        uint32_t rel_y = y - grid_start_y;
        if constexpr (RowMajor) {
            uint32_t grid_width = grid_end_x - grid_start_x + 1;
            return rel_y * grid_width + rel_x;
        } else {
            uint32_t grid_height = grid_end_y - grid_start_y + 1;
            return rel_x * grid_height + rel_y;
        }
    }
};

}  // namespace pre_sdpa
