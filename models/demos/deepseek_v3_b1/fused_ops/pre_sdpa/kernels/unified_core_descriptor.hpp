// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "compile_time_args.h"

namespace pre_sdpa {

// Runtime logical coordinates (set by firmware)
extern uint8_t my_logical_x_;
extern uint8_t my_logical_y_;

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

    // Constructor: initialize from firmware-set logical coordinates
    UnifiedCoreDescriptor() : x(my_logical_x_), y(my_logical_y_) {}

    // Convenience: compute linear core ID
    uint32_t linear_id(uint32_t grid_width) const { return y * grid_width + x; }
};

}  // namespace pre_sdpa
