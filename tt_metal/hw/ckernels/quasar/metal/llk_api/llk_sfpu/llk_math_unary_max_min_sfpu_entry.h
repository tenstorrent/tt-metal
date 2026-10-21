// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_unary_max_min.h"
#include "llk_assert.h"

namespace ckernel {
namespace sfpu {

// Compute API entry points for unary_{max,min}{,_int32,_uint32}_tile, named as on Blackhole / Wormhole.
// The kernel takes the Dest format as a template argument (any float width runs the Float32 path) and
// needs no init beyond the common SFPU one. Int32 Dest holds two's complement.

template <bool IS_MAX_OP = true>
inline void unary_max_min_init() {}

template <bool IS_MAX_OP = true, bool IS_UNSIGNED = false>
inline void unary_max_min_int32_init() {}

template <bool IS_MAX_OP, bool APPROXIMATION_MODE, int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_unary_max_min(const std::uint32_t value) {
    calculate_unary_max_min<IS_MAX_OP, DataFormat::Float32, APPROXIMATION_MODE, ITERATIONS>(value);
}

// Quasar has no UInt32 Dest format, so the unsigned variants (unary_*_uint32_tile) reject the call.
template <bool IS_MAX_OP, bool IS_UNSIGNED, bool APPROXIMATION_MODE, int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_unary_max_min_int32([[maybe_unused]] const std::uint32_t value) {
    if constexpr (IS_UNSIGNED) {
        LLK_ASSERT(false, "unary_max_uint32_tile / unary_min_uint32_tile are not supported on Quasar");
    } else {
        calculate_unary_max_min<IS_MAX_OP, DataFormat::Int32, APPROXIMATION_MODE, ITERATIONS>(value);
    }
}

}  // namespace sfpu
}  // namespace ckernel
