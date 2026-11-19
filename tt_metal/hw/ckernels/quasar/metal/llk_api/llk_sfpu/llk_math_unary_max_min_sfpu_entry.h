// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_unary_max_min.h"

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

// Quasar has no UInt32 data format: the compute API's unary_*_uint32_tile static_assert when called, so only the
// Int32 path is reachable here.
template <bool IS_MAX_OP, bool IS_UNSIGNED, bool APPROXIMATION_MODE, int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_unary_max_min_int32(const std::uint32_t value) {
    static_assert(!IS_UNSIGNED, "unary_max_uint32_tile / unary_min_uint32_tile are not supported on Quasar");
    calculate_unary_max_min<IS_MAX_OP, DataFormat::Int32, APPROXIMATION_MODE, ITERATIONS>(value);
}

}  // namespace sfpu
}  // namespace ckernel
