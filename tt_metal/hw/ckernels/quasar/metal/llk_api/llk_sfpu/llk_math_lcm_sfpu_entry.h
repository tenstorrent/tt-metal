// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_lcm.h"

namespace ckernel {
namespace sfpu {

// Compute API entry points for lcm_tile_init / lcm_tile, named as on Blackhole / Wormhole. Int32 Dest
// holds two's complement (SIGN_MAGNITUDE_FORMAT = false), and |operand| <= 2^15 - 1 as lcm_tile documents.

inline void calculate_sfpu_lcm_init() { calculate_lcm_init(); }

inline void calculate_sfpu_lcm(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    calculate_lcm(dst_index_in0, dst_index_in1, dst_index_out);
}

}  // namespace sfpu
}  // namespace ckernel
