// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_gcd.h"

namespace ckernel {
namespace sfpu {

// Compute API entry points for gcd_tile_init / gcd_tile, named as on Blackhole / Wormhole. Int32 Dest
// holds two's complement (SIGN_MAGNITUDE_FORMAT = false), as the Blackhole kernel assumes.

inline void calculate_sfpu_gcd_init() { calculate_gcd_init(); }

inline void calculate_sfpu_gcd(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    calculate_gcd(dst_index_in0, dst_index_in1, dst_index_out);
}

}  // namespace sfpu
}  // namespace ckernel
