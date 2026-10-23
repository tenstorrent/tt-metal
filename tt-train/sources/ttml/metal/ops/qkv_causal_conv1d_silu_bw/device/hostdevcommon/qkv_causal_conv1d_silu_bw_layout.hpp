// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Constants shared by the host and the kernels of qkv_causal_conv1d_silu_bw.
#pragma once

#include <cstdint>
#ifdef KERNEL_BUILD
#include "api/tile_constants.h"
#else
#include <tt-metalium/constants.hpp>
#endif

namespace qkv_conv_bw {

constexpr uint32_t tap_count = 4;
constexpr uint32_t halo_rows = tap_count - 1;  // rows of history above a tile row
constexpr uint32_t tile_height = tt::constants::TILE_HEIGHT;
constexpr uint32_t window_rows = halo_rows + tile_height;
// Tiles per DST acquire in the compute kernel (fp32 DST); the sigmoid buffer is one channel block, so every step ends
// on its boundary whatever the batch.
constexpr uint32_t dest_tiles = 4;

}  // namespace qkv_conv_bw
