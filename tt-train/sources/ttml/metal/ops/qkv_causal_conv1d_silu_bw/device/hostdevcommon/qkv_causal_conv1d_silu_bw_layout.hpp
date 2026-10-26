// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Constants shared by the host and the kernels of qkv_causal_conv1d_silu_bw.
#pragma once

#include <cstdint>

#include "ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/hostdevcommon/qkv_causal_conv1d_silu_layout.hpp"

namespace qkv_conv_bw {

// The backward stages the input the way the forward does: the same taps, face and strip constants and the same
// ReaderScratch, all from the forward's header.
using namespace qkv_conv;

constexpr uint32_t halo_rows = history_rows;  // rows of history above a tile row
constexpr uint32_t window_rows = halo_rows + tile_height;
// Tiles per DST acquire in the compute kernel (fp32 DST); the sigmoid buffer is one channel block, so every step ends
// on its boundary whatever the batch.
constexpr uint32_t dest_tiles = max_dest_tiles_fp32;

}  // namespace qkv_conv_bw
