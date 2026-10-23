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

// The reader stages each channel tile of a tile row as a left and a right strip of face rows: the 3 halo rows, then
// the tile's 32 rows, so the tile's faces land in them with four contiguous face reads and every shifted tile is four
// contiguous face copies out of the two strips (the forward's scheme, qkv_causal_conv1d_silu_layout.hpp).
constexpr uint32_t element_bytes = sizeof(uint16_t);
constexpr uint32_t face_rows = tt::constants::FACE_HEIGHT;
constexpr uint32_t face_row_bytes = tt::constants::FACE_WIDTH * element_bytes;
constexpr uint32_t face_bytes = tt::constants::FACE_HW * element_bytes;
constexpr uint32_t tile_bytes = tt::constants::TILE_HW * element_bytes;
constexpr uint32_t tile_row_bytes = tt::constants::TILE_WIDTH * element_bytes;
// Alignment of every DRAM read the reader issues into L1.
constexpr uint32_t read_alignment = 64;
constexpr uint32_t align_up(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}
// A strip starts strip_phase bytes past an aligned address so that its row halo_rows, where the tile's faces land,
// is aligned.
constexpr uint32_t strip_rows = halo_rows + tile_height;
constexpr uint32_t strip_stride = align_up(strip_rows * face_row_bytes, read_alignment);
constexpr uint32_t strip_phase = read_alignment - (halo_rows * face_row_bytes) % read_alignment;
// Reader scratch, from an aligned base: two parities of [left, right] strips per channel tile, the previous tile
// row's two bottom faces per channel tile, and the 3 history rows of the channel block. Slack for aligning the base.
struct ReaderScratch {
    uint32_t channel_tiles_per_block;
    constexpr uint32_t staging() const {
        return 2 * channel_tiles_per_block * 2 * strip_stride + read_alignment;
    }
    constexpr uint32_t rows() const {
        return staging() + channel_tiles_per_block * 2 * face_bytes;
    }
    constexpr uint32_t bytes() const {
        return rows() + halo_rows * channel_tiles_per_block * tile_row_bytes + read_alignment;
    }
};

}  // namespace qkv_conv_bw
