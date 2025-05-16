// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstring>

#include "dataflow_api.h"

constexpr uint32_t TILE_WIDTH = 32U;
constexpr uint32_t TILE_HEIGHT = 32U;
constexpr uint32_t FACE_WIDTH = 16U;
constexpr uint32_t FACE_HEIGHT = 16U;

inline uint32_t get_tilized_idx(uint32_t h, uint32_t w) {
    // Get local coordinates within the tile
    uint32_t local_row = h % TILE_HEIGHT;
    uint32_t local_col = w % TILE_WIDTH;

    // Determine the index offset based on which quadrant we're in
    uint32_t offset = 0;

    // If we're in the right half (columns beyond FACE_WIDTH)
    if (local_col >= FACE_WIDTH) {
        local_col -= FACE_WIDTH;
        offset += FACE_HEIGHT * FACE_WIDTH;  // Right face offset
    }

    // If we're in the bottom half (rows beyond FACE_WIDTH)
    if (local_row >= FACE_WIDTH) {
        local_row -= FACE_WIDTH;
        offset += FACE_HEIGHT * TILE_WIDTH;  // Bottom face offset
    }

    // Final index within the tile
    uint32_t index = offset + local_row * FACE_WIDTH + local_col;
    return index;
}

inline std::pair<uint32_t, uint32_t> get_page_and_offset(uint32_t tiled_row, uint32_t tiled_H) {
    uint32_t n = tiled_row / tiled_H;
    uint32_t h = (tiled_row % tiled_H) * 32;

    uint32_t page = n;
    uint32_t offset = h * sizeof(uint32_t);
    return {page, offset};
}
