// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <algorithm>
#include <cstring>
#include <random>

#include <tt-metalium/logger.hpp>

namespace tt {
namespace test_utils {

//! Generic Library of templated tilization functions.

// Given a tensor that is row-major datums, make it tilized
// so that its row major within a tile, and each tile's data
// is contiguous
template <typename T, unsigned int TileHeight = 32, unsigned int TileWidth = 32>
std::vector<T> tilize(std::vector<T> data, int rows, int cols) {
    TT_FATAL((rows % TileHeight) == 0, "rows={} % TileHeight={} must equal 0", rows, TileHeight);
    TT_FATAL((cols % TileWidth) == 0, "rows={} % TileHeight={} must equal 0", cols, TileWidth);
    TT_FATAL((data.size() == rows * cols), "data with size {} doesn't fit all {} x {} values", data.size(), rows, cols);
    constexpr unsigned int elements_in_tile = TileHeight * TileWidth;
    unsigned int num_tiles_r = rows / TileHeight;
    unsigned int num_tiles_c = cols / TileWidth;
    std::vector<T> result(rows * cols);
    for (auto r = 0; r < num_tiles_r; r++) {
        for (auto c = 0; c < num_tiles_c; c++) {
#pragma unroll
            for (auto i = 0; i < TileHeight; i++) {  // tile rows
                size_t src_idx = (r * elements_in_tile * num_tiles_c) + (i * num_tiles_c * TileWidth) + (c * TileWidth);
                size_t dst_idx = (r * elements_in_tile * num_tiles_c) + (c * elements_in_tile) + (i * TileWidth);
                std::memcpy(&result[dst_idx], &data[src_idx], TileWidth * sizeof(T));
            }
        }
    }
    return result;
}
// Given a tilized data (each tile's data is contiguous and row major within the tile)
// transform it back to row major full tensor. (This function inverts the tilize() function)
template <typename T, unsigned int TileHeight = 32, unsigned int TileWidth = 32>
std::vector<T> untilize(std::vector<T> data, int rows, int cols) {
    TT_FATAL((rows % TileHeight) == 0, "rows={} % TileHeight={} must equal 0", rows, TileHeight);
    TT_FATAL((cols % TileWidth) == 0, "rows={} % TileHeight={} must equal 0", cols, TileWidth);
    TT_FATAL((data.size() == rows * cols), "data with size {} doesn't fit all {} x {} values", data.size(), rows, cols);
    constexpr unsigned int elements_in_tile = TileHeight * TileWidth;
    unsigned int num_tiles_r = rows / TileHeight;
    unsigned int num_tiles_c = cols / TileWidth;
    std::vector<T> result(rows * cols);
    for (auto r = 0; r < num_tiles_r; r++) {
        for (auto i = 0; i < TileHeight; i++) {
            for (auto c = 0; c < num_tiles_c; c++) {
                // Note: the only difference with tilize - switched src and dst indices
                size_t src_idx = (r * elements_in_tile * num_tiles_c) + (c * elements_in_tile) + (i * TileWidth);
                size_t dst_idx = (r * elements_in_tile * num_tiles_c) + (i * num_tiles_c * TileWidth) + (c * TileWidth);
                std::memcpy(&result[dst_idx], &data[src_idx], TileWidth * sizeof(T));
            }
        }
    }
    return result;
}

}  // namespace test_utils
}  // namespace tt
