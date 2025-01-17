// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <algorithm>
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
    constexpr unsigned int elements_in_tile = TileHeight * TileWidth;
    unsigned int num_tiles_r = rows / TileHeight;
    unsigned int num_tiles_c = cols / TileWidth;
    std::vector<T> result;
    for (auto r = 0; r < num_tiles_r; r++) {
        for (auto c = 0; c < num_tiles_c; c++) {
            for (auto j = 0; j < TileHeight; j++) {     // tile rows
                for (auto i = 0; i < TileWidth; i++) {  // tile cols
                    int index = r * elements_in_tile * num_tiles_c + j * cols + c * TileWidth + i;
                    result.push_back(data.at(index));
                }
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
    constexpr unsigned int elements_in_tile = TileHeight * TileWidth;
    unsigned int num_tiles_r = rows / TileHeight;
    unsigned int num_tiles_c = cols / TileWidth;
    std::vector<T> result;
    for (auto r = 0; r < num_tiles_r; r++) {
        for (auto i = 0; i < TileHeight; i++) {
            for (auto c = 0; c < num_tiles_c; c++) {
                int offset = r * elements_in_tile * num_tiles_c + c * elements_in_tile + i * TileWidth;
                for (auto j = 0; j < TileWidth; j++) {
                    result.push_back(data.at(offset + j));
                }
            }
        }
    }
    return result;
}

}  // namespace test_utils
}  // namespace tt
