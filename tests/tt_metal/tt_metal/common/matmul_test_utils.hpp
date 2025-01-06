// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "common/bfloat16.hpp"
#include "tt_metal/common/test_tiles.hpp"
#include "hostdevcommon/common_values.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"

using namespace tt;

inline std::vector<bfloat16> select_columns(std::vector<bfloat16> data, int M, int K, int N) {
    if (N == K) {
        return data;
    }
    std::vector<bfloat16> result;
    if (N > K) {
        for (int i = 0; i < M * 32; i++) {
            for (int j = 0; j < K * 32; j++) {
                int offset = i * K * 32;
                result.push_back(data.at(offset + j));
            }
            for (int j = 0; j < (N - K) * 32; j++) {
                result.push_back((float)0);
            }
        }
    } else {
        for (int i = 0; i < M * 32; i++) {
            for (int j = 0; j < N * 32; j++) {
                int offset = i * K * 32;
                result.push_back(data.at(offset + j));
            }
        }
    }

    return result;
}

inline std::vector<bfloat16> get_row_slice(
    std::vector<bfloat16> data, int total_row_slices, int row_slice_index, int rows, int cols) {
    std::vector<bfloat16> result;
    int rows_per_slice = rows / total_row_slices;
    for (int i = rows_per_slice * row_slice_index * cols; i < rows_per_slice * (row_slice_index + 1) * cols; i++) {
        result.push_back(data.at(i));
    }
    return result;
}

inline std::vector<bfloat16> get_col_slice(
    std::vector<bfloat16> data, int total_col_slices, int col_slice_index, int rows, int cols) {
    std::vector<bfloat16> result;
    int cols_per_slice = cols / total_col_slices;
    for (int r = 0; r < rows; r++) {
        for (int c = cols_per_slice * col_slice_index; c < cols_per_slice * (col_slice_index + 1); c++) {
            result.push_back(data.at(r * cols + c));
        }
    }
    return result;
}

inline void print_faces(std::vector<bfloat16> data, string name) {
    std::cout << name << ": " << std::endl;
    int index = 0;

    int tile_index = 0;
    int face_index = 0;
    for (int i = 0; i < data.size(); i++) {
        if (i % 256 == 0) {
            std::cout << "Tile " << tile_index / 4 << std::endl;
            std::cout << "Face = " << face_index << std::endl;
            face_index++;
            tile_index++;
            if (face_index == 4) {
                face_index = 0;
            }
        }
        std::cout << data.at(i).to_float() << ", ";
        if ((i + 1) % 16 == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

// Transpose 2D matrix of tiles so that its column major of tiles instead of row major.
// this is usually used for activation so that blocks data is contiguous in memory
// until we have a more generalized read kernel that can read tiles from different
// location in memory to make up a block in the activations CB
inline std::vector<std::uint32_t> transpose_tiles(
    std::vector<std::uint32_t> data, int row_tiles, int col_tiles, int in0_block_w) {
    std::vector<std::uint32_t> result;
    int tile_size = 512;
    for (int c = 0; c < col_tiles; c += in0_block_w) {
        for (int r = 0; r < row_tiles; r++) {
            for (int k = 0; k < in0_block_w; k++) {
                int offset = tile_size * col_tiles * r + c * tile_size + k * tile_size;
                for (int i = 0; i < tile_size; i++) {
                    result.push_back(data.at(offset + i));
                }
            }
        }
    }
    return result;
}

inline bool move_tiles_to_dram(
    tt_metal::Device* device, std::vector<uint32_t> tensor, int tiles_r, int tiles_c, uint32_t dram_buffer_addr) {
    bool pass = true;
    int tile_size = 512;  // 32*32 packed into u32
    int tile_size_bytes = 32 * 32 * 2;
    int start_index = 0;
    int tile_id = 0;
    std::vector<uint32_t> tile;
    for (int i = 0; i < tiles_r; i++) {
        for (int j = 0; j < tiles_c; j++) {
            tile.clear();
            tile.insert(tile.end(), tensor.begin() + start_index, tensor.begin() + start_index + tile_size);
            uint32_t dram_addr = (tile_id / device->num_dram_channels()) * tile_size_bytes + dram_buffer_addr;
            int dram_channel = tile_id % device->num_dram_channels();

            pass &= tt_metal::detail::WriteToDeviceDRAMChannel(device, dram_channel, dram_addr, tile);
            start_index += tile_size;
            tile_id++;
        }
    }
    return pass;
}

inline bool move_tiles_to_dram(
    tt_metal::Device* device, std::vector<uint32_t> tensor, int tiles_r, int tiles_c, std::shared_ptr<Buffer> buffer) {
    bool pass = true;
    int tile_size = 512;  // 32*32 packed into uint32_t
    int tile_size_bytes = 32 * 32 * 2;
    int start_index = 0;
    int tile_id = 0;
    CommandQueue& cq = device->command_queue();
    std::vector<uint32_t> tile;
    std::vector<uint32_t> tiles;
    for (int i = 0; i < tiles_r; i++) {
        for (int j = 0; j < tiles_c; j++) {
            tile.clear();
            tile.insert(tile.end(), tensor.begin() + start_index, tensor.begin() + start_index + tile_size);

            tiles.insert(tiles.end(), tile.begin(), tile.end());
            start_index += tile_size;
        }
    }

    EnqueueWriteBuffer(cq, buffer, tiles, false);
    return pass;
}
