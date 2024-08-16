// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "bfloat16.hpp"

template <typename T>
void tilize(std::vector<T>& input, uint32_t m, uint32_t n) {
    TT_FATAL(input.size() > 0 and m > 0 and n > 0, "None of the input size, m, nor n can be 0");
    TT_FATAL((input.size() % (m * n)) == 0, "Input size must be divisible by m  and n");

    std::vector<T> tilized_input;
    tilized_input.reserve(input.size());

    uint32_t block_num_elements = m * n;
    uint32_t num_blocks = input.size() / block_num_elements;

    const auto write_face =
        [](std::vector<T>& tilized_input, const std::vector<T>& input, uint32_t face_height, uint32_t face_width, uint32_t face_idx, uint32_t n)
        -> void {
        for (uint32_t i = 0; i < face_height; i++) {
            for (uint32_t j = 0; j < face_width; j++) {
                tilized_input.push_back(input[face_idx + j]);
            }
            face_idx += n;
        }
    };

    if constexpr (std::is_same<T, bfloat16>()) {
        uint32_t TILE_HEIGHT = 32;
        uint32_t TILE_WIDTH = 32;
        uint32_t FACE_HEIGHT = 16;
        uint32_t FACE_WIDTH = 16;
        uint32_t row_tiles = m / TILE_HEIGHT;
        uint32_t col_tiles = n / TILE_WIDTH;
        uint32_t row_of_tiles_num_elements = TILE_HEIGHT * n;
        TT_FATAL((m % TILE_HEIGHT == 0) and (n % TILE_WIDTH == 0), "m and n must be divisible by 32");
        uint32_t block_start = 0;
        for (size_t i = 0; i < num_blocks; i++) {
            uint32_t tile_start = block_start;
            for (uint32_t row_tile = 0; row_tile < row_tiles; row_tile++) {
                uint32_t row_tile_start = tile_start;
                for (uint32_t col_tile = 0; col_tile < col_tiles; col_tile++) {
                    uint32_t face0_id = row_tile_start;
                    uint32_t face1_id = face0_id + FACE_WIDTH;
                    uint32_t face2_id = face0_id + n * FACE_HEIGHT;
                    uint32_t face3_id = face2_id + FACE_WIDTH;

                    write_face(tilized_input, input, FACE_HEIGHT, FACE_WIDTH, face0_id, n);
                    write_face(tilized_input, input, FACE_HEIGHT, FACE_WIDTH, face1_id, n);
                    write_face(tilized_input, input, FACE_HEIGHT, FACE_WIDTH, face2_id, n);
                    write_face(tilized_input, input, FACE_HEIGHT, FACE_WIDTH, face3_id, n);
                    row_tile_start += TILE_WIDTH;
                }
                tile_start += row_of_tiles_num_elements;
            }
            block_start += block_num_elements;
        }
    } else {
        TT_THROW("Invalid type passed into tilize");
    }

    input = std::move(tilized_input);
}

template <typename T>
void untilize(std::vector<T>& input, uint32_t m, uint32_t n) {
    TT_FATAL(input.size() > 0 and m > 0 and n > 0, "None of the input size, m, nor n can be 0");
    TT_FATAL((input.size() % (m * n)) == 0, "Input size must be divisible by m  and n");

    std::vector<T> untilized_input;
    untilized_input.reserve(input.size());

    uint32_t block_num_elements = m * n;
    uint32_t num_blocks = input.size() / block_num_elements;

    const auto untilize_row = [](std::vector<T>& untilized_input,
                                 const std::vector<T>& input,
                                 uint32_t face_height,
                                 uint32_t face_width,
                                 uint32_t tile_idx,
                                 uint32_t TILE_WIDTH,
                                 uint32_t n) -> void {
        uint32_t face_num_elements = face_height * face_width;
        uint32_t face_start = tile_idx;
        for (uint32_t m = 0; m < 2; m++) {
            for (uint32_t i = 0; i < face_height; i++) {
                uint32_t row_start = face_start + i * face_width;
                for (uint32_t j = 0; j < n / TILE_WIDTH; j++) {  // Iterates over all the column tiles
                    // Grab 16 elements from tile j, face 0/2
                    for (uint32_t k = 0; k < face_width; k++) {
                        untilized_input.push_back(input[row_start + k]);
                    }

                    // Grab 16 elements from tile j, face 1/3
                    row_start += face_height * face_width;
                    for (uint32_t k = 0; k < face_width; k++) {
                        untilized_input.push_back(input[row_start + k]);
                    }
                    row_start += face_height * face_width * 3;  // If on face 1, need to get to face 0 of next tile, and
                                                                // if on face 3, need to get to face 2 of next tile
                }
            }
            face_start += face_height * face_width * 2;  // Get to face 2 of current tile
        }
    };

    if constexpr (std::is_same<T, bfloat16>()) {
        uint32_t TILE_HEIGHT = 32;
        uint32_t TILE_WIDTH = 32;
        uint32_t FACE_HEIGHT = 16;
        uint32_t FACE_WIDTH = 16;
        uint32_t row_tiles = m / TILE_HEIGHT;
        uint32_t col_tiles = n / TILE_WIDTH;
        uint32_t row_of_tiles_num_elements = TILE_HEIGHT * n;
        TT_FATAL((m % TILE_HEIGHT == 0) and (n % TILE_WIDTH == 0), "m and n must be divisible by 32");
        uint32_t block_start = 0;
        for (size_t i = 0; i < num_blocks; i++) {
            uint32_t row_tile_start = block_start;
            for (uint32_t row_tile = 0; row_tile < row_tiles; row_tile++) {
                untilize_row(untilized_input, input, FACE_HEIGHT, FACE_WIDTH, row_tile_start, TILE_WIDTH, n);
                row_tile_start += row_of_tiles_num_elements;
            }
            block_start += block_num_elements;
        }
    } else {
        TT_THROW("Invalid type passed into untilize");
    }

    input = std::move(untilized_input);
}
