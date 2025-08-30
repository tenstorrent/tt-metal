// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <random>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/bfloat4.hpp>

#include <vector>
#include "assert.hpp"

namespace tt::test_utils {

inline std::vector<uint32_t> create_random_vector_of_bfp4(
    uint32_t num_bytes, bool is_exp_a, int rand_max_float, int seed, float offset = 0.0f) {
    uint32_t single_bfp4_tile_size = tile_size(tt::DataFormat::Bfp4_b);
    TT_ASSERT(num_bytes % single_bfp4_tile_size == 0);
    uint32_t num_tiles = num_bytes / single_bfp4_tile_size;

    auto rand_float = std::bind(std::uniform_real_distribution<float>(0, rand_max_float), std::mt19937(seed));

    int packed_data_size = num_bytes / sizeof(float);
    int num_float_in_tile = 1024;
    int float_data_size = num_tiles * num_float_in_tile;

    std::vector<float> fp32_vec(float_data_size, 0);
    for (int i = 0; i < fp32_vec.size(); i++) {
        fp32_vec.at(i) = rand_float() + offset;
    }

    std::vector<uint32_t> packed_result =
        pack_as_bfp4_tiles(tt::stl::make_const_span(fp32_vec), /*row_major_input=*/true, is_exp_a);

    TT_ASSERT(packed_result.size() == packed_data_size);

    return packed_result;
}

inline std::vector<uint32_t> create_random_vector_of_bfp8(
    uint32_t num_bytes, bool is_exp_a, int rand_max_float, int seed, float offset = 0.0f) {
    uint32_t single_bfp8_tile_size = tile_size(tt::DataFormat::Bfp8_b);
    TT_ASSERT(num_bytes % single_bfp8_tile_size == 0);
    uint32_t num_tiles = num_bytes / single_bfp8_tile_size;

    auto rand_float = std::bind(std::uniform_real_distribution<float>(0, rand_max_float), std::mt19937(seed));

    int packed_data_size = num_bytes / sizeof(float);
    int num_float_in_tile = 1024;
    int float_data_size = num_tiles * num_float_in_tile;

    std::vector<float> fp32_vec(float_data_size, 0);
    for (int i = 0; i < fp32_vec.size(); i++) {
        fp32_vec.at(i) = rand_float() + offset;
    }

    std::vector<uint32_t> packed_result =
        pack_as_bfp8_tiles(tt::stl::make_const_span(fp32_vec), /*row_major_input=*/true, is_exp_a);

    TT_ASSERT(packed_result.size() == packed_data_size);

    return packed_result;
}

inline std::vector<uint32_t> create_constant_vector_of_bfp8(uint32_t num_bytes, float value, bool is_exp_a) {
    uint32_t single_bfp8_tile_size = tile_size(tt::DataFormat::Bfp8_b);
    TT_ASSERT(num_bytes % single_bfp8_tile_size == 0);
    uint32_t num_tiles = num_bytes / single_bfp8_tile_size;

    int packed_data_size = num_bytes / sizeof(float);
    int num_float_in_tile = 1024;
    int float_data_size = num_tiles * num_float_in_tile;

    std::vector<float> fp32_vec(float_data_size, 0);
    for (int i = 0; i < fp32_vec.size(); i++) {
        fp32_vec.at(i) = value;
    }

    std::vector<uint32_t> packed_result =
        pack_as_bfp8_tiles(tt::stl::make_const_span(fp32_vec), /*row_major_input=*/true, is_exp_a);

    TT_ASSERT(packed_result.size() == packed_data_size);

    return packed_result;
}
}  // namespace tt::test_utils
