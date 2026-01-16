// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <random>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/bfloat4.hpp>
#include "impl/data_format/blockfloat_common.hpp"
#include <vector>
#include <tt_stl/assert.hpp>

namespace tt::test_utils {

template <tt::DataFormat BfpFormat>
auto create_random_vector_of_bfp(uint32_t num_bytes, bool is_exp_a, int rand_max_float, int seed, float offset = 0.0f) {
    uint32_t single_bfp_tile_size = tile_size(BfpFormat);
    TT_FATAL(
        num_bytes % single_bfp_tile_size == 0,
        "num_bytes {} must be divisible by tile_size {}",
        num_bytes,
        single_bfp_tile_size);
    uint32_t num_tiles = num_bytes / single_bfp_tile_size;

    auto rand_float = std::bind(std::uniform_real_distribution<float>(0, rand_max_float), std::mt19937(seed));

    int packed_data_size = num_bytes / sizeof(float);
    int num_float_in_tile = 1024;
    int float_data_size = num_tiles * num_float_in_tile;

    std::vector<float> fp32_vec(float_data_size, 0);
    for (float& val : fp32_vec) {
        val = rand_float() + offset;
    }

    std::vector<uint32_t> packed_result =
        pack_as_bfp_tiles<BfpFormat>(ttsl::make_const_span(fp32_vec), /*row_major_input=*/true, is_exp_a);

    TT_FATAL(
        packed_result.size() == packed_data_size,
        "packed_result size {} does not match expected size {}",
        packed_result.size(),
        packed_data_size);

    return packed_result;
}

inline std::vector<uint32_t> create_random_vector_of_bfp4(
    uint32_t num_bytes, bool is_exp_a, int rand_max_float, int seed, float offset = 0.0f) {
    return create_random_vector_of_bfp<tt::DataFormat::Bfp4_b>(num_bytes, is_exp_a, rand_max_float, seed, offset);
}

inline std::vector<uint32_t> create_random_vector_of_bfp8(
    uint32_t num_bytes, bool is_exp_a, int rand_max_float, int seed, float offset = 0.0f) {
    return create_random_vector_of_bfp<tt::DataFormat::Bfp8_b>(num_bytes, is_exp_a, rand_max_float, seed, offset);
}

inline std::vector<uint32_t> create_constant_vector_of_bfp8(uint32_t num_bytes, float value, bool is_exp_a) {
    uint32_t single_bfp8_tile_size = tile_size(tt::DataFormat::Bfp8_b);
    TT_FATAL(
        num_bytes % single_bfp8_tile_size == 0,
        "num_bytes {} must be divisible by bfp8 tile_size {}",
        num_bytes,
        single_bfp8_tile_size);
    uint32_t num_tiles = num_bytes / single_bfp8_tile_size;

    int packed_data_size = num_bytes / sizeof(float);
    int num_float_in_tile = 1024;
    int float_data_size = num_tiles * num_float_in_tile;

    std::vector<float> fp32_vec(float_data_size, 0);
    for (float& val : fp32_vec) {
        val = value;
    }

    std::vector<uint32_t> packed_result =
        pack_as_bfp8_tiles(ttsl::make_const_span(fp32_vec), /*row_major_input=*/true, is_exp_a);

    TT_FATAL(
        packed_result.size() == packed_data_size,
        "packed_result size {} does not match expected size {}",
        packed_result.size(),
        packed_data_size);

    return packed_result;
}
}  // namespace tt::test_utils
