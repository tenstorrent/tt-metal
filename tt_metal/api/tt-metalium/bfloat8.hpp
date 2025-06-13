// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <tt-metalium/tile.hpp>
#include <tt_stl/span.hpp>
#include <optional>
#include <vector>

std::vector<uint32_t> pack_fp32_vec_as_bfp8_tiles(
    tt::stl::Span<const float> fp32_vec,
    bool row_major_input,
    bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

std::vector<float> unpack_bfp8_tiles_into_float_vec(
    tt::stl::Span<const uint32_t> bfp8_tiles,
    bool row_major_output,
    bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

std::vector<uint32_t> create_random_vector_of_bfp8(
    uint32_t num_bytes, bool is_exp_a, int rand_max_float, int seed, float offset = 0.0f);

std::vector<uint32_t> create_constant_vector_of_bfp8(uint32_t num_bytes, float value, bool is_exp_a);
