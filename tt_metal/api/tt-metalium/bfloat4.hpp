// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <tt-metalium/tile.hpp>
#include <tt_stl/span.hpp>

#include <optional>
#include <vector>

constexpr int log2(int n) {
    int log = 0;
    while (n >>= 1) {
        ++log;
    }
    return log;
}

[[deprecated("Use pack_as_bfp4_tiles instead.")]]
std::vector<uint32_t> pack_fp32_vec_as_bfp4_tiles(
    tt::stl::Span<const float> fp32_vec,
    bool row_major_input,
    bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

template <typename T>
std::vector<uint32_t> pack_as_bfp4_tiles(
    tt::stl::Span<const T> data,
    bool row_major_input,
    bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

std::vector<float> unpack_bfp4_tiles_into_float_vec(
    tt::stl::Span<const uint32_t> bfp_tiles,
    bool row_major_output,
    bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);
