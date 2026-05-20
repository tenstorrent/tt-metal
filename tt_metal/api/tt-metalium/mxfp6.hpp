// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/mx_common.hpp>
#include <tt-metalium/mx_tile_pack.hpp>
#include <tt-metalium/tile.hpp>
#include <tt_stl/span.hpp>

#include <optional>
#include <vector>

namespace tt::tt_metal::mx {

// MXFP6R = S1E3M2: 6 bits stored in the high bits of an 8-bit byte (bits 7-2),
// bits 1-0 are zero.
inline constexpr FormatParams kMxFp6RParams = {
    .block_size = 32,
    .scale_bias = 0x7F,
    .elem_exp_bits = 3,
    .elem_man_bits = 2,
    .elem_exp_bias = 3,
    .elem_exp_max_unbiased = 4,
    .elem_exp_min_unbiased = -2,
    .elem_exp_subnorm_encoding = 0,
    .elem_man_max = 0x3,
    .elem_width_bits = 6,
    .elem_width_storage_bits = 8,
    .sat_supported = true,
    .elem_sat_pos_bits = 0x1F,
    .elem_sat_neg_bits = 0x3F,
    .inf_rep = InfNanRepresentation::NotRepresentable,
    .nan_rep = InfNanRepresentation::NotRepresentable,
};

// MXFP6P = S1E2M3: 6 bits stored in the high bits of an 8-bit byte (bits 7-2),
// bits 1-0 are zero.
inline constexpr FormatParams kMxFp6PParams = {
    .block_size = 32,
    .scale_bias = 0x7F,
    .elem_exp_bits = 2,
    .elem_man_bits = 3,
    .elem_exp_bias = 1,
    .elem_exp_max_unbiased = 2,
    .elem_exp_min_unbiased = 0,
    .elem_exp_subnorm_encoding = 0,
    .elem_man_max = 0x7,
    .elem_width_bits = 6,
    .elem_width_storage_bits = 8,
    .sat_supported = true,
    .elem_sat_pos_bits = 0x1F,
    .elem_sat_neg_bits = 0x3F,
    .inf_rep = InfNanRepresentation::NotRepresentable,
    .nan_rep = InfNanRepresentation::NotRepresentable,
};

}  // namespace tt::tt_metal::mx

template <typename T>
inline std::vector<uint32_t> pack_as_mxfp6r_tiles(
    tt::stl::Span<const T> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile = std::nullopt) {
    return tt::tt_metal::mx::pack_as_mx_tiles_impl(data, row_major_input, tile, tt::tt_metal::mx::kMxFp6RParams);
}

std::vector<float> unpack_mxfp6r_tiles_into_float_vec(
    tt::stl::Span<const uint32_t> mxfp6_tiles,
    bool row_major_output,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

template <typename T>
inline std::vector<uint32_t> pack_as_mxfp6p_tiles(
    tt::stl::Span<const T> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile = std::nullopt) {
    return tt::tt_metal::mx::pack_as_mx_tiles_impl(data, row_major_input, tile, tt::tt_metal::mx::kMxFp6PParams);
}

std::vector<float> unpack_mxfp6p_tiles_into_float_vec(
    tt::stl::Span<const uint32_t> mxfp6_tiles,
    bool row_major_output,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);
