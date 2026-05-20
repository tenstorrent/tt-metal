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

// MXFP4 = S1E2M1: 4 bits packed two-per-byte (no padding).
inline constexpr FormatParams kMxFp4Params = {
    .block_size = 32,
    .scale_bias = 0x7F,
    .elem_exp_bits = 2,
    .elem_man_bits = 1,
    .elem_exp_bias = 1,
    .elem_exp_max_unbiased = 2,
    .elem_exp_min_unbiased = 0,
    .elem_exp_subnorm_encoding = 0,
    .elem_man_max = 0x1,
    .elem_width_bits = 4,
    .elem_width_storage_bits = 4,
    .sat_supported = true,
    .elem_sat_pos_bits = 0x7,
    .elem_sat_neg_bits = 0xF,
    .inf_rep = InfNanRepresentation::NotRepresentable,
    .nan_rep = InfNanRepresentation::NotRepresentable,
};

}  // namespace tt::tt_metal::mx

template <typename T>
inline std::vector<uint32_t> pack_as_mxfp4_tiles(
    tt::stl::Span<const T> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile = std::nullopt) {
    return tt::tt_metal::mx::pack_as_mx_tiles_impl(data, row_major_input, tile, tt::tt_metal::mx::kMxFp4Params);
}

std::vector<float> unpack_mxfp4_tiles_into_float_vec(
    tt::stl::Span<const uint32_t> mxfp4_tiles,
    bool row_major_output,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);
