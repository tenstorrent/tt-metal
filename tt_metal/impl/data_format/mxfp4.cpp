// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/mxfp4.hpp>
#include <tt-metalium/bfloat16.hpp>

#include <optional>
#include <vector>

#include <tt_stl/span.hpp>

#include "mx_common.hpp"
#include "mx_tile_pack.hpp"
#include "tile.hpp"
#include "tracy/Tracy.hpp"

namespace {

// MXFP4 = S1E2M1: 4 bits packed two-per-byte (no padding).
constexpr tt::tt_metal::mx::FormatParams kMxFp4Params = {
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
    .inf_rep = tt::tt_metal::mx::InfNanRepresentation::NotRepresentable,
    .nan_rep = tt::tt_metal::mx::InfNanRepresentation::NotRepresentable,
};

}  // namespace

template <typename T>
std::vector<uint32_t> pack_as_mxfp4_tiles(
    tt::stl::Span<const T> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile) {
    ZoneScoped;
    return tt::tt_metal::mx::pack_as_mx_tiles_impl(data, row_major_input, tile, kMxFp4Params);
}

template std::vector<uint32_t> pack_as_mxfp4_tiles<bfloat16>(
    tt::stl::Span<const bfloat16> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile);

template std::vector<uint32_t> pack_as_mxfp4_tiles<float>(
    tt::stl::Span<const float> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile);

template std::vector<uint32_t> pack_as_mxfp4_tiles<int32_t>(
    tt::stl::Span<const int32_t> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile);

template std::vector<uint32_t> pack_as_mxfp4_tiles<uint32_t>(
    tt::stl::Span<const uint32_t> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile);

template std::vector<uint32_t> pack_as_mxfp4_tiles<uint8_t>(
    tt::stl::Span<const uint8_t> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile);

template std::vector<uint32_t> pack_as_mxfp4_tiles<uint16_t>(
    tt::stl::Span<const uint16_t> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile);

std::vector<float> unpack_mxfp4_tiles_into_float_vec(
    tt::stl::Span<const uint32_t> mxfp4_tiles, bool row_major_output, const std::optional<tt::tt_metal::Tile>& tile) {
    ZoneScoped;
    return tt::tt_metal::mx::unpack_mx_tiles_into_float_vec_impl(mxfp4_tiles, row_major_output, tile, kMxFp4Params);
}
