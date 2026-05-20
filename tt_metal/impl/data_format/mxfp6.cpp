// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/mxfp6.hpp>

#include <optional>
#include <vector>

#include <tt_stl/span.hpp>

#include <tt-metalium/mx_tile_pack.hpp>
#include <tt-metalium/tile.hpp>
#include "tracy/Tracy.hpp"

std::vector<float> unpack_mxfp6r_tiles_into_float_vec(
    tt::stl::Span<const uint32_t> mxfp6_tiles, bool row_major_output, const std::optional<tt::tt_metal::Tile>& tile) {
    ZoneScoped;
    return tt::tt_metal::mx::unpack_mx_tiles_into_float_vec_impl(
        mxfp6_tiles, row_major_output, tile, tt::tt_metal::mx::kMxFp6RParams);
}

std::vector<float> unpack_mxfp6p_tiles_into_float_vec(
    tt::stl::Span<const uint32_t> mxfp6_tiles, bool row_major_output, const std::optional<tt::tt_metal::Tile>& tile) {
    ZoneScoped;
    return tt::tt_metal::mx::unpack_mx_tiles_into_float_vec_impl(
        mxfp6_tiles, row_major_output, tile, tt::tt_metal::mx::kMxFp6PParams);
}
