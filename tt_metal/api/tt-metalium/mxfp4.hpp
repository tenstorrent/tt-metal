// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/tile.hpp>
#include <tt_stl/span.hpp>

#include <cstdint>
#include <optional>
#include <vector>

// MXFP4 = S1E2M1: 4 bits packed two-per-byte (no padding).
//
// Public pack/unpack API. The underlying MX toolkit (FormatParams, the generic
// pack/unpack implementation, and the per-format descriptors) is an internal
// detail and lives in tt_metal/impl/data_format/. pack_as_mxfp4_tiles is only
// instantiated for the element types listed at the bottom of mxfp4.cpp.

template <typename T>
std::vector<uint32_t> pack_as_mxfp4_tiles(
    tt::stl::Span<const T> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

std::vector<float> unpack_mxfp4_tiles_into_float_vec(
    tt::stl::Span<const uint32_t> mxfp4_tiles,
    bool row_major_output,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

namespace tt::tt_metal {

template <typename T>
std::vector<uint32_t> pack_as_mxfp4_tiles(
    tt::stl::Span<const T> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile = std::nullopt) {
    return ::pack_as_mxfp4_tiles(data, row_major_input, tile);
}

inline std::vector<float> unpack_mxfp4_tiles_into_float_vec(
    tt::stl::Span<const uint32_t> mxfp4_tiles,
    bool row_major_output,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt) {
    return ::unpack_mxfp4_tiles_into_float_vec(mxfp4_tiles, row_major_output, tile);
}

}  // namespace tt::tt_metal
