// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/tile.hpp>
#include <tt_stl/span.hpp>

#include <cstdint>
#include <optional>
#include <vector>

// MXFP6R = S1E3M2 / MXFP6P = S1E2M3: 6 bits stored in the high bits of an 8-bit
// byte (bits 7-2), bits 1-0 are zero.
//
// Public pack/unpack API. The underlying MX toolkit (FormatParams, the generic
// pack/unpack implementation, and the per-format descriptors) is an internal
// detail and lives in tt_metal/impl/data_format/. The pack_as_mxfp6*_tiles
// templates are only instantiated for the element types listed at the bottom of
// mxfp6.cpp.

template <typename T>
std::vector<uint32_t> pack_as_mxfp6r_tiles(
    tt::stl::Span<const T> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

std::vector<float> unpack_mxfp6r_tiles_into_float_vec(
    tt::stl::Span<const uint32_t> mxfp6_tiles,
    bool row_major_output,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

template <typename T>
std::vector<uint32_t> pack_as_mxfp6p_tiles(
    tt::stl::Span<const T> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

std::vector<float> unpack_mxfp6p_tiles_into_float_vec(
    tt::stl::Span<const uint32_t> mxfp6_tiles,
    bool row_major_output,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);
