// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/tile.hpp>
#include <tt_stl/span.hpp>

#include <optional>
#include <vector>

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
