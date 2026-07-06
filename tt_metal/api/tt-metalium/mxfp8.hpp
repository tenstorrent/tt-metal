// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/tile.hpp>
#include <tt_stl/span.hpp>

#include <cstdint>
#include <optional>
#include <vector>

/**
 * @brief Pack a dense tensor into MXFP8 E5M2 (a.k.a. MXFP8R, S1E5M2 with IEEE
 *        Inf/NaN, OCP microscaling) tiles.
 *
 * @tparam T Input element type (typically float or bfloat16-like host type).
 *           Only the types explicitly instantiated at the bottom of mxfp8.cpp
 *           are supported; others produce a link error.
 * @param data Flat input data buffer containing all tensor elements.
 * @param row_major_input True if @p data is row-major; false if tile-major.
 * @param tile Optional tile shape descriptor; uses the default tile when nullopt.
 * @return Packed MXFP8 E5M2 tile payload words.
 */
template <typename T>
std::vector<uint32_t> pack_as_mxfp8_e5m2_tiles(
    ttsl::Span<const T> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

/**
 * @brief Unpack MXFP8 E5M2 tiles into a float vector.
 *
 * @param mxfp8_tiles Packed MXFP8 E5M2 tile payload words.
 * @param row_major_output True to produce row-major output; false for tile-major.
 * @param tile Optional tile shape descriptor; uses the default tile when nullopt.
 * @return Decoded values as float.
 */
std::vector<float> unpack_mxfp8_e5m2_tiles_into_float_vec(
    ttsl::Span<const uint32_t> mxfp8_tiles,
    bool row_major_output,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

/**
 * @brief Pack a dense tensor into MXFP8 E4M3 (a.k.a. MXFP8P, S1E4M3FN finite-only,
 *        OCP microscaling) tiles.
 *
 * @tparam T Input element type (typically float or bfloat16-like host type).
 *           Only the types explicitly instantiated at the bottom of mxfp8.cpp
 *           are supported; others produce a link error.
 * @param data Flat input data buffer containing all tensor elements.
 * @param row_major_input True if @p data is row-major; false if tile-major.
 * @param tile Optional tile shape descriptor; uses the default tile when nullopt.
 * @return Packed MXFP8 E4M3 tile payload words.
 */
template <typename T>
std::vector<uint32_t> pack_as_mxfp8_e4m3_tiles(
    ttsl::Span<const T> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

/**
 * @brief Unpack MXFP8 E4M3 tiles into a float vector.
 *
 * @param mxfp8_tiles Packed MXFP8 E4M3 tile payload words.
 * @param row_major_output True to produce row-major output; false for tile-major.
 * @param tile Optional tile shape descriptor; uses the default tile when nullopt.
 * @return Decoded values as float.
 */
std::vector<float> unpack_mxfp8_e4m3_tiles_into_float_vec(
    ttsl::Span<const uint32_t> mxfp8_tiles,
    bool row_major_output,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

namespace tt::tt_metal {

template <typename T>
std::vector<uint32_t> pack_as_mxfp8_e5m2_tiles(
    ttsl::Span<const T> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile = std::nullopt) {
    return ::pack_as_mxfp8_e5m2_tiles(data, row_major_input, tile);
}

inline std::vector<float> unpack_mxfp8_e5m2_tiles_into_float_vec(
    ttsl::Span<const uint32_t> mxfp8_tiles,
    bool row_major_output,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt) {
    return ::unpack_mxfp8_e5m2_tiles_into_float_vec(mxfp8_tiles, row_major_output, tile);
}

template <typename T>
std::vector<uint32_t> pack_as_mxfp8_e4m3_tiles(
    ttsl::Span<const T> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile = std::nullopt) {
    return ::pack_as_mxfp8_e4m3_tiles(data, row_major_input, tile);
}

inline std::vector<float> unpack_mxfp8_e4m3_tiles_into_float_vec(
    ttsl::Span<const uint32_t> mxfp8_tiles,
    bool row_major_output,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt) {
    return ::unpack_mxfp8_e4m3_tiles_into_float_vec(mxfp8_tiles, row_major_output, tile);
}

}  // namespace tt::tt_metal
