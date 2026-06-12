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
 * @brief Pack a dense tensor into an MX block-scaled signed-integer format.
 *
 * MxInt8/MxInt4/MxInt2 store 32-element blocks, each with one shared E8M0 scale
 * (8 bits) and signed two's-complement integer elements with an implicit
 * power-of-two scale (1/64 for MxInt8's S1.6, 1/4 for MxInt4's S1.2, 1 for
 * MxInt2's S1.0). Tile layout: [scales padded to L1 alignment][packed elements].
 *
 * @tparam T Input element type (typically float). Only the types explicitly
 *           instantiated at the bottom of mxint.cpp are supported.
 * @param data Flat input data buffer containing all tensor elements.
 * @param row_major_input True if @p data is row-major; false if tile-major.
 * @param tile Optional tile shape descriptor; uses the default tile when nullopt.
 * @return Packed MxInt tile payload words.
 */
template <typename T>
std::vector<uint32_t> pack_as_mxint8_tiles(
    tt::stl::Span<const T> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

template <typename T>
std::vector<uint32_t> pack_as_mxint4_tiles(
    tt::stl::Span<const T> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

template <typename T>
std::vector<uint32_t> pack_as_mxint2_tiles(
    tt::stl::Span<const T> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

/**
 * @brief Unpack MxInt tiles into a float vector.
 *
 * @param mxint_tiles Packed MxInt tile payload words.
 * @param row_major_output True to produce row-major output; false for tile-major.
 * @param tile Optional tile shape descriptor; uses the default tile when nullopt.
 * @return Decoded values as float.
 */
std::vector<float> unpack_mxint8_tiles_into_float_vec(
    tt::stl::Span<const uint32_t> mxint_tiles,
    bool row_major_output,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

std::vector<float> unpack_mxint4_tiles_into_float_vec(
    tt::stl::Span<const uint32_t> mxint_tiles,
    bool row_major_output,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

std::vector<float> unpack_mxint2_tiles_into_float_vec(
    tt::stl::Span<const uint32_t> mxint_tiles,
    bool row_major_output,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

namespace tt::tt_metal {

template <typename T>
std::vector<uint32_t> pack_as_mxint8_tiles(
    tt::stl::Span<const T> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile = std::nullopt) {
    return ::pack_as_mxint8_tiles(data, row_major_input, tile);
}

template <typename T>
std::vector<uint32_t> pack_as_mxint4_tiles(
    tt::stl::Span<const T> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile = std::nullopt) {
    return ::pack_as_mxint4_tiles(data, row_major_input, tile);
}

template <typename T>
std::vector<uint32_t> pack_as_mxint2_tiles(
    tt::stl::Span<const T> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile = std::nullopt) {
    return ::pack_as_mxint2_tiles(data, row_major_input, tile);
}

inline std::vector<float> unpack_mxint8_tiles_into_float_vec(
    tt::stl::Span<const uint32_t> mxint_tiles,
    bool row_major_output,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt) {
    return ::unpack_mxint8_tiles_into_float_vec(mxint_tiles, row_major_output, tile);
}

inline std::vector<float> unpack_mxint4_tiles_into_float_vec(
    tt::stl::Span<const uint32_t> mxint_tiles,
    bool row_major_output,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt) {
    return ::unpack_mxint4_tiles_into_float_vec(mxint_tiles, row_major_output, tile);
}

inline std::vector<float> unpack_mxint2_tiles_into_float_vec(
    tt::stl::Span<const uint32_t> mxint_tiles,
    bool row_major_output,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt) {
    return ::unpack_mxint2_tiles_into_float_vec(mxint_tiles, row_major_output, tile);
}

}  // namespace tt::tt_metal
