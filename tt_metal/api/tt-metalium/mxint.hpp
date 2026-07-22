// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/tile.hpp>
#include <tt_stl/span.hpp>

#include <cstdint>
#include <optional>
#include <vector>

// MxInt8/MxInt4/MxInt2 are OCP-microscaling signed-integer formats. Each tile is
// split into 32-element blocks; every block carries one shared E8M0 scale (8
// bits) plus signed two's-complement integer elements with an implicit
// power-of-two scale: MxInt8 is S1.6 (8-bit elements, scale 1/64), MxInt4 is
// S1.2 (4-bit, two per byte, scale 1/4), MxInt2 is S1.0 (2-bit, four per byte,
// scale 1). On-tile layout is [scales padded to L1 alignment][packed elements].

/**
 * @brief Pack a dense tensor into MxInt8 (S1.6, signed two's-complement int8
 *        elements with an implicit 1/64 scale and an E8M0 block scale) tiles.
 *
 * @tparam T Input element type (typically float or bfloat16-like host type).
 *           Only the types explicitly instantiated at the bottom of mxint.cpp
 *           are supported; others produce a link error.
 * @param data Flat input data buffer containing all tensor elements.
 * @param row_major_input True if @p data is row-major; false if tile-major.
 * @param tile Optional tile shape descriptor; uses the default tile when nullopt.
 * @return Packed MxInt8 tile payload words.
 */
template <typename T>
std::vector<uint32_t> pack_as_mxint8_tiles(
    ttsl::Span<const T> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

/**
 * @brief Pack a dense tensor into MxInt4 (S1.2, signed two's-complement int4
 *        elements packed two per byte with an implicit 1/4 scale and an E8M0
 *        block scale) tiles.
 *
 * @tparam T Input element type (typically float or bfloat16-like host type).
 *           Only the types explicitly instantiated at the bottom of mxint.cpp
 *           are supported; others produce a link error.
 * @param data Flat input data buffer containing all tensor elements.
 * @param row_major_input True if @p data is row-major; false if tile-major.
 * @param tile Optional tile shape descriptor; uses the default tile when nullopt.
 * @return Packed MxInt4 tile payload words.
 */
template <typename T>
std::vector<uint32_t> pack_as_mxint4_tiles(
    ttsl::Span<const T> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

/**
 * @brief Pack a dense tensor into MxInt2 (S1.0, signed two's-complement int2
 *        elements packed four per byte with an implicit unit scale and an E8M0
 *        block scale) tiles.
 *
 * @tparam T Input element type (typically float or bfloat16-like host type).
 *           Only the types explicitly instantiated at the bottom of mxint.cpp
 *           are supported; others produce a link error.
 * @param data Flat input data buffer containing all tensor elements.
 * @param row_major_input True if @p data is row-major; false if tile-major.
 * @param tile Optional tile shape descriptor; uses the default tile when nullopt.
 * @return Packed MxInt2 tile payload words.
 */
template <typename T>
std::vector<uint32_t> pack_as_mxint2_tiles(
    ttsl::Span<const T> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

/**
 * @brief Unpack MxInt8 tiles into a float vector.
 *
 * @param mxint_tiles Packed MxInt8 tile payload words.
 * @param row_major_output True to produce row-major output; false for tile-major.
 * @param tile Optional tile shape descriptor; uses the default tile when nullopt.
 * @return Decoded values as float.
 */
std::vector<float> unpack_mxint8_tiles_into_float_vec(
    ttsl::Span<const uint32_t> mxint_tiles,
    bool row_major_output,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

/**
 * @brief Unpack MxInt4 tiles into a float vector.
 *
 * @param mxint_tiles Packed MxInt4 tile payload words.
 * @param row_major_output True to produce row-major output; false for tile-major.
 * @param tile Optional tile shape descriptor; uses the default tile when nullopt.
 * @return Decoded values as float.
 */
std::vector<float> unpack_mxint4_tiles_into_float_vec(
    ttsl::Span<const uint32_t> mxint_tiles,
    bool row_major_output,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

/**
 * @brief Unpack MxInt2 tiles into a float vector.
 *
 * @param mxint_tiles Packed MxInt2 tile payload words.
 * @param row_major_output True to produce row-major output; false for tile-major.
 * @param tile Optional tile shape descriptor; uses the default tile when nullopt.
 * @return Decoded values as float.
 */
std::vector<float> unpack_mxint2_tiles_into_float_vec(
    ttsl::Span<const uint32_t> mxint_tiles,
    bool row_major_output,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

namespace tt::tt_metal {

/**
 * @brief Pack a dense tensor into MxInt8 tiles.
 * @see ::pack_as_mxint8_tiles for parameter and format details.
 */
template <typename T>
std::vector<uint32_t> pack_as_mxint8_tiles(
    ttsl::Span<const T> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile = std::nullopt) {
    return ::pack_as_mxint8_tiles(data, row_major_input, tile);
}

/**
 * @brief Pack a dense tensor into MxInt4 tiles.
 * @see ::pack_as_mxint4_tiles for parameter and format details.
 */
template <typename T>
std::vector<uint32_t> pack_as_mxint4_tiles(
    ttsl::Span<const T> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile = std::nullopt) {
    return ::pack_as_mxint4_tiles(data, row_major_input, tile);
}

/**
 * @brief Pack a dense tensor into MxInt2 tiles.
 * @see ::pack_as_mxint2_tiles for parameter and format details.
 */
template <typename T>
std::vector<uint32_t> pack_as_mxint2_tiles(
    ttsl::Span<const T> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile = std::nullopt) {
    return ::pack_as_mxint2_tiles(data, row_major_input, tile);
}

/**
 * @brief Unpack MxInt8 tiles into a float vector.
 * @see ::unpack_mxint8_tiles_into_float_vec for parameter details.
 */
inline std::vector<float> unpack_mxint8_tiles_into_float_vec(
    ttsl::Span<const uint32_t> mxint_tiles,
    bool row_major_output,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt) {
    return ::unpack_mxint8_tiles_into_float_vec(mxint_tiles, row_major_output, tile);
}

/**
 * @brief Unpack MxInt4 tiles into a float vector.
 * @see ::unpack_mxint4_tiles_into_float_vec for parameter details.
 */
inline std::vector<float> unpack_mxint4_tiles_into_float_vec(
    ttsl::Span<const uint32_t> mxint_tiles,
    bool row_major_output,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt) {
    return ::unpack_mxint4_tiles_into_float_vec(mxint_tiles, row_major_output, tile);
}

/**
 * @brief Unpack MxInt2 tiles into a float vector.
 * @see ::unpack_mxint2_tiles_into_float_vec for parameter details.
 */
inline std::vector<float> unpack_mxint2_tiles_into_float_vec(
    ttsl::Span<const uint32_t> mxint_tiles,
    bool row_major_output,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt) {
    return ::unpack_mxint2_tiles_into_float_vec(mxint_tiles, row_major_output, tile);
}

}  // namespace tt::tt_metal
