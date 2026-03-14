// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <tt-metalium/tile.hpp>
#include <tt_stl/span.hpp>

#include <optional>
#include <vector>

/**
 * @brief Pack a dense tensor into BFP2 tiles.
 *
 * @tparam T Input element type (typically float or bfloat16-like host type).
 * @param data Flat input data buffer containing all tensor elements.
 * @param row_major_input True if @p data is row-major; false if tile-major.
 * @param is_exp_a Selects exponent sharing mode used by the BFP2 encoder.
 * @param tile Optional tile shape descriptor; uses the default tile when nullopt.
 * @return Packed BFP2 tile payload words.
 */
template <typename T>
std::vector<uint32_t> pack_as_bfp2_tiles(
    tt::stl::Span<const T> data,
    bool row_major_input,
    bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

/**
 * @brief Unpack BFP2 tiles into a float vector.
 *
 * @param bfp_tiles Packed BFP2 tile payload words.
 * @param row_major_output True to produce row-major output; false for tile-major.
 * @param is_exp_a Exponent sharing mode that matches the original packed data.
 * @param tile Optional tile shape descriptor; uses the default tile when nullopt.
 * @return Decoded values as float.
 */
std::vector<float> unpack_bfp2_tiles_into_float_vec(
    tt::stl::Span<const uint32_t> bfp_tiles,
    bool row_major_output,
    bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

namespace tt::tt_metal {

template <typename T>
std::vector<uint32_t> pack_as_bfp2_tiles(
    tt::stl::Span<const T> data,
    bool row_major_input,
    bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt) {
    return ::pack_as_bfp2_tiles(data, row_major_input, is_exp_a, tile);
}

inline std::vector<float> unpack_bfp2_tiles_into_float_vec(
    tt::stl::Span<const uint32_t> bfp_tiles,
    bool row_major_output,
    bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt) {
    return ::unpack_bfp2_tiles_into_float_vec(bfp_tiles, row_major_output, is_exp_a, tile);
}

}  // namespace tt::tt_metal
