// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/mxfp4.hpp>
#include <tt-metalium/mxfp6.hpp>
#include <tt-metalium/mxfp8.hpp>
#include <tt-metalium/mxint.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>

namespace tt::test_utils {

// Pack a flat float buffer into MX tiles of the requested format.
inline std::vector<uint32_t> pack_as_mx_tiles(
    tt::DataFormat fmt,
    tt::stl::Span<const float> floats,
    bool row_major_input,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt) {
    switch (fmt) {
        case tt::DataFormat::MxFp4: return tt::tt_metal::pack_as_mxfp4_tiles(floats, row_major_input, tile);
        case tt::DataFormat::MxFp6R: return tt::tt_metal::pack_as_mxfp6r_tiles(floats, row_major_input, tile);
        case tt::DataFormat::MxFp6P: return tt::tt_metal::pack_as_mxfp6p_tiles(floats, row_major_input, tile);
        case tt::DataFormat::MxFp8R: return tt::tt_metal::pack_as_mxfp8_e5m2_tiles(floats, row_major_input, tile);
        case tt::DataFormat::MxFp8P: return tt::tt_metal::pack_as_mxfp8_e4m3_tiles(floats, row_major_input, tile);
        case tt::DataFormat::MxInt8: return tt::tt_metal::pack_as_mxint8_tiles(floats, row_major_input, tile);
        case tt::DataFormat::MxInt4: return tt::tt_metal::pack_as_mxint4_tiles(floats, row_major_input, tile);
        case tt::DataFormat::MxInt2: return tt::tt_metal::pack_as_mxint2_tiles(floats, row_major_input, tile);
        default: TT_THROW("pack_as_mx_tiles: not an MX DataFormat: {}", static_cast<int>(fmt));
    }
}

// Decode MX tiles of the requested format into a flat float buffer.
inline std::vector<float> mx_to_floats(
    tt::DataFormat fmt,
    tt::stl::Span<const uint32_t> packed,
    bool row_major_output = false,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt) {
    switch (fmt) {
        case tt::DataFormat::MxFp4:
            return tt::tt_metal::unpack_mxfp4_tiles_into_float_vec(packed, row_major_output, tile);
        case tt::DataFormat::MxFp6R:
            return tt::tt_metal::unpack_mxfp6r_tiles_into_float_vec(packed, row_major_output, tile);
        case tt::DataFormat::MxFp6P:
            return tt::tt_metal::unpack_mxfp6p_tiles_into_float_vec(packed, row_major_output, tile);
        case tt::DataFormat::MxFp8R:
            return tt::tt_metal::unpack_mxfp8_e5m2_tiles_into_float_vec(packed, row_major_output, tile);
        case tt::DataFormat::MxFp8P:
            return tt::tt_metal::unpack_mxfp8_e4m3_tiles_into_float_vec(packed, row_major_output, tile);
        case tt::DataFormat::MxInt8:
            return tt::tt_metal::unpack_mxint8_tiles_into_float_vec(packed, row_major_output, tile);
        case tt::DataFormat::MxInt4:
            return tt::tt_metal::unpack_mxint4_tiles_into_float_vec(packed, row_major_output, tile);
        case tt::DataFormat::MxInt2:
            return tt::tt_metal::unpack_mxint2_tiles_into_float_vec(packed, row_major_output, tile);
        default: TT_THROW("mx_to_floats: not an MX DataFormat: {}", static_cast<int>(fmt));
    }
}

// Convenience overload accepting a std::vector (the common test call shape).
inline std::vector<float> mx_to_floats(
    tt::DataFormat fmt,
    const std::vector<uint32_t>& packed,
    bool row_major_output = false,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt) {
    return mx_to_floats(fmt, tt::stl::make_const_span(packed), row_major_output, tile);
}

}  // namespace tt::test_utils
