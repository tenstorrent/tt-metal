// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <tt-metalium/mxfp4.hpp>
#include <tt-metalium/mxfp6.hpp>
#include <tt-metalium/mxfp8.hpp>
#include <tt-metalium/mxint.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>
#include <tt-metalium/tile.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>

namespace tt::tt_metal::tensor_impl {

// Pack float data into MX tiles for the given MX DataType. The MX packers only take float input.
inline std::vector<uint32_t> pack_as_mx_tiles(
    DataType dtype, ttsl::Span<const float> data, bool row_major_input, const Tile& tile) {
    switch (dtype) {
        case DataType::MXFP8_E4M3: return pack_as_mxfp8_e4m3_tiles(data, row_major_input, tile);
        case DataType::MXFP8_E5M2: return pack_as_mxfp8_e5m2_tiles(data, row_major_input, tile);
        case DataType::MXFP6_E2M3: return pack_as_mxfp6p_tiles(data, row_major_input, tile);
        case DataType::MXFP6_E3M2: return pack_as_mxfp6r_tiles(data, row_major_input, tile);
        case DataType::MXFP4: return pack_as_mxfp4_tiles(data, row_major_input, tile);
        case DataType::MXINT8: return pack_as_mxint8_tiles(data, row_major_input, tile);
        case DataType::MXINT4: return pack_as_mxint4_tiles(data, row_major_input, tile);
        case DataType::MXINT2: return pack_as_mxint2_tiles(data, row_major_input, tile);
        default: TT_THROW("pack_as_mx_tiles: {} is not an MX data type", dtype);
    }
}

// Unpack MX tiles of the given MX DataType into float.
inline std::vector<float> unpack_mx_tiles_into_float_vec(
    DataType dtype, ttsl::Span<const uint32_t> packed, bool row_major_output, const Tile& tile) {
    switch (dtype) {
        case DataType::MXFP8_E4M3: return unpack_mxfp8_e4m3_tiles_into_float_vec(packed, row_major_output, tile);
        case DataType::MXFP8_E5M2: return unpack_mxfp8_e5m2_tiles_into_float_vec(packed, row_major_output, tile);
        case DataType::MXFP6_E2M3: return unpack_mxfp6p_tiles_into_float_vec(packed, row_major_output, tile);
        case DataType::MXFP6_E3M2: return unpack_mxfp6r_tiles_into_float_vec(packed, row_major_output, tile);
        case DataType::MXFP4: return unpack_mxfp4_tiles_into_float_vec(packed, row_major_output, tile);
        case DataType::MXINT8: return unpack_mxint8_tiles_into_float_vec(packed, row_major_output, tile);
        case DataType::MXINT4: return unpack_mxint4_tiles_into_float_vec(packed, row_major_output, tile);
        case DataType::MXINT2: return unpack_mxint2_tiles_into_float_vec(packed, row_major_output, tile);
        default: TT_THROW("unpack_mx_tiles_into_float_vec: {} is not an MX data type", dtype);
    }
}

}  // namespace tt::tt_metal::tensor_impl
