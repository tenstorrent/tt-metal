// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/mxint.hpp>

#include <optional>
#include <vector>

#include <tt_stl/span.hpp>

#include <tt-metalium/tile.hpp>
#include "tt_metal/tools/profiler/tracy_debug_zones.hpp"

#include "mx_tile_pack.hpp"

namespace {

// MxInt8 = S1.6: signed two's-complement int8 element with an implicit 2^-6
// scale (divisor 64). 8 bits stored one-per-byte. Range: ±127/64 ≈ ±1.984
// (symmetric — the −128 encoding 0x80 is left unused per OCP symmetry guidance).
// 32-element OCP MX block with an E8M0 shared scale. Internal descriptor; not
// part of the public API.
constexpr tt::tt_metal::mx::FormatParams kMxInt8Params = {
    .block_size = 32,
    .scale_bias = 0x7F,
    .elem_exp_max_unbiased = 0,  // post-block-scale values land in [1, 2)
    .elem_width_bits = 8,
    .elem_width_storage_bits = 8,
    .is_integer = true,
    .elem_int_scale = 64,
    .elem_int_max = 127,
};

// MxInt4 = S1.2: signed two's-complement int4 element (2 packed per byte, low
// nibble = even index) with an implicit 2^-2 scale (divisor 4). Range: ±7/4 =
// ±1.75 (symmetric — the −8 encoding 0b1000 is left unused).
constexpr tt::tt_metal::mx::FormatParams kMxInt4Params = {
    .block_size = 32,
    .scale_bias = 0x7F,
    .elem_exp_max_unbiased = 0,
    .elem_width_bits = 4,
    .elem_width_storage_bits = 4,
    .is_integer = true,
    .elem_int_scale = 4,
    .elem_int_max = 7,
};

// MxInt2 = S1.0: signed two's-complement int2 element (4 packed per byte) with
// an implicit 2^0 scale (divisor 1). Only −1/0/+1 are representable (the −2
// encoding 0b10 is left unused for symmetry).
constexpr tt::tt_metal::mx::FormatParams kMxInt2Params = {
    .block_size = 32,
    .scale_bias = 0x7F,
    .elem_exp_max_unbiased = 0,
    .elem_width_bits = 2,
    .elem_width_storage_bits = 2,
    .is_integer = true,
    .elem_int_scale = 1,
    .elem_int_max = 1,
};

}  // namespace

template <typename T>
std::vector<uint32_t> pack_as_mxint8_tiles(
    ttsl::Span<const T> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile) {
    TTZoneScopedD(DATA_FORMAT);
    return tt::tt_metal::mx::pack_as_mx_tiles_impl(data, row_major_input, tile, kMxInt8Params);
}

template <typename T>
std::vector<uint32_t> pack_as_mxint4_tiles(
    ttsl::Span<const T> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile) {
    TTZoneScopedD(DATA_FORMAT);
    return tt::tt_metal::mx::pack_as_mx_tiles_impl(data, row_major_input, tile, kMxInt4Params);
}

template <typename T>
std::vector<uint32_t> pack_as_mxint2_tiles(
    ttsl::Span<const T> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile) {
    TTZoneScopedD(DATA_FORMAT);
    return tt::tt_metal::mx::pack_as_mx_tiles_impl(data, row_major_input, tile, kMxInt2Params);
}

// Explicit instantiations — keep in sync with the supported input element types.
template std::vector<uint32_t> pack_as_mxint8_tiles<float>(
    ttsl::Span<const float>, bool, const std::optional<tt::tt_metal::Tile>&);
template std::vector<uint32_t> pack_as_mxint4_tiles<float>(
    ttsl::Span<const float>, bool, const std::optional<tt::tt_metal::Tile>&);
template std::vector<uint32_t> pack_as_mxint2_tiles<float>(
    ttsl::Span<const float>, bool, const std::optional<tt::tt_metal::Tile>&);

std::vector<float> unpack_mxint8_tiles_into_float_vec(
    ttsl::Span<const uint32_t> mxint_tiles, bool row_major_output, const std::optional<tt::tt_metal::Tile>& tile) {
    TTZoneScopedD(DATA_FORMAT);
    return tt::tt_metal::mx::unpack_mx_tiles_into_float_vec_impl(mxint_tiles, row_major_output, tile, kMxInt8Params);
}

std::vector<float> unpack_mxint4_tiles_into_float_vec(
    ttsl::Span<const uint32_t> mxint_tiles, bool row_major_output, const std::optional<tt::tt_metal::Tile>& tile) {
    TTZoneScopedD(DATA_FORMAT);
    return tt::tt_metal::mx::unpack_mx_tiles_into_float_vec_impl(mxint_tiles, row_major_output, tile, kMxInt4Params);
}

std::vector<float> unpack_mxint2_tiles_into_float_vec(
    ttsl::Span<const uint32_t> mxint_tiles, bool row_major_output, const std::optional<tt::tt_metal::Tile>& tile) {
    TTZoneScopedD(DATA_FORMAT);
    return tt::tt_metal::mx::unpack_mx_tiles_into_float_vec_impl(mxint_tiles, row_major_output, tile, kMxInt2Params);
}
