// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/mxfp8.hpp>

#include <optional>
#include <vector>

#include <tt_stl/span.hpp>

#include <tt-metalium/tile.hpp>
#include "tt_metal/tools/profiler/tracy_debug_zones.hpp"

#include "mx_tile_pack.hpp"

namespace {

// MXFP8 E5M2 (a.k.a. MXFP8R) = S1E5M2 with IEEE-style Inf/NaN. 8 bits stored
// one-per-byte. Max normal = (1 + 3/4) * 2^15 = 57344. OCP MX block of 32 with
// E8M0 scale. Internal format descriptor; not part of the public API.
constexpr tt::tt_metal::mx::FormatParams kMxFp8E5M2Params = {
    .block_size = 32,
    .scale_bias = 0x7F,
    .elem_exp_bits = 5,
    .elem_man_bits = 2,
    .elem_exp_bias = 15,
    .elem_exp_max_unbiased = 15,
    .elem_exp_min_unbiased = -14,
    .elem_exp_subnorm_encoding = 0,
    .elem_man_max = 0x3,
    .elem_width_bits = 8,
    .elem_width_storage_bits = 8,
    .sat_supported = true,
    .elem_sat_pos_bits = 0x7B,  // 0_11110_11 = +max normal
    .elem_sat_neg_bits = 0xFB,  // 1_11110_11 = -max normal
    .inf_rep = tt::tt_metal::mx::InfNanRepresentation::ExpAllOnesManZero,
    .nan_rep = tt::tt_metal::mx::InfNanRepresentation::ExpAllOnesManNonZero,
};

// MXFP8 E4M3FN (a.k.a. MXFP8P) = S1E4M3, finite-only — no Inf and NaN only at
// S.1111.111. 8 bits stored one-per-byte. Max normal = (1 + 6/8) * 2^8 = 448.
// OCP MX block of 32 with E8M0 scale. Internal format descriptor; not part of
// the public API.
constexpr tt::tt_metal::mx::FormatParams kMxFp8E4M3Params = {
    .block_size = 32,
    .scale_bias = 0x7F,
    .elem_exp_bits = 4,
    .elem_man_bits = 3,
    .elem_exp_bias = 7,
    .elem_exp_max_unbiased = 8,
    .elem_exp_min_unbiased = -6,
    .elem_exp_subnorm_encoding = 0,
    .elem_man_max = 0x6,  // mant 0b111 at max exp is reserved for NaN
    .elem_width_bits = 8,
    .elem_width_storage_bits = 8,
    .sat_supported = true,
    .elem_sat_pos_bits = 0x7E,  // 0_1111_110 = +max normal
    .elem_sat_neg_bits = 0xFE,  // 1_1111_110 = -max normal
    .inf_rep = tt::tt_metal::mx::InfNanRepresentation::NotRepresentable,
    .nan_rep = tt::tt_metal::mx::InfNanRepresentation::ExpAllOnesManAllOnes,
};

}  // namespace

template <typename T>
std::vector<uint32_t> pack_as_mxfp8_e5m2_tiles(
    ttsl::Span<const T> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile) {
    TTZoneScopedD(DATA_FORMAT);
    return tt::tt_metal::mx::pack_as_mx_tiles_impl(data, row_major_input, tile, kMxFp8E5M2Params);
}

template <typename T>
std::vector<uint32_t> pack_as_mxfp8_e4m3_tiles(
    ttsl::Span<const T> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile) {
    TTZoneScopedD(DATA_FORMAT);
    return tt::tt_metal::mx::pack_as_mx_tiles_impl(data, row_major_input, tile, kMxFp8E4M3Params);
}

// Explicit instantiations — keep in sync with the supported input element types.
template std::vector<uint32_t> pack_as_mxfp8_e5m2_tiles<float>(
    ttsl::Span<const float>, bool, const std::optional<tt::tt_metal::Tile>&);
template std::vector<uint32_t> pack_as_mxfp8_e4m3_tiles<float>(
    ttsl::Span<const float>, bool, const std::optional<tt::tt_metal::Tile>&);

std::vector<float> unpack_mxfp8_e5m2_tiles_into_float_vec(
    ttsl::Span<const uint32_t> mxfp8_tiles, bool row_major_output, const std::optional<tt::tt_metal::Tile>& tile) {
    TTZoneScopedD(DATA_FORMAT);
    return tt::tt_metal::mx::unpack_mx_tiles_into_float_vec_impl(mxfp8_tiles, row_major_output, tile, kMxFp8E5M2Params);
}

std::vector<float> unpack_mxfp8_e4m3_tiles_into_float_vec(
    ttsl::Span<const uint32_t> mxfp8_tiles, bool row_major_output, const std::optional<tt::tt_metal::Tile>& tile) {
    TTZoneScopedD(DATA_FORMAT);
    return tt::tt_metal::mx::unpack_mx_tiles_into_float_vec_impl(mxfp8_tiles, row_major_output, tile, kMxFp8E4M3Params);
}
