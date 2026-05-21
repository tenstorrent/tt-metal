// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/mx_common.hpp>
#include <tt-metalium/tile.hpp>
#include <tt_stl/span.hpp>

#include <optional>
#include <vector>

namespace tt::tt_metal::mx {

// MXFP8 E5M2 (a.k.a. MXFP8R): 1 sign / 5 exp / 2 mantissa with IEEE-style Inf/NaN.
// Max normal = (1 + 3/4) * 2^15 = 57344. OCP MX block of 32 with E8M0 scale.
inline constexpr FormatParams kMxFp8E5M2Params = {
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
    .inf_rep = InfNanRepresentation::ExpAllOnesManZero,
    .nan_rep = InfNanRepresentation::ExpAllOnesManNonZero,
};

// MXFP8 E4M3FN (a.k.a. MXFP8P): 1 sign / 4 exp / 3 mantissa, finite-only — no
// Inf and NaN only at S.1111.111. Max normal = (1 + 6/8) * 2^8 = 448. OCP MX
// block of 32 with E8M0 scale.
inline constexpr FormatParams kMxFp8E4M3Params = {
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
    .inf_rep = InfNanRepresentation::NotRepresentable,
    .nan_rep = InfNanRepresentation::ExpAllOnesManAllOnes,
};

}  // namespace tt::tt_metal::mx

template <typename T>
inline std::vector<uint32_t> pack_as_mxfp8_e5m2_tiles(
    tt::stl::Span<const T> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile = std::nullopt) {
    return tt::tt_metal::mx::pack_as_mx_tiles_impl(data, row_major_input, tile, tt::tt_metal::mx::kMxFp8E5M2Params);
}

template <typename T>
inline std::vector<uint32_t> pack_as_mxfp8_e4m3_tiles(
    tt::stl::Span<const T> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile = std::nullopt) {
    return tt::tt_metal::mx::pack_as_mx_tiles_impl(data, row_major_input, tile, tt::tt_metal::mx::kMxFp8E4M3Params);
}

std::vector<float> unpack_mxfp8_e5m2_tiles_into_float_vec(
    tt::stl::Span<const uint32_t> mxfp8_tiles,
    bool row_major_output,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

std::vector<float> unpack_mxfp8_e4m3_tiles_into_float_vec(
    tt::stl::Span<const uint32_t> mxfp8_tiles,
    bool row_major_output,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);
