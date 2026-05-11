// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/mxfp8.hpp>
#include <tt-metalium/bfloat16.hpp>

#include <bit>
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>

#include "constants.hpp"
#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include "mx_common.hpp"
#include "tile.hpp"
#include "tracy/Tracy.hpp"

namespace {

// 2^n for integer n, avoiding the libm overhead of std::ldexp on the hot
// per-element unpack path. Fast path bit-constructs a normal float for
// n in [-126, 127]; the rare edges (subnormal scale, E8M0 NaN scale) defer
// to std::ldexp so behavior matches at boundaries.
inline float pow2_f32(int k) {
    if (k >= -126 && k <= 127) {
        return std::bit_cast<float>(static_cast<uint32_t>(127 + k) << 23);
    }
    return std::ldexp(1.0f, k);
}

// MXFP8 E5M2 (a.k.a. MXFP8R): 1 sign / 5 exp / 2 mantissa with IEEE-style Inf/NaN.
// Max normal = (1 + 3/4) * 2^15 = 57344. OCP MX block of 32 with E8M0 scale.
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

// MXFP8 E4M3FN (a.k.a. MXFP8P): 1 sign / 4 exp / 3 mantissa, finite-only — no
// Inf and NaN only at S.1111.111. Max normal = (1 + 6/8) * 2^8 = 448. OCP MX
// block of 32 with E8M0 scale.
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

template <typename T>
std::vector<uint32_t> pack_as_mxfp8_tiles_impl(
    tt::stl::Span<const T> data,
    bool row_major_input,
    const std::optional<tt::tt_metal::Tile>& tile,
    const tt::tt_metal::mx::FormatParams& params) {
    auto tile_H = tile.has_value() ? tile->get_tile_shape()[0] : tt::constants::TILE_HEIGHT;
    auto tile_W = tile.has_value() ? tile->get_tile_shape()[1] : tt::constants::TILE_WIDTH;
    auto face_H = tile.has_value() ? tile->get_face_shape()[0] : tt::constants::FACE_HEIGHT;
    auto face_W = tile.has_value() ? tile->get_face_shape()[1] : tt::constants::FACE_WIDTH;
    auto tile_HW = tile_H * tile_W;
    auto subtiles_in_tile_row = tile_H / face_H;
    auto subtiles_in_tile_col = tile_W / face_W;

    TT_ASSERT(tile_HW % params.block_size == 0, "MXFP8 tile must be a multiple of {} elements", params.block_size);
    TT_ASSERT(data.size() % tile_HW == 0, "Input size must be a multiple of tile size");

    uint32_t l1_alignment = tt::tt_metal::MetalContext::instance().hal().get_alignment(tt::tt_metal::HalMemType::L1);
    auto word_counts = tt::tt_metal::mx::compute_tile_word_counts(tile_HW, l1_alignment, params);
    uint32_t exp_count = word_counts.exp_count;
    uint32_t exp_words = word_counts.exp_words;
    uint32_t elem_words = word_counts.elem_words;

    uint32_t num_tiles = data.size() / tile_HW;
    std::vector<uint32_t> packed;
    packed.reserve(num_tiles * (exp_words + elem_words));

    size_t linear_index = 0;

    std::vector<float> tile_values;
    tile_values.reserve(tile_HW);
    std::vector<uint8_t> exps;
    exps.reserve(exp_count);
    std::vector<uint8_t> elems;
    elems.reserve(tile_HW);

    for (uint32_t tile_index = 0; tile_index < num_tiles; ++tile_index) {
        tile_values.clear();
        exps.clear();
        elems.clear();

        if (row_major_input) {
            for (uint32_t tr = 0; tr < subtiles_in_tile_row; ++tr) {
                for (uint32_t tc = 0; tc < subtiles_in_tile_col; ++tc) {
                    for (uint32_t i = 0; i < face_H; ++i) {
                        uint32_t row = tr * face_H + i;
                        for (uint32_t j = 0; j < face_W; ++j) {
                            uint32_t col = tc * face_W + j;
                            size_t data_index = (row * tile_W) + col + (tile_index * tile_HW);
                            tile_values.push_back(static_cast<float>(data[data_index]));
                        }
                    }
                }
            }
        } else {
            for (uint32_t i = 0; i < tile_HW; ++i) {
                tile_values.push_back(static_cast<float>(data[linear_index++]));
            }
        }

        for (uint32_t blk_idx = 0; blk_idx < exp_count; ++blk_idx) {
            // TODO: once we start testing stochastic rounding for exponents,
            // add a mechanism that passes exp_rnd_en to compute_block_scale.
            auto block_scale = tt::tt_metal::mx::compute_block_scale(
                tt::stl::Span<const float>(tile_values.data(), tile_values.size()),
                blk_idx * params.block_size,
                params);
            exps.push_back(block_scale.shared_exp_biased);

            int scale_exp = block_scale.shared_exp_adj;
            uint32_t base = blk_idx * params.block_size;
            const float scale_pack = pow2_f32(-scale_exp);
            for (uint32_t i = 0; i < params.block_size; ++i) {
                float v = tile_values[base + i];
                float scaled = v * scale_pack;
                elems.push_back(static_cast<uint8_t>(tt::tt_metal::mx::convert_to_mx_elem_bits(scaled, params)));
            }
        }

        tt::tt_metal::mx::pack_exp_words(exps, exp_words, packed);
        tt::tt_metal::mx::pack_elem_words(elems, elem_words, params, packed);
    }

    return packed;
}

std::vector<float> unpack_mxfp8_tiles_into_float_vec_impl(
    tt::stl::Span<const uint32_t> mxfp8_tiles,
    bool row_major_output,
    const std::optional<tt::tt_metal::Tile>& tile,
    const tt::tt_metal::mx::FormatParams& params) {
    auto tile_H = tile.has_value() ? tile->get_tile_shape()[0] : tt::constants::TILE_HEIGHT;
    auto tile_W = tile.has_value() ? tile->get_tile_shape()[1] : tt::constants::TILE_WIDTH;
    auto face_H = tile.has_value() ? tile->get_face_shape()[0] : tt::constants::FACE_HEIGHT;
    auto face_W = tile.has_value() ? tile->get_face_shape()[1] : tt::constants::FACE_WIDTH;
    auto tile_HW = tile_H * tile_W;
    auto subtiles_in_tile_row = tile_H / face_H;
    auto subtiles_in_tile_col = tile_W / face_W;

    TT_ASSERT(tile_HW % params.block_size == 0, "MXFP8 tile must be a multiple of {} elements", params.block_size);

    uint32_t l1_alignment = tt::tt_metal::MetalContext::instance().hal().get_alignment(tt::tt_metal::HalMemType::L1);
    auto word_counts = tt::tt_metal::mx::compute_tile_word_counts(tile_HW, l1_alignment, params);
    uint32_t exp_count = word_counts.exp_count;
    uint32_t exp_words = word_counts.exp_words;
    uint32_t elem_words = word_counts.elem_words;

    uint32_t tile_size_words = exp_words + elem_words;
    TT_ASSERT(mxfp8_tiles.size() % tile_size_words == 0, "Input size must be a multiple of MXFP8 tile size");
    uint32_t num_tiles = mxfp8_tiles.size() / tile_size_words;

    std::vector<float> output;
    output.resize(num_tiles * tile_HW);
    size_t linear_index = 0;

    size_t word_offset = 0;

    std::vector<uint8_t> exps;
    exps.reserve(exp_count);
    std::vector<uint8_t> elems;
    elems.reserve(tile_HW);
    std::vector<float> tile_values;
    tile_values.resize(tile_HW);

    for (uint32_t tile_index = 0; tile_index < num_tiles; ++tile_index) {
        tt::tt_metal::mx::unpack_exp_words(mxfp8_tiles, word_offset, exp_words, exp_count, exps);
        tt::tt_metal::mx::unpack_elem_words(mxfp8_tiles, word_offset, elem_words, tile_HW, params, elems);

        for (uint32_t blk = 0; blk < exp_count; ++blk) {
            uint8_t scale_exp_biased = exps[blk];
            int scale_exp_unbiased = static_cast<int>(scale_exp_biased) - params.scale_bias;
            const float scale_unpack = pow2_f32(scale_exp_unbiased);
            uint32_t base = blk * params.block_size;
            for (uint32_t j = 0; j < params.block_size; ++j) {
                uint32_t i = base + j;
                float elem_pre_scale = tt::tt_metal::mx::convert_from_mx_elem_bits(elems[i], scale_exp_biased, params);
                tile_values[i] = elem_pre_scale * scale_unpack;
            }
        }

        if (row_major_output) {
            size_t tile_base = tile_index * tile_HW;
            size_t idx = 0;
            for (uint32_t tr = 0; tr < subtiles_in_tile_row; ++tr) {
                for (uint32_t tc = 0; tc < subtiles_in_tile_col; ++tc) {
                    for (uint32_t i = 0; i < face_H; ++i) {
                        uint32_t row = tr * face_H + i;
                        for (uint32_t j = 0; j < face_W; ++j) {
                            uint32_t col = tc * face_W + j;
                            output[tile_base + row * tile_W + col] = tile_values[idx++];
                        }
                    }
                }
            }
        } else {
            for (float v : tile_values) {
                output[linear_index++] = v;
            }
        }
    }

    return output;
}

}  // namespace

template <typename T>
std::vector<uint32_t> pack_as_mxfp8_e5m2_tiles(
    tt::stl::Span<const T> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile) {
    ZoneScoped;
    return pack_as_mxfp8_tiles_impl(data, row_major_input, tile, kMxFp8E5M2Params);
}

template std::vector<uint32_t> pack_as_mxfp8_e5m2_tiles<bfloat16>(
    tt::stl::Span<const bfloat16> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile);

template std::vector<uint32_t> pack_as_mxfp8_e5m2_tiles<float>(
    tt::stl::Span<const float> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile);

template std::vector<uint32_t> pack_as_mxfp8_e5m2_tiles<int32_t>(
    tt::stl::Span<const int32_t> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile);

template std::vector<uint32_t> pack_as_mxfp8_e5m2_tiles<uint32_t>(
    tt::stl::Span<const uint32_t> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile);

template std::vector<uint32_t> pack_as_mxfp8_e5m2_tiles<uint8_t>(
    tt::stl::Span<const uint8_t> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile);

template std::vector<uint32_t> pack_as_mxfp8_e5m2_tiles<uint16_t>(
    tt::stl::Span<const uint16_t> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile);

template <typename T>
std::vector<uint32_t> pack_as_mxfp8_e4m3_tiles(
    tt::stl::Span<const T> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile) {
    ZoneScoped;
    return pack_as_mxfp8_tiles_impl(data, row_major_input, tile, kMxFp8E4M3Params);
}

template std::vector<uint32_t> pack_as_mxfp8_e4m3_tiles<bfloat16>(
    tt::stl::Span<const bfloat16> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile);

template std::vector<uint32_t> pack_as_mxfp8_e4m3_tiles<float>(
    tt::stl::Span<const float> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile);

template std::vector<uint32_t> pack_as_mxfp8_e4m3_tiles<int32_t>(
    tt::stl::Span<const int32_t> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile);

template std::vector<uint32_t> pack_as_mxfp8_e4m3_tiles<uint32_t>(
    tt::stl::Span<const uint32_t> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile);

template std::vector<uint32_t> pack_as_mxfp8_e4m3_tiles<uint8_t>(
    tt::stl::Span<const uint8_t> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile);

template std::vector<uint32_t> pack_as_mxfp8_e4m3_tiles<uint16_t>(
    tt::stl::Span<const uint16_t> data, bool row_major_input, const std::optional<tt::tt_metal::Tile>& tile);

std::vector<float> unpack_mxfp8_e5m2_tiles_into_float_vec(
    tt::stl::Span<const uint32_t> mxfp8_tiles, bool row_major_output, const std::optional<tt::tt_metal::Tile>& tile) {
    ZoneScoped;
    return unpack_mxfp8_tiles_into_float_vec_impl(mxfp8_tiles, row_major_output, tile, kMxFp8E5M2Params);
}

std::vector<float> unpack_mxfp8_e4m3_tiles_into_float_vec(
    tt::stl::Span<const uint32_t> mxfp8_tiles, bool row_major_output, const std::optional<tt::tt_metal::Tile>& tile) {
    ZoneScoped;
    return unpack_mxfp8_tiles_into_float_vec_impl(mxfp8_tiles, row_major_output, tile, kMxFp8E4M3Params);
}
