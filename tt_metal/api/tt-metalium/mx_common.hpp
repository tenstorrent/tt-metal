// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/tile.hpp>

namespace tt::tt_metal::mx {

enum class InfNanRepresentation : uint8_t {
    NotRepresentable = 0,
    ExpAllOnesManZero,
    ExpAllOnesManNonZero,
    ExpAllOnesManAllOnes,
};

struct FormatParams {
    uint32_t block_size = 32;
    int scale_bias = 0x7F;
    int elem_exp_bits = 0;
    int elem_man_bits = 0;
    int elem_exp_bias = 0;
    int elem_exp_max_unbiased = 0;
    int elem_exp_min_unbiased = 0;
    int elem_exp_subnorm_encoding = 0;
    uint32_t elem_man_max = 0;
    int elem_width_bits = 0;
    int elem_width_storage_bits = 0;
    bool sat_supported = false;
    uint32_t elem_sat_pos_bits = 0;
    uint32_t elem_sat_neg_bits = 0;
    InfNanRepresentation inf_rep = InfNanRepresentation::NotRepresentable;
    InfNanRepresentation nan_rep = InfNanRepresentation::NotRepresentable;
};

struct RoundResult {
    uint32_t mantissa = 0;
    uint32_t overflow = 0;
};

struct BlockScaleResult {
    uint8_t shared_exp_biased = 0;
    // Unbiased shared exponent. Effective scale is 2^shared_exp_adj — kept as
    // an int so callers can ldexp directly and avoid building/dividing by a
    // float power of two.
    int shared_exp_adj = 0;
};

struct TileWordCounts {
    uint32_t exp_count = 0;
    uint32_t exp_bytes = 0;
    uint32_t exp_words = 0;
    uint32_t elem_words = 0;
};

RoundResult round_ties_even(uint32_t input_mantissa, int output_width, int input_width = 23);

BlockScaleResult compute_block_scale(
    tt::stl::Span<const float> values, size_t block_offset, const FormatParams& params, bool exp_rnd_en = false);

uint32_t compute_exp_count(uint32_t elem_count, const FormatParams& params);
uint32_t compute_exp_bytes(uint32_t exp_count, uint32_t l1_alignment);
uint32_t compute_elem_words(uint32_t elem_count, const FormatParams& params);
TileWordCounts compute_tile_word_counts(uint32_t elem_count, const FormatParams& params);

uint32_t convert_to_mx_elem_bits(float datum, const FormatParams& params);
float convert_from_mx_elem_bits(uint32_t elem_bits, uint8_t scale_exp_biased, const FormatParams& params);

void pack_exp_words(const std::vector<uint8_t>& exps, uint32_t exp_words, std::vector<uint32_t>& out);
void pack_elem_words(
    const std::vector<uint8_t>& elems, uint32_t elem_words, const FormatParams& params, std::vector<uint32_t>& out);

void unpack_exp_words(
    tt::stl::Span<const uint32_t> words,
    size_t& word_offset,
    uint32_t exp_words,
    uint32_t exp_count,
    std::vector<uint8_t>& out);
void unpack_elem_words(
    tt::stl::Span<const uint32_t> words,
    size_t& word_offset,
    uint32_t elem_words,
    uint32_t elem_count,
    const FormatParams& params,
    std::vector<uint8_t>& out);

// 2^k for integer k, avoiding the libm overhead of std::ldexp on the hot
// per-element MX pack/unpack paths. Fast path bit-constructs a normal float
// for k in [-126, 127]; the rare edges (subnormal scale, E8M0 NaN scale)
// defer to std::ldexp so behavior matches at boundaries.
inline float pow2_f32(int k) {
    if (k >= -126 && k <= 127) {
        return __builtin_bit_cast(float, static_cast<uint32_t>(127 + k) << 23);
    }
    return std::ldexp(1.0f, k);
}

// Generic MX tile packer. Inline so per-format wrappers (pack_as_mxfp4_tiles,
// pack_as_mxfp8_*_tiles, ...) can specialize on element type T without
// instantiation boilerplate, and so call sites passing a constexpr FormatParams
// can constant-propagate field accesses through the inner loops.
template <typename T>
std::vector<uint32_t> pack_as_mx_tiles_impl(
    tt::stl::Span<const T> data,
    bool row_major_input,
    const std::optional<tt::tt_metal::Tile>& tile,
    const FormatParams& params) {
    auto tile_H = tile.has_value() ? tile->get_tile_shape()[0] : tt::constants::TILE_HEIGHT;
    auto tile_W = tile.has_value() ? tile->get_tile_shape()[1] : tt::constants::TILE_WIDTH;
    auto face_H = tile.has_value() ? tile->get_face_shape()[0] : tt::constants::FACE_HEIGHT;
    auto face_W = tile.has_value() ? tile->get_face_shape()[1] : tt::constants::FACE_WIDTH;
    auto tile_HW = tile_H * tile_W;
    auto subtiles_in_tile_row = tile_H / face_H;
    auto subtiles_in_tile_col = tile_W / face_W;

    TT_ASSERT(tile_HW % params.block_size == 0, "MX tile must be a multiple of {} elements", params.block_size);
    TT_ASSERT(data.size() % tile_HW == 0, "Input size must be a multiple of tile size");

    auto word_counts = compute_tile_word_counts(tile_HW, params);
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
            auto block_scale = compute_block_scale(
                tt::stl::Span<const float>(tile_values.data(), tile_values.size()),
                blk_idx * params.block_size,
                params);
            exps.push_back(block_scale.shared_exp_biased);

            int scale_exp = block_scale.shared_exp_adj;
            const float scale_pack = pow2_f32(-scale_exp);
            uint32_t base = blk_idx * params.block_size;
            for (uint32_t i = 0; i < params.block_size; ++i) {
                float v = tile_values[base + i];
                float scaled = v * scale_pack;
                elems.push_back(static_cast<uint8_t>(convert_to_mx_elem_bits(scaled, params)));
            }
        }

        pack_exp_words(exps, exp_words, packed);
        pack_elem_words(elems, elem_words, params, packed);
    }

    return packed;
}

}  // namespace tt::tt_metal::mx
