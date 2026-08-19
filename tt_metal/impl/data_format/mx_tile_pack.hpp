// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bit>
#include <cstdint>
#include <optional>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/tile.hpp>

#include "mx_common.hpp"

namespace tt::tt_metal::mx {

// Per-block statistics gathered while we cast the input to float, so the
// shared exponent can be derived without a second pass over the values.
struct BlockStats {
    uint32_t max_abs_bits = 0;
    bool all_nan = true;
    bool any_inf = false;
    bool all_inf_or_zero = true;
};

inline void update_block_stats(BlockStats& s, float f) {
    uint32_t bits = std::bit_cast<uint32_t>(f);
    uint32_t abs_bits = bits & 0x7FFFFFFFu;
    uint32_t exp_field = abs_bits >> 23;
    uint32_t mant_field = abs_bits & 0x7FFFFFu;
    bool is_nan = (exp_field == 0xFFu) && (mant_field != 0);
    bool is_inf = (exp_field == 0xFFu) && (mant_field == 0);
    bool is_zero = (abs_bits == 0u);
    s.all_nan = s.all_nan && is_nan;
    s.any_inf = s.any_inf || is_inf;
    s.all_inf_or_zero = s.all_inf_or_zero && (is_inf || is_nan || is_zero);
    if (!is_nan && !is_inf && abs_bits > s.max_abs_bits) {
        s.max_abs_bits = abs_bits;
    }
}

// Quantize one (already block-scaled) value to its packed element bits, choosing
// the integer (MxInt2/4/8) or floating-point (MXFP*) encoding from FormatParams.
// `params` is a constexpr at every call site, so the branch folds away after
// inlining.
inline uint32_t mx_encode_elem(float scaled, const FormatParams& params) {
    return params.is_integer ? convert_to_mxint_elem_bits(scaled, params) : convert_to_mxfp_elem_bits(scaled, params);
}

// Decode one packed element field back to its pre-block-scale float value, again
// dispatching on the integer vs floating-point encoding. The integer decoder
// needs no scale byte: its 0xFF (NaN-scale) block is zeroed by the caller.
inline float mx_decode_elem(uint32_t elem_bits, uint8_t scale_exp_biased, const FormatParams& params) {
    return params.is_integer ? convert_from_mxint_elem_bits(elem_bits, params)
                             : convert_from_mxfp_elem_bits(elem_bits, scale_exp_biased, params);
}

// Generic MX tile packer. Works for any FormatParams whose
// `block_size * elem_width_storage_bits` is a multiple of 32 (whole-word
// per-block packing) and whose `elem_width_storage_bits` divides 32 evenly
// (MXFP4: 4-bit storage / 8 elems per word, MXFP6: 8-bit storage / 4 elems
// per word). Format-specific bit layout falls out of `FormatParams`.
//
// The block-scale derivation (E8M0), [scales | elements] tile layout, and
// per-block word packing are shared by every MX format. Only the per-element
// conversion differs: floating-point formats extract IEEE fields, integer
// formats (params.is_integer) quantize to a signed two's-complement integer.
// params is a constexpr at each call site (kMxFp4Params, kMxInt8Params, ...),
// so the is_integer branch constant-folds away per translation unit.
template <typename T>
std::vector<uint32_t> pack_as_mx_tiles_impl(
    ttsl::Span<const T> data,
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
    TT_ASSERT(params.block_size % 4 == 0, "MX block size must be a multiple of 4");
    TT_ASSERT(
        (params.block_size * params.elem_width_storage_bits) % 32 == 0,
        "MX block must pack into a whole number of 32-bit words");
    TT_ASSERT(
        params.elem_width_storage_bits > 0 && 32u % params.elem_width_storage_bits == 0,
        "MX elem storage width must divide 32 bits evenly");

    auto word_counts = compute_tile_word_counts(tile_HW, params);
    uint32_t exp_count = word_counts.exp_count;
    uint32_t exp_words = word_counts.exp_words;
    uint32_t elem_words = word_counts.elem_words;
    uint32_t tile_size_words = exp_words + elem_words;

    const uint32_t block_size = params.block_size;
    const uint32_t elem_storage_bits = params.elem_width_storage_bits;
    const uint32_t elements_per_word = 32u / elem_storage_bits;
    const uint32_t elem_words_per_block = (block_size * elem_storage_bits) / 32u;
    const uint32_t elem_shift = elem_storage_bits - params.elem_width_bits;
    const uint32_t elem_mask = (params.elem_width_bits >= 32) ? 0xFFFFFFFFu : ((1u << params.elem_width_bits) - 1u);

    uint32_t num_tiles = data.size() / tile_HW;
    // Zero-initialised so the alignment-padding tail of each tile's exp section
    // ends up as zeros, matching the legacy pack_exp_words behaviour.
    std::vector<uint32_t> packed(static_cast<size_t>(num_tiles) * tile_size_words);

    size_t linear_index = 0;
    std::vector<float> tile_values(tile_HW);
    std::vector<BlockStats> stats(exp_count);

    for (uint32_t tile_index = 0; tile_index < num_tiles; ++tile_index) {
        const size_t tile_base = static_cast<size_t>(tile_index) * tile_size_words;
        const size_t elem_base = tile_base + exp_words;

        for (uint32_t b = 0; b < exp_count; ++b) {
            stats[b] = BlockStats{};
        }

        // Cast input to float into tile_values in storage order, accumulating
        // per-block stats inline. This replaces the second pass that
        // compute_block_scale would otherwise make over the same floats.
        uint32_t cur_blk = 0;
        uint32_t in_blk = 0;
        if (row_major_input) {
            size_t stored_index = 0;
            for (uint32_t tr = 0; tr < subtiles_in_tile_row; ++tr) {
                for (uint32_t tc = 0; tc < subtiles_in_tile_col; ++tc) {
                    for (uint32_t i = 0; i < face_H; ++i) {
                        uint32_t row = tr * face_H + i;
                        for (uint32_t j = 0; j < face_W; ++j) {
                            uint32_t col = tc * face_W + j;
                            size_t data_index = (row * tile_W) + col + (tile_index * tile_HW);
                            float f = static_cast<float>(data[data_index]);
                            tile_values[stored_index++] = f;
                            update_block_stats(stats[cur_blk], f);
                            if (++in_blk == block_size) {
                                in_blk = 0;
                                ++cur_blk;
                            }
                        }
                    }
                }
            }
        } else {
            for (uint32_t i = 0; i < tile_HW; ++i) {
                float f = static_cast<float>(data[linear_index++]);
                tile_values[i] = f;
                update_block_stats(stats[cur_blk], f);
                if (++in_blk == block_size) {
                    in_blk = 0;
                    ++cur_blk;
                }
            }
        }

        for (uint32_t blk_idx = 0; blk_idx < exp_count; ++blk_idx) {
            // TODO: once we start testing stochastic rounding for exponents,
            // add a mechanism that passes exp_rnd_en to finalize_block_scale.
            const auto& s = stats[blk_idx];
            auto block_scale = finalize_block_scale(s.max_abs_bits, s.all_nan, s.all_inf_or_zero, s.any_inf, params);

            packed[tile_base + (blk_idx >> 2)] |= static_cast<uint32_t>(block_scale.shared_exp_biased)
                                                  << ((blk_idx & 0x3u) * 8);

            int scale_exp = block_scale.shared_exp_adj;
            const float scale_pack = pow2_f32(-scale_exp);
            uint32_t base = blk_idx * block_size;
            size_t blk_word_base = elem_base + static_cast<size_t>(blk_idx) * elem_words_per_block;

            for (uint32_t w = 0; w < elem_words_per_block; ++w) {
                uint32_t word = 0;
                for (uint32_t b = 0; b < elements_per_word; ++b) {
                    float v = tile_values[base + w * elements_per_word + b];
                    float scaled = v * scale_pack;
                    uint32_t bits = mx_encode_elem(scaled, params);
                    word |= (bits & elem_mask) << (elem_shift + b * elem_storage_bits);
                }
                packed[blk_word_base + w] = word;
            }
        }
    }

    return packed;
}

// Generic MX tile unpacker. Same FormatParams constraints as
// pack_as_mx_tiles_impl, and likewise serves both the floating-point and the
// integer (params.is_integer) MX formats. Inline rather than .cpp-out-of-line
// so call sites with a constexpr FormatParams (kMxFp4Params, kMxInt8Params, ...)
// can constant-fold the inner loops, the is_integer branch, and the
// `convert_from_mxfp_elem_bits` branches.
inline std::vector<float> unpack_mx_tiles_into_float_vec_impl(
    ttsl::Span<const uint32_t> mx_tiles,
    bool row_major_output,
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
    TT_ASSERT(params.block_size % 4 == 0, "MX block size must be a multiple of 4");
    TT_ASSERT(
        (params.block_size * params.elem_width_storage_bits) % 32 == 0,
        "MX block must pack into a whole number of 32-bit words");
    TT_ASSERT(
        params.elem_width_storage_bits > 0 && 32u % params.elem_width_storage_bits == 0,
        "MX elem storage width must divide 32 bits evenly");

    auto word_counts = compute_tile_word_counts(tile_HW, params);
    uint32_t exp_count = word_counts.exp_count;
    uint32_t exp_words = word_counts.exp_words;
    uint32_t elem_words = word_counts.elem_words;
    uint32_t tile_size_words = exp_words + elem_words;
    TT_ASSERT(mx_tiles.size() % tile_size_words == 0, "Input size must be a multiple of MX tile size");
    uint32_t num_tiles = mx_tiles.size() / tile_size_words;

    const uint32_t block_size = params.block_size;
    const uint32_t elem_storage_bits = params.elem_width_storage_bits;
    const uint32_t elements_per_word = 32u / elem_storage_bits;
    const uint32_t elem_words_per_block = (block_size * elem_storage_bits) / 32u;
    const uint32_t elem_shift = elem_storage_bits - params.elem_width_bits;
    const uint32_t elem_unit_mask = (elem_storage_bits >= 32) ? 0xFFFFFFFFu : ((1u << elem_storage_bits) - 1u);
    const uint32_t elem_mask = (params.elem_width_bits >= 32) ? 0xFFFFFFFFu : ((1u << params.elem_width_bits) - 1u);

    std::vector<float> output(static_cast<size_t>(num_tiles) * tile_HW);

    // Scratch buffer is only needed when we have to scatter from storage order
    // (block-by-block) into row-major output. In the linear-output case we
    // decode straight into `output`, saving an extra full-tile write pass.
    std::vector<float> tile_scratch;
    if (row_major_output) {
        tile_scratch.resize(tile_HW);
    }

    for (uint32_t tile_index = 0; tile_index < num_tiles; ++tile_index) {
        const size_t tile_base = static_cast<size_t>(tile_index) * tile_size_words;
        const size_t elem_base = tile_base + exp_words;

        float* tile_target =
            row_major_output ? tile_scratch.data() : (output.data() + static_cast<size_t>(tile_index) * tile_HW);

        for (uint32_t blk = 0; blk < exp_count; ++blk) {
            uint8_t scale_exp_biased =
                static_cast<uint8_t>((mx_tiles[tile_base + (blk >> 2)] >> ((blk & 0x3u) * 8)) & 0xFFu);
            uint32_t base = blk * block_size;
            size_t blk_word_base = elem_base + static_cast<size_t>(blk) * elem_words_per_block;

            // OCP MX rule: a NaN block scale (0xFF) zeros the whole block. The
            // integer formats have no NaN element to carry it, so handle it here
            // and avoid the 0 * 2^(0xFF-bias) = 0 * inf = NaN trap of the generic
            // multiply below. The floating-point path is left untouched: it
            // relies on convert_from_mxfp_elem_bits returning NaN for a 0xFF
            // scale (then NaN * inf = NaN), preserving its existing behaviour.
            if (params.is_integer && scale_exp_biased == 0xFFu) {
                for (uint32_t i = 0; i < block_size; ++i) {
                    tile_target[base + i] = 0.0f;
                }
                continue;
            }

            int scale_exp_unbiased = static_cast<int>(scale_exp_biased) - params.scale_bias;
            const float scale_unpack = pow2_f32(scale_exp_unbiased);

            for (uint32_t w = 0; w < elem_words_per_block; ++w) {
                uint32_t word = mx_tiles[blk_word_base + w];
                for (uint32_t b = 0; b < elements_per_word; ++b) {
                    uint32_t raw_unit = (word >> (b * elem_storage_bits)) & elem_unit_mask;
                    uint32_t elem_bits = (raw_unit >> elem_shift) & elem_mask;
                    float elem_pre_scale = mx_decode_elem(elem_bits, scale_exp_biased, params);
                    tile_target[base + w * elements_per_word + b] = elem_pre_scale * scale_unpack;
                }
            }
        }

        if (row_major_output) {
            size_t tile_out_base = static_cast<size_t>(tile_index) * tile_HW;
            size_t idx = 0;
            for (uint32_t tr = 0; tr < subtiles_in_tile_row; ++tr) {
                for (uint32_t tc = 0; tc < subtiles_in_tile_col; ++tc) {
                    for (uint32_t i = 0; i < face_H; ++i) {
                        uint32_t row = tr * face_H + i;
                        for (uint32_t j = 0; j < face_W; ++j) {
                            uint32_t col = tc * face_W + j;
                            output[tile_out_base + row * tile_W + col] = tile_scratch[idx++];
                        }
                    }
                }
            }
        }
    }

    return output;
}

}  // namespace tt::tt_metal::mx
