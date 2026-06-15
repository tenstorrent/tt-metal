// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/tile.hpp>

#include "mx_common.hpp"
#include "mx_tile_pack.hpp"  // BlockStats / update_block_stats

namespace tt::tt_metal::mx {

// Packer is the precise reference the device output is compared against, so
// its rounding must be deterministic regardless of ambient FP state. This
// helper computes ties-to-even purely from the value, with no mode dependence.
inline float rint_ties_even(float x) {
    const float floor_x = std::floor(x);
    const float frac = x - floor_x;
    if (frac < 0.5f) {
        return floor_x;
    }
    if (frac > 0.5f) {
        return floor_x + 1.0f;
    }
    // Exact tie: round to the even neighbour. floor_x is integral, so its
    // parity decides; std::fmod is nonzero iff floor_x is odd.
    return (std::fmod(floor_x, 2.0f) == 0.0f) ? floor_x : floor_x + 1.0f;
}

// Quantize a single (already block-scaled) value into an MxInt element: a signed
// two's-complement integer of width `params.elem_width_bits`, returned masked
// into the low bits. Mirrors the validated tt-llk golden
// (_mxint_block_scale_and_quantize):
//   - NaN element -> 0 (MxInt has no NaN representation)
//   - +/-Inf element -> saturate to +/-elem_int_max
//   - otherwise int = clamp(round_ties_even(scaled * elem_int_scale), +/-elem_int_max)
// `scaled` is the input value already divided by the block scale (2^shared_exp).
inline uint32_t convert_to_mxint_elem_bits(float scaled, const FormatParams& params) {
    const uint32_t elem_mask = (params.elem_width_bits >= 32) ? 0xFFFFFFFFu : ((1u << params.elem_width_bits) - 1u);
    const int elem_max = params.elem_int_max;

    uint32_t bits = std::bit_cast<uint32_t>(scaled);
    uint32_t abs_bits = bits & 0x7FFFFFFFu;
    uint32_t exp_field = abs_bits >> 23;
    uint32_t mant_field = abs_bits & 0x7FFFFFu;
    bool is_nan = (exp_field == 0xFFu) && (mant_field != 0);
    bool is_inf = (exp_field == 0xFFu) && (mant_field == 0);

    int int_val;
    if (is_nan) {
        int_val = 0;
    } else if (is_inf) {
        int_val = (bits >> 31) ? -elem_max : elem_max;
    } else {
        float q = rint_ties_even(scaled * static_cast<float>(params.elem_int_scale));
        if (q > static_cast<float>(elem_max)) {
            int_val = elem_max;
        } else if (q < static_cast<float>(-elem_max)) {
            int_val = -elem_max;
        } else {
            int_val = static_cast<int>(q);
        }
    }

    return static_cast<uint32_t>(int_val) & elem_mask;
}

// Decode a single MxInt element field (width `params.elem_width_bits`, two's
// complement) into the value before the block scale is applied:
// int_val / elem_int_scale. The caller multiplies by the block scale 2^(scale-bias).
inline float convert_from_mxint_elem_bits(uint32_t elem_bits, const FormatParams& params) {
    const uint32_t width = params.elem_width_bits;
    const uint32_t sign_bit = 1u << (width - 1);
    int int_val = static_cast<int>(elem_bits & ((1u << width) - 1u));
    if (elem_bits & sign_bit) {
        int_val -= static_cast<int>(1u << width);  // sign-extend two's complement
    }
    return static_cast<float>(int_val) / static_cast<float>(params.elem_int_scale);
}

// Generic MxInt tile packer. Shares the OCP MX block-scale derivation (E8M0),
// the [scales | elements] tile layout, and the per-block word packing with the
// floating-point MX packer (pack_as_mx_tiles_impl); only the per-element
// quantization differs (integer instead of IEEE field extraction).
template <typename T>
std::vector<uint32_t> pack_as_mxint_tiles_impl(
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

    TT_ASSERT(params.is_integer, "pack_as_mxint_tiles_impl requires an integer FormatParams");
    TT_ASSERT(tile_HW % params.block_size == 0, "MX tile must be a multiple of {} elements", params.block_size);
    TT_ASSERT(data.size() % tile_HW == 0, "Input size must be a multiple of tile size");
    TT_ASSERT(
        params.elem_width_storage_bits > 0 && 32u % params.elem_width_storage_bits == 0,
        "MX elem storage width must divide 32 bits evenly");

    auto word_counts = compute_tile_word_counts(tile_HW, params);
    uint32_t exp_words = word_counts.exp_words;
    uint32_t elem_words = word_counts.elem_words;
    uint32_t exp_count = word_counts.exp_count;
    uint32_t tile_size_words = exp_words + elem_words;

    const uint32_t block_size = params.block_size;
    const uint32_t elem_storage_bits = params.elem_width_storage_bits;
    const uint32_t elements_per_word = 32u / elem_storage_bits;
    const uint32_t elem_words_per_block = (block_size * elem_storage_bits) / 32u;
    const uint32_t elem_shift = elem_storage_bits - params.elem_width_bits;
    const uint32_t elem_mask = (params.elem_width_bits >= 32) ? 0xFFFFFFFFu : ((1u << params.elem_width_bits) - 1u);

    uint32_t num_tiles = data.size() / tile_HW;
    // Zero-initialised so the alignment-padding tail of each tile's scale section
    // ends up as zeros (matches the fp MX packer).
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

        // Cast input to float in storage order, accumulating per-block stats.
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
            const auto& s = stats[blk_idx];
            // MxInt post-scaling values land in [1, 2): finalize_block_scale with
            // elem_exp_max_unbiased = 0 (set in the kMxInt*Params) yields the OCP
            // E8M0 shared exponent floor(log2(max_abs)), plus the all-NaN (0xFF)
            // and all-Inf-or-zero (0xFE) special cases.
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
                    uint32_t bits = convert_to_mxint_elem_bits(scaled, params);
                    word |= (bits & elem_mask) << (elem_shift + b * elem_storage_bits);
                }
                packed[blk_word_base + w] = word;
            }
        }
    }

    return packed;
}

// Generic MxInt tile unpacker. Mirror of pack_as_mxint_tiles_impl.
inline std::vector<float> unpack_mxint_tiles_into_float_vec_impl(
    tt::stl::Span<const uint32_t> mx_tiles,
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

    TT_ASSERT(params.is_integer, "unpack_mxint_tiles_into_float_vec_impl requires an integer FormatParams");
    TT_ASSERT(tile_HW % params.block_size == 0, "MX tile must be a multiple of {} elements", params.block_size);
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

            // OCP MX rule: a NaN block scale (0xFF) zeros the whole block. Avoids
            // the 0 * 2^(0xFF-bias) = 0 * inf = NaN trap of the generic multiply.
            if (scale_exp_biased == 0xFFu) {
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
                    float elem_pre_scale = convert_from_mxint_elem_bits(elem_bits, params);
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
