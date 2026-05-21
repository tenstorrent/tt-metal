// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/mxfp4.hpp>

#include <cmath>
#include <optional>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/mx_common.hpp>
#include <tt-metalium/tile.hpp>
#include "tracy/Tracy.hpp"

std::vector<float> unpack_mxfp4_tiles_into_float_vec(
    tt::stl::Span<const uint32_t> mxfp4_tiles, bool row_major_output, const std::optional<tt::tt_metal::Tile>& tile) {
    ZoneScoped;

    auto tile_H = tile.has_value() ? tile->get_tile_shape()[0] : tt::constants::TILE_HEIGHT;
    auto tile_W = tile.has_value() ? tile->get_tile_shape()[1] : tt::constants::TILE_WIDTH;
    auto face_H = tile.has_value() ? tile->get_face_shape()[0] : tt::constants::FACE_HEIGHT;
    auto face_W = tile.has_value() ? tile->get_face_shape()[1] : tt::constants::FACE_WIDTH;
    auto tile_HW = tile_H * tile_W;
    auto subtiles_in_tile_row = tile_H / face_H;
    auto subtiles_in_tile_col = tile_W / face_W;

    TT_ASSERT(
        tile_HW % tt::tt_metal::mx::kMxFp4Params.block_size == 0,
        "MXFP4 tile must be a multiple of {} elements",
        tt::tt_metal::mx::kMxFp4Params.block_size);

    auto word_counts = tt::tt_metal::mx::compute_tile_word_counts(tile_HW, tt::tt_metal::mx::kMxFp4Params);
    uint32_t exp_count = word_counts.exp_count;
    uint32_t exp_words = word_counts.exp_words;
    uint32_t elem_words = word_counts.elem_words;

    uint32_t tile_size_words = exp_words + elem_words;
    TT_ASSERT(mxfp4_tiles.size() % tile_size_words == 0, "Input size must be a multiple of MXFP4 tile size");
    uint32_t num_tiles = mxfp4_tiles.size() / tile_size_words;

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
        tt::tt_metal::mx::unpack_exp_words(mxfp4_tiles, word_offset, exp_words, exp_count, exps);
        tt::tt_metal::mx::unpack_elem_words(
            mxfp4_tiles, word_offset, elem_words, tile_HW, tt::tt_metal::mx::kMxFp4Params, elems);

        for (uint32_t blk = 0; blk < exp_count; ++blk) {
            uint8_t scale_exp_biased = exps[blk];
            int scale_exp_unbiased = static_cast<int>(scale_exp_biased) - tt::tt_metal::mx::kMxFp4Params.scale_bias;
            const float scale_unpack = tt::tt_metal::mx::pow2_f32(scale_exp_unbiased);
            uint32_t base = blk * tt::tt_metal::mx::kMxFp4Params.block_size;
            for (uint32_t j = 0; j < tt::tt_metal::mx::kMxFp4Params.block_size; ++j) {
                uint32_t i = base + j;
                float elem_pre_scale = tt::tt_metal::mx::convert_from_mx_elem_bits(
                    elems[i], scale_exp_biased, tt::tt_metal::mx::kMxFp4Params);
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
