// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <tt_stl/span.hpp>

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
TileWordCounts compute_tile_word_counts(uint32_t elem_count, uint32_t l1_alignment, const FormatParams& params);

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

}  // namespace tt::tt_metal::mx
