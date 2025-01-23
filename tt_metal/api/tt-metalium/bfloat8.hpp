// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
#include <immintrin.h>

#include "assert.hpp"
#include "blockfloat_common.hpp"
#include "tt_backend_api_types.hpp"
#include "span.hpp"
#include "tracy/Tracy.hpp"

// TODO: empty struct to facilitate Tensor template logic. Reconsider how/why templating is supported in Tensor
struct bfloat8_b {};

template <bool truncate_bfp_mantissa = false>
inline uint8_t convert_u32_to_bfp8(uint32_t input, uint32_t shared_exp, bool is_exp_a) {
    // check for both +/- 0.0
    constexpr uint32_t EXP_MANTISSA_BMSK = ((1U << 31) - 1);
    bool is_zero = ((input & EXP_MANTISSA_BMSK) == 0);

    if (is_zero) {
        return 0;
    }

    uint32_t mantissa = input & 0x007fffff;
    uint32_t exp = (input & 0x7f800000) >> 23;
    uint32_t sign = (input & 0x80000000) >> 31;

    if (is_exp_a) {
        int32_t se = static_cast<int32_t>(exp);
        // rebias
        se = se - 127 + 15;
        // check for saturation
        if (se > 31) {
            se = 31;
            mantissa = 0x007fffff;
        } else if (se < 0) {
            se = 0;
            mantissa = 0x0;
        }

        exp = static_cast<uint32_t>(se);
    }

    // float mantissa is 23 bits + hidden bit = 24 bits
    // add hidden 1
    mantissa = (1 << 23) | mantissa;

    if (shared_exp >= exp) {
        int exp_diff = shared_exp - exp;
        // shift mantissa further down by exp diff
        // In bit-shift operation (A >> B), the result is undefined if B is greater than or equal to the number of bits
        // in A
        while (exp_diff > 31) {
            mantissa = mantissa >> 31;
            exp_diff -= 31;
        }
        mantissa = mantissa >> exp_diff;
    }

    // this needs to become 7 bits so shift 17 times
    if (truncate_bfp_mantissa) {
        // Truncation: Round down
        mantissa = mantissa >> 17;
    } else {
        // Round mantissa to nearest even
        mantissa += 1 << 16;
        mantissa = mantissa >> 17;
        if (mantissa > 127) {
            mantissa = 127;
        }
    }

    // add sign bit only if result is not 0
    if (0 == mantissa) {
        sign = 0;
    }
    mantissa = (sign << 7) | mantissa;
    return mantissa;
}

inline uint32_t create_packed_bfp8_packed_as_u32(
    const std::vector<uint32_t>& u32_vec, uint32_t shared_exp, bool is_exp_a) {
    int nums_in_dword = 4;
    uint32_t tmp_o = 0;
    uint32_t mask = (1 << (32 / nums_in_dword)) - 1;
    for (int i = nums_in_dword - 1; i >= 0; --i)  // [0] in LSBs of dword
    {
        uint32_t conv_num = convert_u32_to_bfp8(u32_vec[i], shared_exp, is_exp_a);
        tmp_o = tmp_o << (32 / nums_in_dword);
        tmp_o = tmp_o | (conv_num & mask);
    }
    return tmp_o;
}

inline std::vector<uint32_t> pack_fp32_vec_as_bfp8_tiles(
    tt::stl::Span<const float> fp32_vec,
    bool row_major_input,
    bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt) {
    return pack_fp32_vec_as_bfp_tiles<tt::DataFormat::Bfp8_b>(fp32_vec, row_major_input, is_exp_a, tile);
}

inline std::vector<float> unpack_bfp8_tiles_into_float_vec(
    const std::vector<uint32_t>& bfp8_tiles,
    bool row_major_output,
    bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt) {
    ZoneScoped;

    uint32_t l1_alignment = tt::tt_metal::hal.get_alignment(tt::tt_metal::HalMemType::L1);

    auto tile_H = tile.has_value() ? tile->get_tile_shape()[0] : tt::constants::TILE_HEIGHT;
    auto tile_W = tile.has_value() ? tile->get_tile_shape()[1] : tt::constants::TILE_WIDTH;
    auto face_H = tile.has_value() ? tile->get_face_shape()[0] : tt::constants::FACE_HEIGHT;
    auto face_W = tile.has_value() ? tile->get_face_shape()[1] : tt::constants::FACE_WIDTH;
    auto tile_HW = tile_H * tile_W;
    auto face_HW = face_H * face_W;
    auto num_faces = tile_HW / face_HW;
    auto subtiles_in_tile_row = tile_H / face_H;
    auto subtiles_in_tile_col = tile_W / face_W;
    auto subtile_rows = face_H;
    auto subtile_cols = face_W;
    uint32_t num_exp_words = tt::round_up(num_faces * face_H, l1_alignment) / 4;
    uint32_t num_tile_words = tile_HW / 4;
    uint32_t num_bfp8_in_tile = num_tile_words + num_exp_words;

    // the exponent index will always be 0 when tile_HW == 16, between 0-1 when tile_HW == 32, and between 0-3 otherwise
    uint32_t exp_bit_mask = (tile_HW == 16) ? 0x0 : (tile_HW == 32) ? 0x1 : 0x3;

    int num_elements_in_dword = 4;
    uint32_t size_bytes = bfp8_tiles.size() * num_elements_in_dword;  // each uint32_t contains 4 BFP8 values
    uint32_t single_bfp8_tile_size =
        tile.has_value() ? tile->get_tile_size(tt::DataFormat::Bfp8_b) : tile_size(tt::DataFormat::Bfp8_b);
    TT_ASSERT(size_bytes % single_bfp8_tile_size == 0);
    uint32_t num_tiles = size_bytes / single_bfp8_tile_size;

    int data_index;
    int subtile_r;
    int subtile_c;
    const std::vector<uint32_t> mask_vec = {0xff, 0xff00, 0xff0000, 0xff000000};
    const std::vector<uint32_t> shift_vec = {0, 8, 16, 24};
    const __m128i mask = _mm_loadu_si128(reinterpret_cast<const __m128i*>(mask_vec.data()));
    const __m128i shift = _mm_loadu_si128(reinterpret_cast<const __m128i*>(shift_vec.data()));
    __m256i rebias_offset = _mm256_setzero_si256();
    if (is_exp_a) {
        rebias_offset =
            _mm256_set1_epi32(-112);  // This rebias offset must be added if we are working with BFP8 format.
    }
    uint32_t exp_word, sub_word_index;

    uint32_t num_float_in_tile = subtiles_in_tile_row * subtiles_in_tile_col * subtile_rows * subtile_cols;
    uint32_t fp32_element_index = 0;
    std::vector<float> float_vec;
    float_vec.resize(num_tiles * num_float_in_tile);
    for (int tile_index = 0; tile_index < num_tiles; ++tile_index) {
        for (int tr = 0; tr < subtiles_in_tile_row; ++tr) {
            for (int tc = 0; tc < subtiles_in_tile_col; ++tc) {
                for (int i = 0; i < subtile_rows; ++i) {
                    subtile_r = tr * face_H + i;
                    for (int j = 0; j < subtile_cols; j += 8) {
                        subtile_c = tc * face_W + j;
                        data_index =
                            (tr * (subtiles_in_tile_col * face_HW / 4) + tc * (face_HW / 4) + i * (face_W / 4) +
                             j / 4);  // Each uint32_t contains 4 BFP8 values. Divide data index by 4.
                        int tile_and_data_index = data_index + (num_bfp8_in_tile * tile_index);

                        int exponent_index = (data_index >> 4) + (num_bfp8_in_tile * tile_index);
                        exp_word = bfp8_tiles.at(
                            exponent_index);  // Extract the uint32_t value that stores the shared exponent for this set
                                              // of data. Each 32 bit word is shared amongst 64 datums

                        int num_exponent_words_skip = tile_index * num_exp_words;
                        sub_word_index = ((tile_and_data_index - num_exponent_words_skip) >> 2) &
                                         exp_bit_mask;  // Extract the byte in which the shared exponent is stored. Each
                                                        // byte is shared amongst 16 datums.
                        __m256i exp_vector =
                            _mm256_set1_epi32(get_byte(exp_word, sub_word_index));  // Replicate exp scalar in a vector
                        // Take 2 uint32_t values. These are 8 BFP8 values
                        __m128i first = _mm_set1_epi32(bfp8_tiles.at(
                            num_exp_words +
                            tile_and_data_index));  // Replicate first uint32_t 4 times (one for each BFP8 value)
                        __m128i second = _mm_set1_epi32(bfp8_tiles.at(
                            num_exp_words + tile_and_data_index + 1));  //  Replicate second uint32_t 4 times
                        first = _mm_srlv_epi32(
                            _mm_and_si128(first, mask), shift);  // Extract each BFP8 from the first uint32_t
                        second = _mm_srlv_epi32(
                            _mm_and_si128(second, mask), shift);  // Extract each BFP8 from the second uint32_t
                        __m256i combined = _mm256_set_m128i(second, first);  // Concatenate 2 128 vectors to 1 256
                        // Extract sign and mantissa (expo extracted above)
                        __m256i sign = _mm256_srl_epi32(combined, _mm_set_epi64x(0, 7));
                        __m256i man = _mm256_and_si256(combined, _mm256_set1_epi32(0x7f));
                        // Initialize shift amount per datum to 0. This is incremented below.
                        __m256i shift_cnt = _mm256_setzero_si256();
                        __m256i select_mask = _mm256_cmpeq_epi32(
                            man, shift_cnt);  // This mask is used to set mantissa values to 0, if they start at 0.
                        __m256i man_shifted = man;  // Initialize updated mantissa
                        for (int shift_val = 0; shift_val < 7; shift_val++) {
                            // Shift each mantissa and update the corresponding shift_cnt until the 6th bit of the 8 bit
                            // data is set.
                            __m256i shift_mask = _mm256_or_si256(
                                _mm256_cmpgt_epi32(man_shifted, _mm256_set1_epi32(0x40)),
                                _mm256_cmpeq_epi32(
                                    man_shifted,
                                    _mm256_set1_epi32(0x40)));  // If the 6th bit is set, propagate the current mantissa
                                                                // value. Else take the left shifted value
                            man_shifted = _mm256_blendv_epi8(
                                _mm256_sll_epi32(man_shifted, _mm_set_epi64x(0, 1)), man_shifted, shift_mask);
                            shift_cnt = _mm256_blendv_epi8(_mm256_set1_epi32(shift_val + 1), shift_cnt, shift_mask);
                        }
                        man_shifted = _mm256_and_si256(
                            _mm256_sll_epi32(man_shifted, _mm_set_epi64x(0, 1)),
                            _mm256_set1_epi32(0x7f));  // One more shift to clear 6th bit
                        man = _mm256_blendv_epi8(
                            man_shifted,
                            man,
                            select_mask);  // Choose new mantissa or keep old mantissa based on 0 initial condition.
                        // Assert if the exponent and corresponding mantissa for a datum are non-zero and the
                        // subtraction bias (shift_cnt) for that data is greater than the exponent value
                        TT_ASSERT(
                            !(_mm256_movemask_ps(
                                  _mm256_castsi256_ps(_mm256_cmpgt_epi32(exp_vector, _mm256_setzero_si256()))) &
                              _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(shift_cnt, exp_vector))) &
                              !_mm256_movemask_ps(_mm256_castsi256_ps(select_mask))),
                            "Device returned incorrect data for Bfp8 formats: The Shift Count for a non-zero exponent "
                            "is greater than the exponent value.");
                        exp_vector = _mm256_blendv_epi8(
                            _mm256_sub_epi32(exp_vector, _mm256_add_epi32(rebias_offset, shift_cnt)),
                            _mm256_setzero_si256(),
                            select_mask);  // Choose new (rebiased exponent) or keep previous exponent based on mantissa
                                           // intiial condition

                        sign = _mm256_sll_epi32(sign, _mm_set_epi64x(0, 31));              // Shift sign
                        exp_vector = _mm256_sll_epi32(exp_vector, _mm_set_epi64x(0, 23));  // Shift exp
                        man = _mm256_sll_epi32(man, _mm_set_epi64x(0, 16));                // Shift mantissa
                        man = _mm256_or_si256(
                            sign, _mm256_or_si256(exp_vector, man));  // Store final value in mantissa register and save

                        uint32_t float_data_index;
                        if (row_major_output) {
                            float_data_index = subtile_c + (tile_W * subtile_r) + (tile_index * num_float_in_tile);
                        } else {
                            float_data_index = fp32_element_index;
                            fp32_element_index += 8;
                        }
                        _mm256_storeu_ps(&float_vec[float_data_index], _mm256_castsi256_ps(man));
                    }
                }
            }
        }
    }
    return float_vec;
}

inline std::vector<uint32_t> create_random_vector_of_bfp8(
    uint32_t num_bytes, bool is_exp_a, int rand_max_float, int seed, float offset = 0.0f) {
    uint32_t single_bfp8_tile_size = tile_size(tt::DataFormat::Bfp8_b);
    TT_ASSERT(num_bytes % single_bfp8_tile_size == 0);
    uint32_t num_tiles = num_bytes / single_bfp8_tile_size;

    auto rand_float = std::bind(std::uniform_real_distribution<float>(0, rand_max_float), std::mt19937(seed));

    int packed_data_size = num_bytes / sizeof(float);
    int num_float_in_tile = 1024;
    int float_data_size = num_tiles * num_float_in_tile;

    std::vector<float> fp32_vec(float_data_size, 0);
    for (int i = 0; i < fp32_vec.size(); i++) {
        fp32_vec.at(i) = rand_float() + offset;
    }

    std::vector<uint32_t> packed_result = pack_fp32_vec_as_bfp8_tiles(fp32_vec, /*row_major_input=*/true, is_exp_a);

    TT_ASSERT(packed_result.size() == packed_data_size);

    return packed_result;
}

inline std::vector<uint32_t> create_constant_vector_of_bfp8(uint32_t num_bytes, float value, bool is_exp_a) {
    uint32_t single_bfp8_tile_size = tile_size(tt::DataFormat::Bfp8_b);
    TT_ASSERT(num_bytes % single_bfp8_tile_size == 0);
    uint32_t num_tiles = num_bytes / single_bfp8_tile_size;

    int packed_data_size = num_bytes / sizeof(float);
    int num_float_in_tile = 1024;
    int float_data_size = num_tiles * num_float_in_tile;

    std::vector<float> fp32_vec(float_data_size, 0);
    for (int i = 0; i < fp32_vec.size(); i++) {
        fp32_vec.at(i) = value;
    }

    std::vector<uint32_t> packed_result = pack_fp32_vec_as_bfp8_tiles(fp32_vec, /*row_major_input=*/true, is_exp_a);

    TT_ASSERT(packed_result.size() == packed_data_size);

    return packed_result;
}
