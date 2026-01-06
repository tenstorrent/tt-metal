// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>
#include <tt-metalium/bfloat4.hpp>
#include <tt_stl/span.hpp>
#include <array>
#include <bit>
#include <functional>
#include <random>
#include <vector>
#if TT_METAL_ENABLE_AVX
#include <simde/x86/avx2.h>
#endif

#include <tt_stl/assert.hpp>
#include "blockfloat_common.hpp"
#include "constants.hpp"
#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include "math.hpp"
#include "tile.hpp"
#include "tracy/Tracy.hpp"
#include "tt_backend_api_types.hpp"

std::vector<uint32_t> pack_fp32_vec_as_bfp4_tiles(
    tt::stl::Span<const float> fp32_vec,
    bool row_major_input,
    bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile) {
    return pack_as_bfp_tiles<tt::DataFormat::Bfp4_b>(fp32_vec, row_major_input, is_exp_a, tile);
}

template <typename T>
std::vector<uint32_t> pack_as_bfp4_tiles(
    tt::stl::Span<const T> data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile) {
    return pack_as_bfp_tiles<tt::DataFormat::Bfp4_b>(data, row_major_input, is_exp_a, tile);
}

template std::vector<uint32_t> pack_as_bfp4_tiles<bfloat16>(
    tt::stl::Span<const bfloat16> data,
    bool row_major_input,
    bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp4_tiles<float>(
    tt::stl::Span<const float> data,
    bool row_major_input,
    bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp4_tiles<int32_t>(
    tt::stl::Span<const int32_t> data,
    bool row_major_input,
    bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp4_tiles<uint32_t>(
    tt::stl::Span<const uint32_t> data,
    bool row_major_input,
    bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp4_tiles<uint8_t>(
    tt::stl::Span<const uint8_t> data,
    bool row_major_input,
    bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp4_tiles<uint16_t>(
    tt::stl::Span<const uint16_t> data,
    bool row_major_input,
    bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile);

std::vector<float> unpack_bfp4_tiles_into_float_vec(
    tt::stl::Span<const uint32_t> bfp_tiles,
    bool row_major_output,
    bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile) {
    ZoneScoped;

    uint32_t l1_alignment = tt::tt_metal::MetalContext::instance().hal().get_alignment(tt::tt_metal::HalMemType::L1);

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

    int num_elements_in_dword = 8;
    int data_dwords_per_exp = face_W / num_elements_in_dword;
    int num_exps_in_dword = 4;
    int data_dwords_per_exp_dword_log2 = log2(data_dwords_per_exp * num_exps_in_dword);
    int data_dwords_per_exp_log2 = log2(data_dwords_per_exp);

    uint32_t exp_bit_mask = (tile_HW == 16) ? 0x0 : (tile_HW == 32) ? 0x1 : 0x3;

    uint32_t size_bytes = bfp_tiles.size() * 4;
    uint32_t single_bfp_tile_size =
        tile.has_value() ? tile->get_tile_size(tt::DataFormat::Bfp4_b) : tile_size(tt::DataFormat::Bfp4_b);
    TT_ASSERT(size_bytes % single_bfp_tile_size == 0);
    uint32_t num_tiles = size_bytes / single_bfp_tile_size;

    int data_index;
    int subtile_r;
    int subtile_c;
    uint32_t exp_word, sub_word_index;

    uint32_t num_float_in_tile = subtiles_in_tile_row * subtiles_in_tile_col * subtile_rows * subtile_cols;
    uint32_t fp32_element_index = 0;

    uint32_t num_exp_words = tt::round_up(num_faces * face_H, l1_alignment) / num_exps_in_dword;
    uint32_t num_tile_words = tile_HW / num_elements_in_dword;
    int num_bfp_dwords_in_tile = num_tile_words + num_exp_words;
    int num_dwords_per_row = subtile_cols / num_elements_in_dword;

    std::vector<float> float_vec;
    float_vec.resize(num_tiles * num_float_in_tile);

#if TT_METAL_ENABLE_AVX
    const std::vector<uint32_t> mask_vec0 = {0xf, 0xf0, 0xf00, 0xf000};
    const std::vector<uint32_t> mask_vec1 = {0xf0000, 0xf00000, 0xf000000, 0xf0000000};
    const std::vector<uint32_t> shift_vec0 = {0, 4, 8, 12};
    const std::vector<uint32_t> shift_vec1 = {16, 20, 24, 28};
    const simde__m128i mask0 = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(mask_vec0.data()));
    const simde__m128i mask1 = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(mask_vec1.data()));
    const simde__m128i shift0 = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(shift_vec0.data()));
    const simde__m128i shift1 = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(shift_vec1.data()));
    simde__m256i rebias_offset = simde_mm256_setzero_si256();
    if (is_exp_a) {
        rebias_offset = simde_mm256_set1_epi32(-112);
    }

    for (int tile_index = 0; tile_index < num_tiles; ++tile_index) {
        for (int tr = 0; tr < subtiles_in_tile_row; ++tr) {
            for (int tc = 0; tc < subtiles_in_tile_col; ++tc) {
                for (int i = 0; i < subtile_rows; ++i) {
                    subtile_r = tr * subtile_rows + i;
                    for (int j = 0; j < subtile_cols; j += 2 * num_elements_in_dword) {
                        simde__m256i mask_denormal0 = simde_mm256_setzero_si256();
                        simde__m256i mask_denormal1 = simde_mm256_setzero_si256();
                        subtile_c = tc * subtile_cols + j;
                        data_index =
                            (tr * (subtiles_in_tile_col * face_HW / num_elements_in_dword) +
                             tc * (face_HW / num_elements_in_dword) + i * num_dwords_per_row +
                             j / num_elements_in_dword);
                        int tile_and_data_index = data_index + (num_bfp_dwords_in_tile * tile_index);

                        int exponent_index =
                            (data_index >> data_dwords_per_exp_dword_log2) + (num_bfp_dwords_in_tile * tile_index);
                        exp_word = bfp_tiles[exponent_index];

                        int num_exponent_words_skip = tile_index * num_exp_words;
                        sub_word_index = ((tile_and_data_index - num_exponent_words_skip) >> data_dwords_per_exp_log2) &
                                         exp_bit_mask;
                        simde__m256i exp_vector0 = simde_mm256_set1_epi32(get_byte(exp_word, sub_word_index));
                        simde__m256i exp_vector1 = exp_vector0;

                        uint32_t first_val = bfp_tiles[num_exp_words + tile_and_data_index];
                        simde__m128i first0 = simde_mm_set1_epi32(first_val);
                        simde__m128i first1 = simde_mm_set1_epi32(first_val);
                        uint32_t second_val = bfp_tiles[num_exp_words + tile_and_data_index + 1];
                        simde__m128i second0 = simde_mm_set1_epi32(second_val);
                        simde__m128i second1 = simde_mm_set1_epi32(second_val);

                        first0 = simde_mm_srlv_epi32(simde_mm_and_si128(first0, mask0), shift0);
                        first1 = simde_mm_srlv_epi32(simde_mm_and_si128(first1, mask1), shift1);
                        second0 = simde_mm_srlv_epi32(simde_mm_and_si128(second0, mask0), shift0);
                        second1 = simde_mm_srlv_epi32(simde_mm_and_si128(second1, mask1), shift1);
                        simde__m256i combined0 = simde_mm256_set_m128i(first1, first0);
                        simde__m256i combined1 = simde_mm256_set_m128i(second1, second0);

                        simde__m256i sign0 = simde_mm256_srl_epi32(combined0, simde_mm_set_epi64x(0, 3));
                        simde__m256i man0 = simde_mm256_and_si256(combined0, simde_mm256_set1_epi32(0x7));
                        simde__m256i sign1 = simde_mm256_srl_epi32(combined1, simde_mm_set_epi64x(0, 3));
                        simde__m256i man1 = simde_mm256_and_si256(combined1, simde_mm256_set1_epi32(0x7));

                        for (int idx = 0; idx < 2; idx++) {
                            simde__m256i shift_cnt = simde_mm256_setzero_si256();
                            simde__m256i man_shifted = idx == 0 ? man0 : man1;
                            simde__m256i select_mask = simde_mm256_cmpeq_epi32(man_shifted, shift_cnt);
                            for (int shift_val = 0; shift_val < 3; shift_val++) {
                                simde__m256i shift_mask = simde_mm256_or_si256(
                                    simde_mm256_cmpgt_epi32(man_shifted, simde_mm256_set1_epi32(0x4)),
                                    simde_mm256_cmpeq_epi32(man_shifted, simde_mm256_set1_epi32(0x4)));
                                man_shifted = simde_mm256_blendv_epi8(
                                    simde_mm256_sll_epi32(man_shifted, simde_mm_set_epi64x(0, 1)),
                                    man_shifted,
                                    shift_mask);
                                shift_cnt = simde_mm256_blendv_epi8(
                                    simde_mm256_set1_epi32(shift_val + 1), shift_cnt, shift_mask);
                            }
                            man_shifted = simde_mm256_and_si256(
                                simde_mm256_sll_epi32(man_shifted, simde_mm_set_epi64x(0, 1)),
                                simde_mm256_set1_epi32(0x7));
                            if (idx == 0) {
                                man0 = simde_mm256_blendv_epi8(man_shifted, man0, select_mask);
                            } else {
                                man1 = simde_mm256_blendv_epi8(man_shifted, man1, select_mask);
                            }

                            simde__m256i mask_shift_gt_exp =
                                simde_mm256_cmpgt_epi32(shift_cnt, idx == 0 ? exp_vector0 : exp_vector1);
                            simde__m256i mask_nonzero_mantissa =
                                simde_mm256_xor_si256(select_mask, simde_mm256_set1_epi32(-1));
                            simde__m256i mask_denormal =
                                simde_mm256_and_si256(mask_shift_gt_exp, mask_nonzero_mantissa);
                            if (idx == 0) {
                                mask_denormal0 = mask_denormal;
                            } else {
                                mask_denormal1 = mask_denormal;
                            }

                            if (idx == 0) {
                                exp_vector0 = simde_mm256_blendv_epi8(
                                    simde_mm256_sub_epi32(exp_vector0, simde_mm256_add_epi32(rebias_offset, shift_cnt)),
                                    simde_mm256_setzero_si256(),
                                    select_mask);
                            } else {
                                exp_vector1 = simde_mm256_blendv_epi8(
                                    simde_mm256_sub_epi32(exp_vector1, simde_mm256_add_epi32(rebias_offset, shift_cnt)),
                                    simde_mm256_setzero_si256(),
                                    select_mask);
                            }
                        }

                        sign0 = simde_mm256_sll_epi32(sign0, simde_mm_set_epi64x(0, 31));
                        exp_vector0 = simde_mm256_sll_epi32(exp_vector0, simde_mm_set_epi64x(0, 23));
                        man0 = simde_mm256_sll_epi32(man0, simde_mm_set_epi64x(0, 19));
                        sign1 = simde_mm256_sll_epi32(sign1, simde_mm_set_epi64x(0, 31));
                        exp_vector1 = simde_mm256_sll_epi32(exp_vector1, simde_mm_set_epi64x(0, 23));
                        man1 = simde_mm256_sll_epi32(man1, simde_mm_set_epi64x(0, 19));

                        simde__m256i man0_with_sign =
                            simde_mm256_or_si256(sign0, simde_mm256_or_si256(exp_vector0, man0));
                        simde__m256i man1_with_sign =
                            simde_mm256_or_si256(sign1, simde_mm256_or_si256(exp_vector1, man1));

                        man0_with_sign =
                            simde_mm256_blendv_epi8(man0_with_sign, simde_mm256_setzero_si256(), mask_denormal0);
                        man1_with_sign =
                            simde_mm256_blendv_epi8(man1_with_sign, simde_mm256_setzero_si256(), mask_denormal1);

                        uint32_t float_data_index;
                        if (row_major_output) {
                            float_data_index = subtile_c + (tile_W * subtile_r) + (tile_index * num_float_in_tile);
                            simde_mm256_storeu_ps(
                                &float_vec[float_data_index], simde_mm256_castsi256_ps(man0_with_sign));
                            simde_mm256_storeu_ps(
                                &float_vec[float_data_index + 8], simde_mm256_castsi256_ps(man1_with_sign));
                        } else {
                            float_data_index = fp32_element_index;
                            fp32_element_index += 16;
                            simde_mm256_storeu_ps(
                                &float_vec[float_data_index], simde_mm256_castsi256_ps(man0_with_sign));
                            simde_mm256_storeu_ps(
                                &float_vec[float_data_index + 8], simde_mm256_castsi256_ps(man1_with_sign));
                        }
                    }
                }
            }
        }
    }
#else
    for (int tile_index = 0; tile_index < num_tiles; ++tile_index) {
        for (int tr = 0; tr < subtiles_in_tile_row; ++tr) {
            for (int tc = 0; tc < subtiles_in_tile_col; ++tc) {
                for (int i = 0; i < subtile_rows; ++i) {
                    subtile_r = tr * subtile_rows + i;
                    for (int j = 0; j < subtile_cols; j += 2 * num_elements_in_dword) {
                        subtile_c = tc * subtile_cols + j;
                        data_index =
                            (tr * (subtiles_in_tile_col * face_HW / num_elements_in_dword) +
                             tc * (face_HW / num_elements_in_dword) + i * num_dwords_per_row +
                             j / num_elements_in_dword);
                        int tile_and_data_index = data_index + (num_bfp_dwords_in_tile * tile_index);
                        int exponent_index =
                            (data_index >> data_dwords_per_exp_dword_log2) + (num_bfp_dwords_in_tile * tile_index);
                        exp_word = bfp_tiles[exponent_index];
                        int num_exponent_words_skip = tile_index * num_exp_words;
                        sub_word_index = ((tile_and_data_index - num_exponent_words_skip) >> data_dwords_per_exp_log2) &
                                         exp_bit_mask;
                        uint8_t shared_exp = get_byte(exp_word, sub_word_index);

                        uint32_t word0 = bfp_tiles[num_exp_words + tile_and_data_index];
                        uint32_t word1 = bfp_tiles[num_exp_words + tile_and_data_index + 1];
                        uint32_t float_data_index =
                            row_major_output ? (subtile_c + (tile_W * subtile_r) + (tile_index * num_float_in_tile))
                                             : fp32_element_index;

                        for (int lane = 0; lane < 16; ++lane) {
                            uint32_t word = (lane < 8) ? word0 : word1;
                            uint8_t datum = static_cast<uint8_t>((word >> ((lane % 8) * 4)) & 0xF);
                            uint32_t bit_val = convert_bfp_to_u32(tt::DataFormat::Bfp4_b, datum, shared_exp, is_exp_a);
                            float value = std::bit_cast<float>(bit_val);
                            float_vec[float_data_index + lane] = value;
                        }

                        if (!row_major_output) {
                            fp32_element_index += 16;
                        }
                    }
                }
            }
        }
    }
#endif
    return float_vec;
}
