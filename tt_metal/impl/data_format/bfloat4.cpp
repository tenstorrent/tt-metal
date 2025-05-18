// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>
#include <tt-metalium/bfloat4.hpp>
#include <tt_stl/span.hpp>
#include <array>
#include <functional>
#include <random>
#include <vector>
#include <simde/x86/avx2.h>

#include "assert.hpp"
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
    return pack_fp32_vec_as_bfp_tiles<tt::DataFormat::Bfp4_b>(fp32_vec, row_major_input, is_exp_a, tile);
}

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

    // the exponent index will always be 0 when tile_HW == 16, between 0-1 when tile_HW == 32, and between 0-3 otherwise
    uint32_t exp_bit_mask = (tile_HW == 16) ? 0x0 : (tile_HW == 32) ? 0x1 : 0x3;

    uint32_t size_bytes = bfp_tiles.size() * 4;
    uint32_t single_bfp_tile_size =
        tile.has_value() ? tile->get_tile_size(tt::DataFormat::Bfp4_b) : tile_size(tt::DataFormat::Bfp4_b);
    TT_ASSERT(size_bytes % single_bfp_tile_size == 0);
    uint32_t num_tiles = size_bytes / single_bfp_tile_size;

    int data_index;
    int subtile_r;
    int subtile_c;
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
        rebias_offset =
            simde_mm256_set1_epi32(-112);  // This rebias offset must be added if we are working with BFP8 format.
    }
    uint32_t exp_word, sub_word_index;

    uint32_t num_float_in_tile = subtiles_in_tile_row * subtiles_in_tile_col * subtile_rows * subtile_cols;
    uint32_t fp32_element_index = 0;

    uint32_t num_exp_words = tt::round_up(num_faces * face_H, l1_alignment) / num_exps_in_dword;
    uint32_t num_tile_words = tile_HW / num_elements_in_dword;
    int num_bfp_dwords_in_tile = num_tile_words + num_exp_words;
    int num_dwords_per_row = subtile_cols / num_elements_in_dword;

    std::vector<float> float_vec;
    float_vec.resize(num_tiles * num_float_in_tile);
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
                             j / num_elements_in_dword);  // Each uint32_t contains 8 BFP4 values. Divide data index by
                                                          // 8
                        int tile_and_data_index = data_index + (num_bfp_dwords_in_tile * tile_index);

                        int exponent_index =
                            (data_index >> data_dwords_per_exp_dword_log2) + (num_bfp_dwords_in_tile * tile_index);
                        // Extract the uint32_t value that stores the shared exponent for this set
                        // of data. Each 32 bit word is shared amongst 64 datums
                        exp_word = bfp_tiles[exponent_index];

                        int num_exponent_words_skip = tile_index * num_exp_words;
                        sub_word_index = ((tile_and_data_index - num_exponent_words_skip) >> data_dwords_per_exp_log2) &
                                         exp_bit_mask;  // Extract the byte in which the shared exponent is stored. Each
                                                        // byte is shared amongst 16 datums.
                        simde__m256i exp_vector0 = simde_mm256_set1_epi32(
                            get_byte(exp_word, sub_word_index));  // Replicate exp scalar in a vector
                        simde__m256i exp_vector1 = exp_vector0;

                        // Take 2 uint32_t values. These are 16 BFP4 values
                        // Replicate each uint32 value 8 times - one for each bfp4 value
                        uint32_t first_val = bfp_tiles[num_exp_words + tile_and_data_index];
                        simde__m128i first0 = simde_mm_set1_epi32(first_val);
                        simde__m128i first1 = simde_mm_set1_epi32(first_val);
                        uint32_t second_val = bfp_tiles[num_exp_words + tile_and_data_index + 1];
                        simde__m128i second0 = simde_mm_set1_epi32(second_val);
                        simde__m128i second1 = simde_mm_set1_epi32(second_val);

                        first0 = simde_mm_srlv_epi32(
                            simde_mm_and_si128(first0, mask0), shift0);  // Extract each BFP4 from the first0 uint32_t
                        first1 = simde_mm_srlv_epi32(
                            simde_mm_and_si128(first1, mask1), shift1);  // Extract each BFP4 from the first1 uint32_t
                        second0 = simde_mm_srlv_epi32(
                            simde_mm_and_si128(second0, mask0), shift0);  // Extract each BFP4 from the first0 uint32_t
                        second1 = simde_mm_srlv_epi32(
                            simde_mm_and_si128(second1, mask1), shift1);  // Extract each BFP4 from the first1 uint32_t
                        simde__m256i combined0 =
                            simde_mm256_set_m128i(first1, first0);  // Concatenate 2 128 vectors to 1 256
                        simde__m256i combined1 =
                            simde_mm256_set_m128i(second1, second0);  // Concatenate 2 128 vectors to 1 256

                        // Extract sign and mantissa (expo extracted above)
                        simde__m256i sign0 = simde_mm256_srl_epi32(combined0, simde_mm_set_epi64x(0, 3));
                        simde__m256i man0 = simde_mm256_and_si256(combined0, simde_mm256_set1_epi32(0x7));
                        simde__m256i sign1 = simde_mm256_srl_epi32(combined1, simde_mm_set_epi64x(0, 3));
                        simde__m256i man1 = simde_mm256_and_si256(combined1, simde_mm256_set1_epi32(0x7));

                        for (int i = 0; i < 2; i++) {
                            simde__m256i shift_cnt = simde_mm256_setzero_si256();  // Initialize shift amount per datum
                                                                                   // to 0. This is incremented below.
                            simde__m256i man_shifted = i == 0 ? man0 : man1;       // Initialize updated mantissa
                            simde__m256i select_mask = simde_mm256_cmpeq_epi32(
                                man_shifted,
                                shift_cnt);  // This mask is used to set mantissa values to 0, if they start at 0.
                            for (int shift_val = 0; shift_val < 3; shift_val++) {
                                // Shift each mantissa and update the corresponding shift_cnt until the 3rd bit of the 8
                                // bit data is set.
                                simde__m256i shift_mask = simde_mm256_or_si256(
                                    simde_mm256_cmpgt_epi32(man_shifted, simde_mm256_set1_epi32(0x4)),
                                    simde_mm256_cmpeq_epi32(
                                        man_shifted,
                                        simde_mm256_set1_epi32(
                                            0x4)));  // If the 6th bit is set, propagate the current
                                                     // mantissa value. Else take the left shifted value
                                man_shifted = simde_mm256_blendv_epi8(
                                    simde_mm256_sll_epi32(man_shifted, simde_mm_set_epi64x(0, 1)),
                                    man_shifted,
                                    shift_mask);
                                shift_cnt = simde_mm256_blendv_epi8(
                                    simde_mm256_set1_epi32(shift_val + 1), shift_cnt, shift_mask);
                            }
                            man_shifted = simde_mm256_and_si256(
                                simde_mm256_sll_epi32(man_shifted, simde_mm_set_epi64x(0, 1)),
                                simde_mm256_set1_epi32(
                                    0x7));  // One more shift to clear 3rd bit; Mask with 3bits for mantissa
                            if (i == 0) {
                                man0 = simde_mm256_blendv_epi8(
                                    man_shifted, man0, select_mask);  // Choose new mantissa or keep old mantissa based
                                                                      // on 0 initial condition.
                                // Flush denormals to zero
                                // Check if shift > exp and mantissa is not zero
                                simde__m256i mask_shift_gt_exp = simde_mm256_cmpgt_epi32(shift_cnt, exp_vector0);
                                simde__m256i mask_nonzero_mantissa = simde_mm256_xor_si256(select_mask, simde_mm256_set1_epi32(-1));
                                mask_denormal0 = simde_mm256_and_si256(mask_shift_gt_exp, mask_nonzero_mantissa);

                                exp_vector0 = simde_mm256_blendv_epi8(
                                    simde_mm256_sub_epi32(exp_vector0, simde_mm256_add_epi32(rebias_offset, shift_cnt)),
                                    simde_mm256_setzero_si256(),
                                    select_mask);  // Choose new (rebiased exponent) or keep previous exponent based on
                                                   // mantissa intiial condition
                            } else {
                                man1 = simde_mm256_blendv_epi8(
                                    man_shifted, man1, select_mask);  // Choose new mantissa or keep old mantissa based
                                                                      // on 0 initial condition.
                                // Flush denormals to zero
                                // Check if shift > exp and mantissa is not zero
                                simde__m256i mask_shift_gt_exp = simde_mm256_cmpgt_epi32(shift_cnt, exp_vector1);
                                simde__m256i mask_nonzero_mantissa = simde_mm256_xor_si256(select_mask, simde_mm256_set1_epi32(-1));
                                mask_denormal1 = simde_mm256_and_si256(mask_shift_gt_exp, mask_nonzero_mantissa);
                                exp_vector1 = simde_mm256_blendv_epi8(
                                    simde_mm256_sub_epi32(exp_vector1, simde_mm256_add_epi32(rebias_offset, shift_cnt)),
                                    simde_mm256_setzero_si256(),
                                    select_mask);  // Choose new (rebiased exponent) or keep previous exponent based on
                                                   // mantissa intiial condition
                            }
                        }

                        sign0 = simde_mm256_sll_epi32(sign0, simde_mm_set_epi64x(0, 31));              // Shift sign
                        sign1 = simde_mm256_sll_epi32(sign1, simde_mm_set_epi64x(0, 31));              // Shift sign
                        exp_vector0 = simde_mm256_sll_epi32(exp_vector0, simde_mm_set_epi64x(0, 23));  // Shift exp
                        exp_vector1 = simde_mm256_sll_epi32(exp_vector1, simde_mm_set_epi64x(0, 23));  // Shift exp
                        man0 = simde_mm256_sll_epi32(man0, simde_mm_set_epi64x(0, 20));                // Shift mantissa
                        man0 = simde_mm256_or_si256(
                            sign0,
                            simde_mm256_or_si256(
                                exp_vector0, man0));  // Store final value in mantissa register and save
                        man1 = simde_mm256_sll_epi32(man1, simde_mm_set_epi64x(0, 20));  // Shift mantissa
                        man1 = simde_mm256_or_si256(
                            sign1,
                            simde_mm256_or_si256(
                                exp_vector1, man1));  // Store final value in mantissa register and save

                        // Zero out lanes where mask_denormal is true
                        man0 = simde_mm256_blendv_epi8(man0, simde_mm256_setzero_si256(), mask_denormal0);
                        man1 = simde_mm256_blendv_epi8(man1, simde_mm256_setzero_si256(), mask_denormal1);
                        uint32_t float_data_index;
                        if (row_major_output) {
                            float_data_index = subtile_c + (tile_W * subtile_r) + (tile_index * num_float_in_tile);
                        } else {
                            float_data_index = fp32_element_index;
                            fp32_element_index += 2 * num_elements_in_dword;
                        }
                        simde_mm256_storeu_ps(&float_vec[float_data_index], simde_mm256_castsi256_ps(man0));
                        simde_mm256_storeu_ps(
                            &float_vec[float_data_index + num_elements_in_dword], simde_mm256_castsi256_ps(man1));
                    }
                }
            }
        }
    }
    return float_vec;
}

std::vector<uint32_t> create_random_vector_of_bfp4(
    uint32_t num_bytes, bool is_exp_a, int rand_max_float, int seed, float offset) {
    uint32_t single_bfp4_tile_size = tile_size(tt::DataFormat::Bfp4_b);
    TT_ASSERT(num_bytes % single_bfp4_tile_size == 0);
    uint32_t num_tiles = num_bytes / single_bfp4_tile_size;

    auto rand_float = std::bind(std::uniform_real_distribution<float>(0, rand_max_float), std::mt19937(seed));

    int packed_data_size = num_bytes / sizeof(float);
    int num_float_in_tile = 1024;
    int float_data_size = num_tiles * num_float_in_tile;

    std::vector<float> fp32_vec(float_data_size, 0);
    for (int i = 0; i < fp32_vec.size(); i++) {
        fp32_vec.at(i) = rand_float() + offset;
    }

    std::vector<uint32_t> packed_result = pack_fp32_vec_as_bfp4_tiles(fp32_vec, /*row_major_input=*/true, is_exp_a);

    TT_ASSERT(packed_result.size() == packed_data_size);

    return packed_result;
}

std::vector<uint32_t> create_constant_vector_of_bfp4(uint32_t num_bytes, float value, bool is_exp_a) {
    uint32_t single_bfp4_tile_size = tile_size(tt::DataFormat::Bfp4_b);
    TT_ASSERT(num_bytes % single_bfp4_tile_size == 0);
    uint32_t num_tiles = num_bytes / single_bfp4_tile_size;

    int packed_data_size = num_bytes / sizeof(float);
    int num_float_in_tile = 1024;
    int float_data_size = num_tiles * num_float_in_tile;

    std::vector<float> fp32_vec(float_data_size, 0);
    for (int i = 0; i < fp32_vec.size(); i++) {
        fp32_vec.at(i) = value;
    }

    std::vector<uint32_t> packed_result = pack_fp32_vec_as_bfp4_tiles(fp32_vec, /*row_major_input=*/true, is_exp_a);

    TT_ASSERT(packed_result.size() == packed_data_size);

    return packed_result;
}
