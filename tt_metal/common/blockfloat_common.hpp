// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
#include <immintrin.h>

#include "tt_metal/common/assert.hpp"
#include "tt_metal/common/tt_backend_api_types.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"


inline uint8_t get_max_exp(const std::vector<uint32_t> &vec, bool is_exp_a) {
    TT_ASSERT(vec.size() == 16);
    uint32_t max = 0;

    for (int i = 0; i < 16; ++i) {
        // mask & shift out exp
        uint32_t exp = (vec[i] & 0x7f800000) >> 23;

        if (is_exp_a) {
            int32_t se = static_cast<int32_t>(exp);
            // need to rebias from 127 to 15
            se = se - 127 + 15;

            if (se > 31) {
                se = 31;
            }
            else if (se < 0) {
                se = 0;
            }

            exp = static_cast<uint32_t>(se);
        }

        if (exp > max) {
            max = exp;
        }
    }
    return max;
}

inline uint32_t get_exp_dword(const std::vector<uint8_t> &vec) {
    TT_ASSERT(vec.size() == 4);

    uint32_t tmp = 0;
    for(int i = 0; i < 4; ++i) {
        tmp = tmp | ((vec[i] & 0xff) << (i*8));
    }
    return tmp;
}

inline uint32_t get_byte(uint32_t word, uint32_t index) {
    TT_ASSERT(index < 4);
    uint32_t mask = 0xff << (8 * index);
    uint32_t masked = word & mask;
    masked = masked >> (8 * index);
    return masked;
}

template <tt::DataFormat BfpFormat, bool truncate_bfp_mantissa=false>
inline uint8_t convert_u32_to_bfp(uint32_t input, uint32_t shared_exp, bool is_exp_a) {

    TT_ASSERT(
        BfpFormat == tt::DataFormat::Bfp2   ||
        BfpFormat == tt::DataFormat::Bfp4   ||
        BfpFormat == tt::DataFormat::Bfp8   ||
        BfpFormat == tt::DataFormat::Bfp2_b ||
        BfpFormat == tt::DataFormat::Bfp4_b ||
        BfpFormat == tt::DataFormat::Bfp8_b
    );

    constexpr uint32_t MANTISSA_BFP_WIDTH =
        (BfpFormat == tt::DataFormat::Bfp2 || BfpFormat == tt::DataFormat::Bfp2_b) ? 1 :
        (BfpFormat == tt::DataFormat::Bfp4 || BfpFormat == tt::DataFormat::Bfp4_b) ? 3 : 7;
    constexpr uint32_t MANTISSA_BFP_SHIFT = 24 - MANTISSA_BFP_WIDTH;
    constexpr uint32_t MANTISSA_BFP_MAX_VAL = (1 << MANTISSA_BFP_WIDTH) - 1;

    //check for both +/- 0.0
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
        }
        else if (se < 0) {
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
        // In bit-shift operation (A >> B), the result is undefined if B is greater than or equal to the number of bits in A
        while (exp_diff > 31) {
            mantissa = mantissa >> 31;
            exp_diff -= 31;
        }
        mantissa = mantissa >> exp_diff;
    }

    // this needs to become 3 bits so shift 21 times
    if (truncate_bfp_mantissa) {
        // Truncation: Round down
        mantissa = mantissa >> MANTISSA_BFP_SHIFT;
    }
    else {
        // Round mantissa to nearest even
        mantissa += 1 << (MANTISSA_BFP_SHIFT-1);
        mantissa = mantissa >> MANTISSA_BFP_SHIFT;
        if(mantissa > MANTISSA_BFP_MAX_VAL) mantissa = MANTISSA_BFP_MAX_VAL;
    }

    // add sign bit only if result is not 0
    if (0 == mantissa) {
        sign = 0;
    }
    mantissa = (sign << MANTISSA_BFP_WIDTH) | mantissa;
    return mantissa;
}

template <tt::DataFormat BfpFormat>
inline uint32_t create_packed_bfp_packed_as_u32(const std::vector<uint32_t> &u32_vec, uint32_t shared_exp, bool is_exp_a) {
    TT_ASSERT(
        BfpFormat == tt::DataFormat::Bfp2   ||
        BfpFormat == tt::DataFormat::Bfp4   ||
        BfpFormat == tt::DataFormat::Bfp8   ||
        BfpFormat == tt::DataFormat::Bfp2_b ||
        BfpFormat == tt::DataFormat::Bfp4_b ||
        BfpFormat == tt::DataFormat::Bfp8_b
    );
    constexpr int nums_in_dword =
        (BfpFormat == tt::DataFormat::Bfp2 || BfpFormat == tt::DataFormat::Bfp2_b) ? 16 :
        (BfpFormat == tt::DataFormat::Bfp4 || BfpFormat == tt::DataFormat::Bfp4_b) ? 8 : 4;

    uint32_t tmp_o = 0;
    uint32_t mask = (1 << (32 / nums_in_dword)) - 1;
    for (int i = nums_in_dword - 1; i >= 0; --i) // [0] in LSBs of dword
    {
            uint32_t conv_num = convert_u32_to_bfp<BfpFormat, false>(u32_vec[i], shared_exp, is_exp_a);
            tmp_o = tmp_o << (32 / nums_in_dword);
            tmp_o = tmp_o | (conv_num & mask);
    }
    return tmp_o;
}

template <tt::DataFormat BfpFormat>
inline std::vector<uint32_t> pack_fp32_vec_as_bfp_tiles(const std::vector<float> &fp32_vec, bool row_major_input, bool is_exp_a) {
    ZoneScoped;

    TT_ASSERT(
        BfpFormat == tt::DataFormat::Bfp2   ||
        BfpFormat == tt::DataFormat::Bfp4   ||
        BfpFormat == tt::DataFormat::Bfp8   ||
        BfpFormat == tt::DataFormat::Bfp2_b ||
        BfpFormat == tt::DataFormat::Bfp4_b ||
        BfpFormat == tt::DataFormat::Bfp8_b
    );

    uint32_t single_bfp8_tile_size = tile_size(BfpFormat);
    int num_float_in_tile = 1024;
    TT_ASSERT(fp32_vec.size() % num_float_in_tile == 0);
    uint32_t num_tiles = fp32_vec.size() / num_float_in_tile;

    std::vector<uint32_t> packed_result;

    std::vector<uint8_t> exponents;
    std::vector<uint32_t> data;

    int subtiles_in_tile_row = 2;
    int subtiles_in_tile_col = 2;
    int subtile_rows = 16;
    int subtile_cols = 16;
    int num_exponents_in_dword = 4;
    int num_mantissas_in_dword =
        (BfpFormat == tt::DataFormat::Bfp2 || BfpFormat == tt::DataFormat::Bfp2_b) ? 16 :
        (BfpFormat == tt::DataFormat::Bfp4 || BfpFormat == tt::DataFormat::Bfp4_b) ? 8 : 4;
    int fp32_element_index = 0;
    for (int tile_index = 0; tile_index < num_tiles; ++tile_index) {
        std::vector<uint32_t> packed_data;
        for (int tr = 0; tr < subtiles_in_tile_row; ++tr) {
            for (int tc = 0; tc < subtiles_in_tile_col; ++tc) {
                for (int i = 0; i < subtile_rows; ++i) {
                    std::vector<uint32_t> single_row;
                    // populate a single row
                    for (int j = 0; j < subtile_cols; ++j) {
                        int data_index;
                        if (row_major_input) {
                            data_index = (tr*16 + i)*32 + (tc*16 + j) + (num_float_in_tile * tile_index);
                        } else {
                            data_index = fp32_element_index++;
                        }
                        float float_num = fp32_vec.at(data_index);
                        uint32_t uint32_num = *reinterpret_cast<uint32_t*>(&float_num);
                        single_row.push_back(uint32_num);
                    }

                    uint8_t exp = get_max_exp(single_row, is_exp_a);
                    exponents.push_back(exp);

                    if (exponents.size() % num_exponents_in_dword == 0) {
                        packed_result.push_back(get_exp_dword(exponents));
                        exponents.clear();
                    }

                    for (uint32_t u32_datum : single_row) {
                        data.push_back(u32_datum);
                        if (data.size() % num_mantissas_in_dword == 0) {
                            uint32_t datum = create_packed_bfp_packed_as_u32<BfpFormat>(data, exp, is_exp_a);
                            packed_data.push_back(datum);
                            data.clear();
                        }
                    }
                }
            }
        }
        // prepend exponents to follow data packing order:
        //  16 exponents for sub-tile 0​
        //      exp_row0, exp_row1, … exp_row15​
        //  16 exponents for sub-tile 1​
        //  16 exponents for sub-tile 2​
        //  16 exponents for sub-tile 3​
        //  entire sub-tile 0 (RM layout)​
        //  entire sub-tile 1 (RM layout)​
        //  entire sub-tile 2 (RM layout)​
        //  entire sub-tile 3 (RM layout)
        packed_result.insert(packed_result.end(), packed_data.begin(), packed_data.end());
    }

    return packed_result;
}
