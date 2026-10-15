// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/span.hpp>
#include <array>
#include <bit>
#include <cstring>
#include <vector>
#include <thread>
#include <future>
#include <simde/x86/avx2.h>

#include <tt_stl/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include "blockfloat_common.hpp"
#include "common/executor.hpp"
#include "constants.hpp"
#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include "math.hpp"
#include "tile.hpp"
#include "tt_metal/tools/profiler/tracy_debug_zones.hpp"
#include "tt_backend_api_types.hpp"

namespace {

uint8_t get_max_exp(const std::vector<uint32_t>& vec, bool is_exp_a) {
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
            } else if (se < 0) {
                se = 0;
            }

            exp = static_cast<uint32_t>(se);
        }

        max = std::max(exp, max);
    }
    return max;
}

uint32_t get_exp_dword(const std::vector<uint8_t>& vec) {
    TT_ASSERT(vec.size() == 4);

    uint32_t tmp = 0;
    for (int i = 0; i < 4; ++i) {
        tmp = tmp | ((vec[i] & 0xff) << (i * 8));
    }
    return tmp;
}

std::vector<uint32_t> pack_exponents(const std::vector<uint8_t>& exponents, size_t num_elements_in_dword) {
    TT_FATAL(
        exponents.size() % num_elements_in_dword == 0,
        "Input vector size {} must be divisible by num_elements_in_dword",
        exponents.size());

    std::vector<uint32_t> packed_result;
    packed_result.reserve(exponents.size() / num_elements_in_dword);

    for (size_t i = 0; i < exponents.size(); i += num_elements_in_dword) {
        uint32_t packed_value = 0;

        for (size_t j = 0; j < num_elements_in_dword; ++j) {
            packed_value = packed_value | ((exponents[i + j] & 0xff) << (8 * j));
        }

        packed_result.push_back(packed_value);
    }

    return packed_result;
}

template <tt::DataFormat BfpFormat>
uint32_t create_packed_bfp_packed_as_u32(const std::vector<uint32_t>& u32_vec, uint32_t shared_exp, bool is_exp_a) {
    TT_ASSERT(
        BfpFormat == tt::DataFormat::Bfp2 || BfpFormat == tt::DataFormat::Bfp4 || BfpFormat == tt::DataFormat::Bfp8 ||
        BfpFormat == tt::DataFormat::Bfp2_b || BfpFormat == tt::DataFormat::Bfp4_b ||
        BfpFormat == tt::DataFormat::Bfp8_b);
    constexpr int nums_in_dword = []() {
        if constexpr (BfpFormat == tt::DataFormat::Bfp2 || BfpFormat == tt::DataFormat::Bfp2_b) {
            return 16;
        } else if constexpr (BfpFormat == tt::DataFormat::Bfp4 || BfpFormat == tt::DataFormat::Bfp4_b) {
            return 8;
        } else {
            return 4;
        }
    }();

    uint32_t tmp_o = 0;
    uint32_t mask = (1 << (32 / nums_in_dword)) - 1;
    for (int i = nums_in_dword - 1; i >= 0; --i)  // [0] in LSBs of dword
    {
        uint32_t conv_num = convert_u32_to_bfp<BfpFormat, false>(u32_vec[i], shared_exp, is_exp_a);
        tmp_o = tmp_o << (32 / nums_in_dword);
        tmp_o = tmp_o | (conv_num & mask);
    }
    return tmp_o;
}

template uint32_t create_packed_bfp_packed_as_u32<tt::DataFormat::Bfp2>(
    const std::vector<uint32_t>& u32_vec, uint32_t shared_exp, bool is_exp_a);
template uint32_t create_packed_bfp_packed_as_u32<tt::DataFormat::Bfp4>(
    const std::vector<uint32_t>& u32_vec, uint32_t shared_exp, bool is_exp_a);
template uint32_t create_packed_bfp_packed_as_u32<tt::DataFormat::Bfp8>(
    const std::vector<uint32_t>& u32_vec, uint32_t shared_exp, bool is_exp_a);
template uint32_t create_packed_bfp_packed_as_u32<tt::DataFormat::Bfp2_b>(
    const std::vector<uint32_t>& u32_vec, uint32_t shared_exp, bool is_exp_a);
template uint32_t create_packed_bfp_packed_as_u32<tt::DataFormat::Bfp4_b>(
    const std::vector<uint32_t>& u32_vec, uint32_t shared_exp, bool is_exp_a);
template uint32_t create_packed_bfp_packed_as_u32<tt::DataFormat::Bfp8_b>(
    const std::vector<uint32_t>& u32_vec, uint32_t shared_exp, bool is_exp_a);

// Vectorized 16-element subtile row packer for BFP8/BFP4/BFP2 (#8621).
// Computes the shared exponent across 16 FP32 bit patterns (lo8, hi8) and converts/packs all 16
// elements with round-to-nearest-even in SIMD, matching convert_u32_to_bfp bit-for-bit.
template <tt::DataFormat BfpFormat, bool IsExpA>
inline void pack_row16_simd(simde__m256i lo8, simde__m256i hi8, uint8_t* out_exp, uint32_t* out_mantissa) {
    constexpr uint32_t MANTISSA_BFP_WIDTH =
        (BfpFormat == tt::DataFormat::Bfp2 || BfpFormat == tt::DataFormat::Bfp2_b) ? 1 :
        (BfpFormat == tt::DataFormat::Bfp4 || BfpFormat == tt::DataFormat::Bfp4_b) ? 3 : 7;
    constexpr uint32_t MANTISSA_BFP_SHIFT = 24 - MANTISSA_BFP_WIDTH;
    constexpr uint32_t MANTISSA_BFP_MAX_VAL = (1 << MANTISSA_BFP_WIDTH) - 1;
    constexpr uint32_t MANTISSA_ROUND_MASK = (1 << MANTISSA_BFP_SHIFT) - 1;
    constexpr uint32_t TIE_VALUE = 1 << (MANTISSA_BFP_SHIFT - 1);

    const simde__m256i zero = simde_mm256_setzero_si256();
    const simde__m256i exp_mask = simde_mm256_set1_epi32(0x7f800000);
    const simde__m256i man_mask = simde_mm256_set1_epi32(0x007fffff);

    simde__m256i exp_lo = simde_mm256_srli_epi32(simde_mm256_and_si256(lo8, exp_mask), 23);
    simde__m256i exp_hi = simde_mm256_srli_epi32(simde_mm256_and_si256(hi8, exp_mask), 23);
    simde__m256i is_zero_lo = simde_mm256_cmpeq_epi32(exp_lo, zero);
    simde__m256i is_zero_hi = simde_mm256_cmpeq_epi32(exp_hi, zero);

    simde__m256i man_lo = simde_mm256_and_si256(lo8, man_mask);
    simde__m256i man_hi = simde_mm256_and_si256(hi8, man_mask);

    if constexpr (IsExpA) {
        const simde__m256i bias_diff = simde_mm256_set1_epi32(112);
        const simde__m256i max31 = simde_mm256_set1_epi32(31);
        simde__m256i se_lo = simde_mm256_sub_epi32(exp_lo, bias_diff);
        simde__m256i se_hi = simde_mm256_sub_epi32(exp_hi, bias_diff);
        man_lo = simde_mm256_or_si256(man_lo, simde_mm256_and_si256(simde_mm256_cmpgt_epi32(se_lo, max31), man_mask));
        man_hi = simde_mm256_or_si256(man_hi, simde_mm256_and_si256(simde_mm256_cmpgt_epi32(se_hi, max31), man_mask));
        man_lo = simde_mm256_andnot_si256(simde_mm256_cmpgt_epi32(zero, se_lo), man_lo);
        man_hi = simde_mm256_andnot_si256(simde_mm256_cmpgt_epi32(zero, se_hi), man_hi);
        exp_lo = simde_mm256_min_epi32(simde_mm256_max_epi32(se_lo, zero), max31);
        exp_hi = simde_mm256_min_epi32(simde_mm256_max_epi32(se_hi, zero), max31);
    }

    simde__m256i max8 = simde_mm256_max_epu32(exp_lo, exp_hi);
    simde__m128i max4 = simde_mm_max_epu32(simde_mm256_castsi256_si128(max8), simde_mm256_extracti128_si256(max8, 1));
    max4 = simde_mm_max_epu32(max4, simde_mm_shuffle_epi32(max4, 0x4E));
    max4 = simde_mm_max_epu32(max4, simde_mm_shuffle_epi32(max4, 0xB1));
    uint8_t shared_exp = static_cast<uint8_t>(simde_mm_cvtsi128_si32(max4));
    *out_exp = shared_exp;

    const simde__m256i s_exp = simde_mm256_set1_epi32(static_cast<int32_t>(shared_exp));
    const simde__m256i hidden_bit = simde_mm256_set1_epi32(1 << 23);
    const simde__m256i shift_cap = simde_mm256_set1_epi32(31);
    const simde__m256i r_mask = simde_mm256_set1_epi32(MANTISSA_ROUND_MASK);
    const simde__m256i one = simde_mm256_set1_epi32(1);
    const simde__m256i tie = simde_mm256_set1_epi32(TIE_VALUE);
    const simde__m256i max_val = simde_mm256_set1_epi32(MANTISSA_BFP_MAX_VAL);

    auto convert_half = [&](simde__m256i v, simde__m256i man, simde__m256i exp, simde__m256i is_zero) {
        man = simde_mm256_or_si256(man, hidden_bit);
        simde__m256i diff = simde_mm256_min_epu32(simde_mm256_sub_epi32(s_exp, exp), shift_cap);
        man = simde_mm256_srlv_epi32(man, diff);
        simde__m256i round_val = simde_mm256_and_si256(man, r_mask);
        man = simde_mm256_srli_epi32(man, MANTISSA_BFP_SHIFT);
        simde__m256i guard = simde_mm256_and_si256(man, one);
        simde__m256i rup = simde_mm256_cmpgt_epi32(simde_mm256_add_epi32(round_val, guard), tie);
        man = simde_mm256_min_epu32(simde_mm256_sub_epi32(man, rup), max_val);
        man = simde_mm256_andnot_si256(is_zero, man);
        simde__m256i sign = simde_mm256_slli_epi32(simde_mm256_srli_epi32(v, 31), MANTISSA_BFP_WIDTH);
        sign = simde_mm256_andnot_si256(simde_mm256_cmpeq_epi32(man, zero), sign);
        return simde_mm256_or_si256(sign, man);
    };

    simde__m256i res_lo = convert_half(lo8, man_lo, exp_lo, is_zero_lo);
    simde__m256i res_hi = convert_half(hi8, man_hi, exp_hi, is_zero_hi);

    simde__m256i p16 = simde_mm256_packus_epi32(res_lo, res_hi);
    simde__m256i p8 = simde_mm256_packus_epi16(p16, p16);
    uint64_t lane0 = static_cast<uint64_t>(simde_mm_cvtsi128_si64(simde_mm256_castsi256_si128(p8)));
    uint64_t lane1 = static_cast<uint64_t>(simde_mm_cvtsi128_si64(simde_mm256_extracti128_si256(p8, 1)));

    if constexpr (BfpFormat == tt::DataFormat::Bfp8 || BfpFormat == tt::DataFormat::Bfp8_b) {
        out_mantissa[0] = static_cast<uint32_t>(lane0);
        out_mantissa[1] = static_cast<uint32_t>(lane1);
        out_mantissa[2] = static_cast<uint32_t>(lane0 >> 32);
        out_mantissa[3] = static_cast<uint32_t>(lane1 >> 32);
    } else if constexpr (BfpFormat == tt::DataFormat::Bfp4 || BfpFormat == tt::DataFormat::Bfp4_b) {
        auto pack8_nibbles = [](uint64_t w) -> uint32_t {
            w = (w | (w >> 4)) & 0x00ff00ff00ff00ffULL;
            w = (w | (w >> 8)) & 0x0000ffff0000ffffULL;
            return static_cast<uint32_t>(w | (w >> 16));
        };
        uint64_t w_lo = (lane0 & 0xffffffffULL) | (lane1 << 32);
        uint64_t w_hi = (lane0 >> 32) | (lane1 & 0xffffffff00000000ULL);
        out_mantissa[0] = pack8_nibbles(w_lo);
        out_mantissa[1] = pack8_nibbles(w_hi);
    } else {
        auto pack8_crumbs = [](uint64_t w) -> uint32_t {
            w = (w | (w >> 6)) & 0x000f000f000f000fULL;
            w = (w | (w >> 12)) & 0x000000ff000000ffULL;
            return static_cast<uint32_t>((w | (w >> 24)) & 0xffffULL);
        };
        uint64_t w_lo = (lane0 & 0xffffffffULL) | (lane1 << 32);
        uint64_t w_hi = (lane0 >> 32) | (lane1 & 0xffffffff00000000ULL);
        out_mantissa[0] = pack8_crumbs(w_lo) | (pack8_crumbs(w_hi) << 16);
    }
}

template <tt::DataFormat BfpFormat, typename T, bool RowMajorInput, bool IsExpA>
void process_tile_range_impl(
    const T* input_data,
    int start_tile,
    int end_tile,
    int tile_H,
    int tile_W,
    int face_H,
    int face_W,
    int exp_dwords_per_tile,
    size_t dwords_per_tile,
    uint32_t* packed_out) {
    const int num_float_in_tile = tile_H * tile_W;
    const int subtiles_in_tile_row = tile_H / face_H;
    const int subtiles_in_tile_col = tile_W / face_W;
    const int subtile_rows = face_H;
    const int subtile_cols = face_W;
    constexpr int num_mantissas_in_dword =
        (BfpFormat == tt::DataFormat::Bfp2 || BfpFormat == tt::DataFormat::Bfp2_b) ? 16 :
        (BfpFormat == tt::DataFormat::Bfp4 || BfpFormat == tt::DataFormat::Bfp4_b) ? 8 : 4;
    const int dwords_per_row = subtile_cols / num_mantissas_in_dword;

    alignas(16) std::array<uint8_t, 128> exp_bytes{};

    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, bfloat16>) {
        if (subtile_cols == 16) {
            for (int tile_index = start_tile; tile_index < end_tile; ++tile_index) {
                uint32_t* tile_out = packed_out + static_cast<size_t>(tile_index) * dwords_per_tile;
                uint32_t* mantissa_out = tile_out + exp_dwords_per_tile;
                size_t fp32_element_base = RowMajorInput ? 0 : (static_cast<size_t>(tile_index) * num_float_in_tile);
                int row_idx = 0;

                for (int tr = 0; tr < subtiles_in_tile_row; ++tr) {
                    for (int tc = 0; tc < subtiles_in_tile_col; ++tc) {
                        for (int i = 0; i < subtile_rows; ++i, ++row_idx) {
                            size_t row_base = RowMajorInput
                                ? static_cast<size_t>(tr * face_H + i) * tile_W + (tc * face_W) +
                                      (static_cast<size_t>(num_float_in_tile) * tile_index)
                                : fp32_element_base +
                                      static_cast<size_t>(tr * subtiles_in_tile_col + tc) *
                                          (subtile_rows * subtile_cols) +
                                      i * subtile_cols;
                            const T* row_ptr = input_data + row_base;
                            simde__m256i lo8, hi8;
                            if constexpr (std::is_same_v<T, float>) {
                                lo8 = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(row_ptr));
                                hi8 = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(row_ptr + 8));
                            } else {
                                simde__m256i raw =
                                    simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(row_ptr));
                                lo8 = simde_mm256_slli_epi32(
                                    simde_mm256_cvtepu16_epi32(simde_mm256_castsi256_si128(raw)), 16);
                                hi8 = simde_mm256_slli_epi32(
                                    simde_mm256_cvtepu16_epi32(simde_mm256_extracti128_si256(raw, 1)), 16);
                            }
                            pack_row16_simd<BfpFormat, IsExpA>(
                                lo8, hi8, &exp_bytes[row_idx], mantissa_out + row_idx * dwords_per_row);
                        }
                    }
                }
                std::memcpy(tile_out, exp_bytes.data(), static_cast<size_t>(exp_dwords_per_tile) * sizeof(uint32_t));
            }
            return;
        }
    }

    // Zero-allocation scalar path for integer types or non-16 subtile_cols
    std::array<uint32_t, 64> single_row{};
    constexpr uint32_t mask = (1u << (32 / num_mantissas_in_dword)) - 1u;
    for (int tile_index = start_tile; tile_index < end_tile; ++tile_index) {
        uint32_t* tile_out = packed_out + static_cast<size_t>(tile_index) * dwords_per_tile;
        uint32_t* mantissa_out = tile_out + exp_dwords_per_tile;
        size_t fp32_element_base = RowMajorInput ? 0 : (static_cast<size_t>(tile_index) * num_float_in_tile);
        int row_idx = 0;
        size_t mantissa_slot = 0;

        for (int tr = 0; tr < subtiles_in_tile_row; ++tr) {
            for (int tc = 0; tc < subtiles_in_tile_col; ++tc) {
                for (int i = 0; i < subtile_rows; ++i, ++row_idx) {
                    size_t row_base = RowMajorInput
                        ? static_cast<size_t>(tr * face_H + i) * tile_W + (tc * face_W) +
                              (static_cast<size_t>(num_float_in_tile) * tile_index)
                        : fp32_element_base +
                              static_cast<size_t>(tr * subtiles_in_tile_col + tc) * (subtile_rows * subtile_cols) +
                              i * subtile_cols;
                    const T* row_ptr = input_data + row_base;
                    uint32_t max_e = 0;
                    for (int j = 0; j < subtile_cols; ++j) {
                        uint32_t u32_num;
                        if constexpr (std::is_same_v<T, float>) {
                            u32_num = std::bit_cast<uint32_t>(row_ptr[j]);
                        } else if constexpr (std::is_same_v<T, bfloat16>) {
                            u32_num = static_cast<uint32_t>(std::bit_cast<uint16_t>(row_ptr[j])) << 16;
                        } else {
                            float f = static_cast<float>(row_ptr[j]);
                            u32_num = std::bit_cast<uint32_t>(f);
                        }
                        single_row[j] = u32_num;
                        if (j < 16) {
                            uint32_t e = (u32_num & 0x7f800000) >> 23;
                            if constexpr (IsExpA) {
                                int32_t se = std::clamp(static_cast<int32_t>(e) - 112, 0, 31);
                                e = static_cast<uint32_t>(se);
                            }
                            max_e = std::max(e, max_e);
                        }
                    }
                    uint8_t exp = static_cast<uint8_t>(max_e);
                    exp_bytes[row_idx] = exp;
                    for (int j = 0; j < subtile_cols; j += num_mantissas_in_dword) {
                        uint32_t tmp_o = 0;
                        for (int k = num_mantissas_in_dword - 1; k >= 0; --k) {
                            uint32_t conv = convert_u32_to_bfp<BfpFormat, false>(single_row[j + k], exp, IsExpA);
                            tmp_o = (tmp_o << (32 / num_mantissas_in_dword)) | (conv & mask);
                        }
                        mantissa_out[mantissa_slot++] = tmp_o;
                    }
                }
            }
        }
        std::memcpy(tile_out, exp_bytes.data(), static_cast<size_t>(exp_dwords_per_tile) * sizeof(uint32_t));
    }
}

}  // namespace

uint32_t get_byte(uint32_t word, uint32_t index) {
    TT_ASSERT(index < 4);
    uint32_t mask = 0xff << (8 * index);
    uint32_t masked = word & mask;
    masked = masked >> (8 * index);
    return masked;
}

uint32_t convert_bfp_to_u32(tt::DataFormat bfp_format, uint8_t data, uint8_t shared_exp, bool is_exp_a) {
    uint32_t exp = shared_exp;
    uint32_t out_num = 0;
    if ((bfp_format == tt::DataFormat::Bfp2_b) || (bfp_format == tt::DataFormat::Bfp2)) {
        uint32_t sign = data >> 1;
        uint32_t man = data & 0x1;

        // Shift mantissa up until there is a 1 in bit 1
        int shift_cnt = 0;
        if (man == 0) {
            man = 0;
            exp = 0;
        } else {
            // shift again to put first non-hidden mantissa
            // bit in bit 1
            man = man << 1;
            man = man & 0x1;

            // adjust exponent
            TT_ASSERT(exp >= (uint32_t)shift_cnt, "incorrect shift_cnt");
            exp = exp - shift_cnt;

            // if exp_a rebias exp to 127
            if (is_exp_a) {
                exp = exp - 15 + 127;
            }
        }

        // put s, e, m together
        out_num = (sign << 31) | (exp << 23) | (man << 22);
    } else if ((bfp_format == tt::DataFormat::Bfp4_b) || (bfp_format == tt::DataFormat::Bfp4)) {
        uint32_t sign = data >> 3;
        uint32_t man = data & 0x7;

        // Shift mantissa up until there is a 1 in bit 3
        int shift_cnt = 0;
        if (man == 0) {
            man = 0;
            exp = 0;
        } else {
            while ((man & 0x04) == 0) {
                man = man << 1;
                shift_cnt++;
            }
            // shift one more time and zero the
            // hidden top mantissa bit
            // shift again to put first non-hidden mantissa
            // bit in bit 3
            man = man << 1;
            man = man & 0x7;

            // adjust exponent
            TT_ASSERT(exp >= (uint32_t)shift_cnt, "incorrect shift_cnt");
            exp = exp - shift_cnt;

            // if exp_a rebias exp to 127
            if (is_exp_a) {
                exp = exp - 15 + 127;
            }
        }

        // put s, e, m together
        out_num = (sign << 31) | (exp << 23) | (man << 20);
    } else if ((bfp_format == tt::DataFormat::Bfp8_b) || (bfp_format == tt::DataFormat::Bfp8)) {
        uint32_t sign = data >> 7;
        uint32_t man = data & 0x7f;

        // Shift mantissa up until there is a 1 in bit 6
        int shift_cnt = 0;
        if (man == 0) {
            man = 0;
            exp = 0;
        } else {
            while ((man & 0x40) == 0) {
                man = man << 1;
                shift_cnt++;
            }
            // shift one more time and zero the
            // hidden top mantissa bit
            // shift again to put first non-hidden mantissa
            // bit in bit 7
            man = man << 1;
            man = man & 0x7f;

            // adjust exponent
            TT_ASSERT(exp >= (uint32_t)shift_cnt, "incorrect shift_cnt");
            exp = exp - shift_cnt;

            // if exp_a rebias exp to 127
            if (is_exp_a) {
                exp = exp - 15 + 127;
            }
        }

        // put s, e, m together
        out_num = (sign << 31) | (exp << 23) | (man << 16);
    }
    return out_num;
}

template <tt::DataFormat BfpFormat, bool truncate_bfp_mantissa>
uint8_t convert_u32_to_bfp(uint32_t input, uint32_t shared_exp, bool is_exp_a) {
    TT_ASSERT(
        BfpFormat == tt::DataFormat::Bfp2 || BfpFormat == tt::DataFormat::Bfp4 || BfpFormat == tt::DataFormat::Bfp8 ||
        BfpFormat == tt::DataFormat::Bfp2_b || BfpFormat == tt::DataFormat::Bfp4_b ||
        BfpFormat == tt::DataFormat::Bfp8_b);

    constexpr uint32_t MANTISSA_BFP_WIDTH = []() {
        if constexpr (BfpFormat == tt::DataFormat::Bfp2 || BfpFormat == tt::DataFormat::Bfp2_b) {
            return 1;
        } else if constexpr (BfpFormat == tt::DataFormat::Bfp4 || BfpFormat == tt::DataFormat::Bfp4_b) {
            return 3;
        } else {
            return 7;
        }
    }();
    constexpr uint32_t MANTISSA_BFP_SHIFT = 24 - MANTISSA_BFP_WIDTH;
    constexpr uint32_t MANTISSA_BFP_MAX_VAL = (1 << MANTISSA_BFP_WIDTH) - 1;

    uint32_t mantissa = input & 0x007fffff;
    uint32_t exp = (input & 0x7f800000) >> 23;
    uint32_t sign = (input & 0x80000000) >> 31;

    // check for both +/- 0.0 or +/- denormal
    bool is_zero_or_denormal = (exp == 0);

    if (is_zero_or_denormal) {
        return 0;
    }

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

    if (shared_exp > exp) {
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

    // this needs to become 3 bits so shift 21 times
    if (truncate_bfp_mantissa) {
        // Truncation: Round down
        mantissa = mantissa >> MANTISSA_BFP_SHIFT;
    } else {
        // Round mantissa to nearest; ties round to even
        constexpr uint32_t MANTISSA_ROUND_MASK = (1 << MANTISSA_BFP_SHIFT) - 1;
        constexpr uint32_t TIE_VALUE = 1 << (MANTISSA_BFP_SHIFT - 1);
        uint32_t round_value = mantissa & MANTISSA_ROUND_MASK;
        mantissa = mantissa >> MANTISSA_BFP_SHIFT;
        uint32_t guard_bit = mantissa & 0x1;

        if (round_value > TIE_VALUE or (round_value == TIE_VALUE and guard_bit == 1)) {
            // Round up
            mantissa += 1;
        }

        mantissa = std::min(mantissa, MANTISSA_BFP_MAX_VAL);
    }

    // add sign bit only if result is not 0
    if (0 == mantissa) {
        sign = 0;
    }
    mantissa = (sign << MANTISSA_BFP_WIDTH) | mantissa;
    return mantissa;
}

template <tt::DataFormat BfpFormat>
std::vector<uint32_t> pack_fp32_vec_as_bfp_tiles(
    ttsl::Span<const float> fp32_vec,
    bool row_major_input,
    bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile) {
    return pack_as_bfp_tiles<BfpFormat, float>(fp32_vec, row_major_input, is_exp_a, tile);
}

template <tt::DataFormat BfpFormat, typename T>
std::vector<uint32_t> pack_as_bfp_tiles(
    ttsl::Span<const T> input_data,
    bool row_major_input,
    bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile) {
    TTZoneScopedD(DATA_FORMAT);

    TT_ASSERT(
        BfpFormat == tt::DataFormat::Bfp2 || BfpFormat == tt::DataFormat::Bfp4 || BfpFormat == tt::DataFormat::Bfp8 ||
        BfpFormat == tt::DataFormat::Bfp2_b || BfpFormat == tt::DataFormat::Bfp4_b ||
        BfpFormat == tt::DataFormat::Bfp8_b);

    auto tile_H = tile.has_value() ? tile->get_tile_shape()[0] : tt::constants::TILE_HEIGHT;
    auto tile_W = tile.has_value() ? tile->get_tile_shape()[1] : tt::constants::TILE_WIDTH;
    auto face_H = tile.has_value() ? tile->get_face_shape()[0] : tt::constants::FACE_HEIGHT;
    auto face_W = tile.has_value() ? tile->get_face_shape()[1] : tt::constants::FACE_WIDTH;
    auto tile_HW = tile_H * tile_W;
    auto subtiles_in_tile_row = tile_H / face_H;
    auto subtiles_in_tile_col = tile_W / face_W;
    auto subtile_rows = face_H;

    uint32_t l1_alignment = tt::tt_metal::MetalContext::instance().hal().get_alignment(tt::tt_metal::HalMemType::L1);
    bool exponent_padding = (subtile_rows * subtiles_in_tile_col * subtiles_in_tile_row) < l1_alignment;

    int num_float_in_tile = tile_HW;
    TT_ASSERT(input_data.size() % num_float_in_tile == 0);
    uint32_t num_tiles = input_data.size() / num_float_in_tile;
    if (num_tiles == 0) {
        return {};
    }

    constexpr int num_exponents_in_dword = 4;
    constexpr int num_mantissas_in_dword = []() {
        if constexpr (BfpFormat == tt::DataFormat::Bfp2 || BfpFormat == tt::DataFormat::Bfp2_b) {
            return 16;
        } else if constexpr (BfpFormat == tt::DataFormat::Bfp4 || BfpFormat == tt::DataFormat::Bfp4_b) {
            return 8;
        } else {
            return 4;
        }
    }();

    const int rows_per_tile = subtiles_in_tile_row * subtiles_in_tile_col * subtile_rows;
    const int mantissa_dwords_per_tile = num_float_in_tile / num_mantissas_in_dword;
    const int exp_dwords_per_tile =
        exponent_padding ? static_cast<int>(tt::round_up(static_cast<uint32_t>(rows_per_tile), l1_alignment)) /
                               num_exponents_in_dword
                         : rows_per_tile / num_exponents_in_dword;
    const size_t dwords_per_tile = static_cast<size_t>(exp_dwords_per_tile + mantissa_dwords_per_tile);

    // Pre-allocate the entire output buffer once so worker chunks write directly to their disjoint
    // tile slices without per-chunk vectors or a final concatenation copy (#8621).
    std::vector<uint32_t> packed_result(static_cast<size_t>(num_tiles) * dwords_per_tile);
    uint32_t* packed_out = packed_result.data();
    const T* raw_in = input_data.data();

    auto process_tile_range = [&](int start_tile, int end_tile) {
        if (row_major_input) {
            if (is_exp_a) {
                process_tile_range_impl<BfpFormat, T, true, true>(
                    raw_in, start_tile, end_tile, tile_H, tile_W, face_H, face_W, exp_dwords_per_tile, dwords_per_tile, packed_out);
            } else {
                process_tile_range_impl<BfpFormat, T, true, false>(
                    raw_in, start_tile, end_tile, tile_H, tile_W, face_H, face_W, exp_dwords_per_tile, dwords_per_tile, packed_out);
            }
        } else {
            if (is_exp_a) {
                process_tile_range_impl<BfpFormat, T, false, true>(
                    raw_in, start_tile, end_tile, tile_H, tile_W, face_H, face_W, exp_dwords_per_tile, dwords_per_tile, packed_out);
            } else {
                process_tile_range_impl<BfpFormat, T, false, false>(
                    raw_in, start_tile, end_tile, tile_H, tile_W, face_H, face_W, exp_dwords_per_tile, dwords_per_tile, packed_out);
            }
        }
    };

    // Determine how many parallel work items to split the tiles into.
    // Only parallelize if we have enough tiles to justify the dispatch overhead.
    constexpr uint32_t MIN_TILES_PER_CHUNK = 32;
    uint32_t max_chunks = std::thread::hardware_concurrency();
    if (max_chunks == 0) {
        max_chunks = 1;
    }
    uint32_t num_chunks = std::min(max_chunks, num_tiles / MIN_TILES_PER_CHUNK);
    num_chunks = std::max(1u, num_chunks);

    if (num_chunks == 1) {
        process_tile_range(0, num_tiles);
        return packed_result;
    }

    log_debug(
        tt::LogAlways,
        "Converting {} block-float tiles with {} chunks ({} tiles/chunk)",
        num_tiles,
        num_chunks,
        num_tiles / num_chunks);

    std::vector<std::shared_future<void>> futures;
    futures.reserve(num_chunks);

    uint32_t tiles_per_chunk = num_tiles / num_chunks;
    uint32_t remainder = num_tiles % num_chunks;

    uint32_t start_tile = 0;
    for (uint32_t t = 0; t < num_chunks; ++t) {
        uint32_t tiles_for_this_chunk = tiles_per_chunk + (t < remainder ? 1 : 0);
        uint32_t end_tile = start_tile + tiles_for_this_chunk;

        futures.emplace_back(tt::tt_metal::detail::async(
            [&process_tile_range, start_tile, end_tile]() { process_tile_range(start_tile, end_tile); }));

        start_tile = end_tile;
    }

    for (auto& future : futures) {
        future.get();
    }

    return packed_result;
}

// Explicit instantiations
// clang-format off

// truncate_bfp_mantissa = false
template uint8_t convert_u32_to_bfp<tt::DataFormat::Bfp2, false>(uint32_t input, uint32_t shared_exp, bool is_exp_a);
template uint8_t convert_u32_to_bfp<tt::DataFormat::Bfp4, false>(uint32_t input, uint32_t shared_exp, bool is_exp_a);
template uint8_t convert_u32_to_bfp<tt::DataFormat::Bfp8, false>(uint32_t input, uint32_t shared_exp, bool is_exp_a);
template uint8_t convert_u32_to_bfp<tt::DataFormat::Bfp2_b, false>(uint32_t input, uint32_t shared_exp, bool is_exp_a);
template uint8_t convert_u32_to_bfp<tt::DataFormat::Bfp4_b, false>(uint32_t input, uint32_t shared_exp, bool is_exp_a);
template uint8_t convert_u32_to_bfp<tt::DataFormat::Bfp8_b, false>(uint32_t input, uint32_t shared_exp, bool is_exp_a);

// truncate_bfp_mantissa = true
template uint8_t convert_u32_to_bfp<tt::DataFormat::Bfp2, true>(uint32_t input, uint32_t shared_exp, bool is_exp_a);
template uint8_t convert_u32_to_bfp<tt::DataFormat::Bfp4, true>(uint32_t input, uint32_t shared_exp, bool is_exp_a);
template uint8_t convert_u32_to_bfp<tt::DataFormat::Bfp8, true>(uint32_t input, uint32_t shared_exp, bool is_exp_a);
template uint8_t convert_u32_to_bfp<tt::DataFormat::Bfp2_b, true>(uint32_t input, uint32_t shared_exp, bool is_exp_a);
template uint8_t convert_u32_to_bfp<tt::DataFormat::Bfp4_b, true>(uint32_t input, uint32_t shared_exp, bool is_exp_a);
template uint8_t convert_u32_to_bfp<tt::DataFormat::Bfp8_b, true>(uint32_t input, uint32_t shared_exp, bool is_exp_a);

template std::vector<uint32_t> pack_fp32_vec_as_bfp_tiles<tt::DataFormat::Bfp2>(ttsl::Span<const float> fp32_vec, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_fp32_vec_as_bfp_tiles<tt::DataFormat::Bfp4>(ttsl::Span<const float> fp32_vec, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_fp32_vec_as_bfp_tiles<tt::DataFormat::Bfp8>(ttsl::Span<const float> fp32_vec, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_fp32_vec_as_bfp_tiles<tt::DataFormat::Bfp2_b>(ttsl::Span<const float> fp32_vec, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_fp32_vec_as_bfp_tiles<tt::DataFormat::Bfp4_b>(ttsl::Span<const float> fp32_vec, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_fp32_vec_as_bfp_tiles<tt::DataFormat::Bfp8_b>(ttsl::Span<const float> fp32_vec, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);

template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2>(ttsl::Span<const float> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4>(ttsl::Span<const float> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8>(ttsl::Span<const float> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2_b>(ttsl::Span<const float> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4_b>(ttsl::Span<const float> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8_b>(ttsl::Span<const float> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);

template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2>(ttsl::Span<const bfloat16> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4>(ttsl::Span<const bfloat16> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8>(ttsl::Span<const bfloat16> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2_b>(ttsl::Span<const bfloat16> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4_b>(ttsl::Span<const bfloat16> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8_b>(ttsl::Span<const bfloat16> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);

template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2>(ttsl::Span<const int32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4>(ttsl::Span<const int32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8>(ttsl::Span<const int32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2_b>(ttsl::Span<const int32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4_b>(ttsl::Span<const int32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8_b>(ttsl::Span<const int32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);

template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2>(ttsl::Span<const uint32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4>(ttsl::Span<const uint32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8>(ttsl::Span<const uint32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2_b>(ttsl::Span<const uint32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4_b>(ttsl::Span<const uint32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8_b>(ttsl::Span<const uint32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);


template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2>(ttsl::Span<const uint16_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4>(ttsl::Span<const uint16_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8>(ttsl::Span<const uint16_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2_b>(ttsl::Span<const uint16_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4_b>(ttsl::Span<const uint16_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8_b>(ttsl::Span<const uint16_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);


template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2>(ttsl::Span<const uint8_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4>(ttsl::Span<const uint8_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8>(ttsl::Span<const uint8_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2_b>(ttsl::Span<const uint8_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4_b>(ttsl::Span<const uint8_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8_b>(ttsl::Span<const uint8_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);

template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2>(ttsl::Span<const int8_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4>(ttsl::Span<const int8_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8>(ttsl::Span<const int8_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2_b>(ttsl::Span<const int8_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4_b>(ttsl::Span<const int8_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8_b>(ttsl::Span<const int8_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);

// clang-format on
