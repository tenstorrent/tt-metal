// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <tt_stl/span.hpp>
#include <array>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <type_traits>
#include <vector>
#include <simde/x86/avx2.h>

#include <tt_stl/assert.hpp>
#include "blockfloat_common.hpp"
#include "constants.hpp"
#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include "math.hpp"
#include "tile.hpp"
#include "tracy/Tracy.hpp"
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
            // shift_cnt = 6 - (31 - __builtin_clz(man));
            // man = (man << (shift_cnt + 1)) & 0x7f;
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
        // Implementation of rounding process (example is for bfp8):
        // - We want to round 23 bit mantissa to 6 bits with extra hidden bit
        // - Mantissa is broken down to: <5> bits | guard bit | <17> bits of round value
        //   * If round value < 0x10000, round down (ie. mantissa is just <5> bits | guard bit)
        //   * If round value > 0x10000, round up (ie. add 1 to <5> bits | guard bit)
        //   * If round value = 0x10000, we have a tie and round to nearest even:
        //     ** If guard bit = 0, mantissa is even so round down
        //     ** If guard bit = 1, mantissa is odd so round up
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
    tt::stl::Span<const float> fp32_vec,
    bool row_major_input,
    bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile) {
    return pack_as_bfp_tiles<BfpFormat, float>(fp32_vec, row_major_input, is_exp_a, tile);
}

// ---------------------------------------------------------------------------
// Helpers for the optimized BFP host tilizer.
//
// The original implementation used per-row std::vector allocations, scalar
// reduction, and serial tile processing. The new implementation:
//   * Pre-sizes the output exactly and writes tile data into a fixed slot.
//   * Uses small stack-allocated buffers per row (face_W is bounded).
//   * Processes tiles in parallel using std::thread for sufficiently large
//     inputs (each tile is fully independent in the standard production
//     layout where the per-tile exponent count is divisible by 4).
//   * Avoids the redundant float<->uint32_t round-trip for bfloat16 inputs.
//
// IMPORTANT: The packed bytes must be byte-exactly identical to what the
// legacy implementation produced. All rounding (tie-to-even) is delegated to
// the same convert_u32_to_bfp helper used by tests, and the layout (per-tile
// exponent block followed by mantissa data, with optional 16B padding) is
// preserved.
// ---------------------------------------------------------------------------

// Convert one input element to its IEEE-754 single-precision bit pattern.
// For bfloat16 we just shift the 16 bits up by 16 - this avoids the implicit
// float() round-trip that the legacy code used.
template <typename T>
inline uint32_t element_to_u32_bits(const T& v) {
    if constexpr (std::is_same_v<T, float>) {
        return std::bit_cast<uint32_t>(v);
    } else {
        float f = static_cast<float>(v);
        return std::bit_cast<uint32_t>(f);
    }
}

// Pack `count` exponent bytes (one per row) starting at `exp_bytes` into
// `count/4` little-endian uint32_t words at `out_dwords`. Caller guarantees
// `count % 4 == 0`.
//
// One bulk copy beats a loop of 4-byte memcpys: `exp_bytes` may be unaligned
// (it points into a stack buffer of bytes), so we can't legally reinterpret it
// as `const uint32_t*`, but a single `std::memcpy` of `count` bytes lowers to
// a tight rep-mov / SIMD copy with no per-iteration call overhead. The output
// dwords are little-endian-packed, matching the legacy byte-by-byte assembly
// on every supported (little-endian) host.
inline void pack_exp_bytes_to_dwords(const uint8_t* exp_bytes, size_t count, uint32_t* out_dwords) {
    std::memcpy(out_dwords, exp_bytes, count);
}

// ---------------------------------------------------------------------------
// SIMD (AVX2 via simde) fast paths.
//
// These are bit-exact reimplementations of the inner per-row work for the
// common production case: face_W == 16, BFP8/BFP8_b, and `is_exp_a == false`
// (the `_b` variants that skip the 5-bit exponent rebias). Anything outside
// this envelope falls back to the scalar inline code in `pack_one_tile`.
//
// The vectorisation strategy mirrors the one already used in
// `unpack_bfp8_tiles_into_float_vec` in `bfloat8.cpp`: load 8 fp32 values per
// AVX2 register, do bitwise/shift/blend operations lane-wise, then horizontal-
// reduce or pack at the end.
// ---------------------------------------------------------------------------

// Horizontal max of 8 epi32 lanes, returned as a uint32_t.
inline uint32_t simd_hmax_epi32(simde__m256i v) {
    simde__m128i lo = simde_mm256_castsi256_si128(v);
    simde__m128i hi = simde_mm256_extracti128_si256(v, 1);
    simde__m128i m = simde_mm_max_epu32(lo, hi);
    m = simde_mm_max_epu32(m, simde_mm_shuffle_epi32(m, 0x4E));  // swap halves
    m = simde_mm_max_epu32(m, simde_mm_shuffle_epi32(m, 0xB1));  // swap pairs
    return static_cast<uint32_t>(simde_mm_cvtsi128_si32(m));
}

// Compute shared exponent for 16 fp32 values. Pure (no rebias). Equivalent to
// scalar `get_max_exp(..., is_exp_a=false)` for 16 inputs.
inline uint8_t simd_get_max_exp_16_fp32_b(const uint32_t* row16) {
    const simde__m256i exp_mask = simde_mm256_set1_epi32(0x7f800000);
    simde__m256i lo = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(row16));
    simde__m256i hi = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(row16 + 8));
    lo = simde_mm256_srli_epi32(simde_mm256_and_si256(lo, exp_mask), 23);
    hi = simde_mm256_srli_epi32(simde_mm256_and_si256(hi, exp_mask), 23);
    simde__m256i max8 = simde_mm256_max_epu32(lo, hi);
    return static_cast<uint8_t>(simd_hmax_epi32(max8));
}

// Bit-exact SIMD implementation of `convert_u32_to_bfp<Bfp8_b, false>` for 8
// lanes at a time. Returns 8 uint32_t lanes each containing an 8-bit BFP8
// value (sign:1 | mantissa:7) in the low byte.
inline simde__m256i simd_convert_to_bfp8_b_8(simde__m256i input, uint32_t shared_exp) {
    // Decompose into sign/exp/mantissa.
    const simde__m256i v_sign_mask = simde_mm256_set1_epi32(0x80000000);
    const simde__m256i v_exp_mask = simde_mm256_set1_epi32(0x7f800000);
    const simde__m256i v_man_mask = simde_mm256_set1_epi32(0x007fffff);
    const simde__m256i v_zero = simde_mm256_setzero_si256();
    const simde__m256i v_hidden_one = simde_mm256_set1_epi32(0x00800000);
    const simde__m256i v_round_mask = simde_mm256_set1_epi32(0x0001ffff);  // (1<<17)-1
    const simde__m256i v_tie = simde_mm256_set1_epi32(0x00010000);         // 1<<16
    const simde__m256i v_one = simde_mm256_set1_epi32(1);
    const simde__m256i v_max_man = simde_mm256_set1_epi32(0x7f);  // (1<<7)-1

    simde__m256i sign = simde_mm256_srli_epi32(simde_mm256_and_si256(input, v_sign_mask), 31);
    simde__m256i exp = simde_mm256_srli_epi32(simde_mm256_and_si256(input, v_exp_mask), 23);
    simde__m256i man = simde_mm256_and_si256(input, v_man_mask);

    // Track which lanes are zero/denormal (exp == 0). These will be forced to
    // output 0 at the end.
    simde__m256i zero_lane = simde_mm256_cmpeq_epi32(exp, v_zero);

    // Add hidden 1 → 24-bit mantissa.
    man = simde_mm256_or_si256(man, v_hidden_one);

    // exp_diff = max(shared_exp - exp, 0). For Bfp8_b the scalar path uses
    // `if (shared_exp > exp) shift right by exp_diff`. Negative differences
    // mean the lane has a larger exponent than the shared one, which only
    // happens if the lane was zero/denormal (handled by zero_lane mask), so
    // clamping to >= 0 is safe. Clamp shift amount to 31 to avoid UB; mantissa
    // already shifted to 0 long before reaching that.
    simde__m256i shared = simde_mm256_set1_epi32(static_cast<int>(shared_exp));
    simde__m256i diff = simde_mm256_sub_epi32(shared, exp);
    diff = simde_mm256_max_epi32(diff, v_zero);
    simde__m256i v_31 = simde_mm256_set1_epi32(31);
    diff = simde_mm256_min_epi32(diff, v_31);
    man = simde_mm256_srlv_epi32(man, diff);

    // Round-to-nearest, ties-to-even at MANTISSA_BFP_SHIFT = 17.
    simde__m256i round_value = simde_mm256_and_si256(man, v_round_mask);
    simde__m256i man_shifted = simde_mm256_srli_epi32(man, 17);
    simde__m256i guard = simde_mm256_and_si256(man_shifted, v_one);

    // round_up if (round > tie) || (round == tie && guard == 1)
    simde__m256i gt = simde_mm256_cmpgt_epi32(round_value, v_tie);
    simde__m256i eq = simde_mm256_cmpeq_epi32(round_value, v_tie);
    simde__m256i guard_set = simde_mm256_cmpeq_epi32(guard, v_one);
    simde__m256i tie_up = simde_mm256_and_si256(eq, guard_set);
    simde__m256i round_up_mask = simde_mm256_or_si256(gt, tie_up);
    simde__m256i one_if_up = simde_mm256_and_si256(round_up_mask, v_one);
    man_shifted = simde_mm256_add_epi32(man_shifted, one_if_up);

    // Saturate to 7-bit max.
    man_shifted = simde_mm256_min_epu32(man_shifted, v_max_man);

    // Force zero/denormal lanes to 0 mantissa, sign cleared.
    man_shifted = simde_mm256_andnot_si256(zero_lane, man_shifted);

    // Drop sign for any lane whose mantissa rounded to zero (matches scalar).
    simde__m256i man_is_zero = simde_mm256_cmpeq_epi32(man_shifted, v_zero);
    sign = simde_mm256_andnot_si256(man_is_zero, sign);

    // Pack sign:1 || mantissa:7 into the low byte of each lane.
    return simde_mm256_or_si256(simde_mm256_slli_epi32(sign, 7), man_shifted);
}

// Pack 16 fp32 values (one full face row) into 4 BFP8_b output dwords using
// SIMD. Writes the 4 dwords (16 bytes) as a single 128-bit store.
inline void simd_pack_face_row_bfp8_b(const uint32_t* row16, uint32_t shared_exp, uint32_t* out_dwords) {
    simde__m256i lo = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(row16));
    simde__m256i hi = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(row16 + 8));

    simde__m256i bfp_lo = simd_convert_to_bfp8_b_8(lo, shared_exp);  // 8 lanes, byte in low byte
    simde__m256i bfp_hi = simd_convert_to_bfp8_b_8(hi, shared_exp);

    // Pack 4 lanes per output dword: dword0 = bfp_lo[0..3], dword1 = bfp_lo[4..7],
    //                                dword2 = bfp_hi[0..3], dword3 = bfp_hi[4..7].
    // Each lane currently holds an 8-bit value in bits [7:0]. Shift the 4 lanes
    // by 0/8/16/24 then OR-reduce the 4 dwords inside each 128-bit half.
    const simde__m256i v_shift_low4 = simde_mm256_set_epi32(24, 16, 8, 0, 24, 16, 8, 0);
    bfp_lo = simde_mm256_sllv_epi32(bfp_lo, v_shift_low4);
    bfp_hi = simde_mm256_sllv_epi32(bfp_hi, v_shift_low4);

    // Reduce 4 dwords -> 1 dword in lane 0 (other lanes left as don't-care).
    auto reduce_to_lane0 = [](simde__m128i v) -> simde__m128i {
        simde__m128i s = simde_mm_or_si128(v, simde_mm_shuffle_epi32(v, 0x4E));  // OR halves
        return simde_mm_or_si128(s, simde_mm_shuffle_epi32(s, 0xB1));            // OR pairs
    };

    simde__m128i r0 = reduce_to_lane0(simde_mm256_castsi256_si128(bfp_lo));     // dword0 in lane 0
    simde__m128i r1 = reduce_to_lane0(simde_mm256_extracti128_si256(bfp_lo, 1));  // dword1
    simde__m128i r2 = reduce_to_lane0(simde_mm256_castsi256_si128(bfp_hi));     // dword2
    simde__m128i r3 = reduce_to_lane0(simde_mm256_extracti128_si256(bfp_hi, 1));  // dword3

    // Assemble [dword0, dword1, dword2, dword3] in one __m128i without going
    // through scalar registers, then issue a single 16-byte store.
    simde__m128i p01 = simde_mm_unpacklo_epi32(r0, r1);  // [d0, d1, *, *]
    simde__m128i p23 = simde_mm_unpacklo_epi32(r2, r3);  // [d2, d3, *, *]
    simde__m128i packed = simde_mm_unpacklo_epi64(p01, p23);  // [d0, d1, d2, d3]
    simde_mm_storeu_si128(reinterpret_cast<simde__m128i*>(out_dwords), packed);
}

// Compile-time predicate: SIMD path is only built for the input types we know
// can be cheaply gathered into a contiguous fp32 row (float and bfloat16).
template <typename T>
constexpr bool kSimdEligibleType = std::is_same_v<T, float> || std::is_same_v<T, bfloat16>;

// Gather one face row of 16 elements into a contiguous uint32_t[16] buffer of
// IEEE-754 single-precision bit patterns. Specialized to use AVX2 loads when
// T is float (zero-copy) and a 16-bit -> 32-bit shift when T is bfloat16.
template <typename T>
inline void gather_face_row_16_fp32(const T* src, uint32_t* dst) {
    if constexpr (std::is_same_v<T, float>) {
        // Direct memcpy (AVX2 loads happen later when consumed).
        std::memcpy(dst, src, 16 * sizeof(uint32_t));
    } else if constexpr (std::is_same_v<T, bfloat16>) {
        // bfloat16 -> fp32 is just `(uint32_t)bits << 16`. Read 16 bf16 values
        // (32 bytes) as one 256-bit AVX2 load, split into two 128-bit halves,
        // widen each half to epi32, then shift left by 16.
        const simde__m256i bf = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(src));
        const simde__m128i bf_lo = simde_mm256_castsi256_si128(bf);
        const simde__m128i bf_hi = simde_mm256_extracti128_si256(bf, 1);
        simde__m256i lo = simde_mm256_slli_epi32(simde_mm256_cvtepu16_epi32(bf_lo), 16);
        simde__m256i hi = simde_mm256_slli_epi32(simde_mm256_cvtepu16_epi32(bf_hi), 16);
        simde_mm256_storeu_si256(reinterpret_cast<simde__m256i*>(dst), lo);
        simde_mm256_storeu_si256(reinterpret_cast<simde__m256i*>(dst + 8), hi);
    } else {
        // Generic fallback (shouldn't be reached when kSimdEligibleType<T>).
        for (int i = 0; i < 16; ++i) {
            dst[i] = element_to_u32_bits<T>(src[i]);
        }
    }
}

// Pack a single tile's worth of input data into the BFP layout. Writes
// `num_exp_dwords` exponent dwords followed by `num_data_dwords` mantissa
// dwords starting at `tile_out`. The input slab for this tile is read from
// `tile_in` (which must already point at this tile's first element when
// `row_major_input` is false). When `row_major_input` is true, the function
// computes the correct row_major offsets itself using `tile_W` and `face_*`.
template <tt::DataFormat BfpFormat, typename T>
inline void pack_one_tile(
    const T* tile_in,
    bool row_major_input,
    bool is_exp_a,
    uint32_t tile_W,
    uint32_t face_H,
    uint32_t face_W,
    uint32_t subtiles_in_tile_row,
    uint32_t subtiles_in_tile_col,
    uint32_t num_exp_dwords,
    bool exponent_padding,
    uint32_t l1_alignment,
    uint32_t* tile_out) {
    constexpr uint32_t num_mantissas_in_dword = []() {
        if constexpr (BfpFormat == tt::DataFormat::Bfp2 || BfpFormat == tt::DataFormat::Bfp2_b) {
            return 16u;
        } else if constexpr (BfpFormat == tt::DataFormat::Bfp4 || BfpFormat == tt::DataFormat::Bfp4_b) {
            return 8u;
        } else {
            return 4u;
        }
    }();

    constexpr uint32_t MAX_FACE_W = 64;  // generous upper bound; standard is 16
    TT_ASSERT(face_W <= MAX_FACE_W, "face_W ({}) exceeds MAX_FACE_W ({})", face_W, MAX_FACE_W);

    // Stack-allocated per-row scratch (avoids the per-row std::vector alloc
    // that dominated the original implementation).
    uint32_t single_row[MAX_FACE_W];

    // Exponent staging area: at most one byte per row of every face. With
    // standard production tiles (32x32 tile, 16x16 face) this is 64 bytes.
    // Cap at 256 to match the legacy `exponents_with_padding` reserve bound.
    constexpr uint32_t MAX_EXPONENTS_PER_TILE = 1024;
    const uint32_t num_exponents_per_tile = subtiles_in_tile_row * subtiles_in_tile_col * face_H;
    TT_ASSERT(
        num_exponents_per_tile <= MAX_EXPONENTS_PER_TILE,
        "num_exponents_per_tile ({}) exceeds MAX_EXPONENTS_PER_TILE ({})",
        num_exponents_per_tile,
        MAX_EXPONENTS_PER_TILE);
    uint8_t exp_buf[MAX_EXPONENTS_PER_TILE];

    // Mantissa output area starts after the exponent dwords for this tile.
    uint32_t* mantissa_out = tile_out + num_exp_dwords;

    // Walk the tile in face-major then row-major order. For each row we:
    //   1. Gather the face_W input u32 bit patterns into single_row.
    //   2. Compute the shared exponent byte and store it in exp_buf.
    //   3. Pack mantissas (num_mantissas_in_dword at a time) into mantissa_out.
    //
    // The mantissa packing index is just `mantissa_dword_idx` (monotonic),
    // because face_W is always a multiple of num_mantissas_in_dword in
    // production formats and we asserted face_W >= num_mantissas_in_dword
    // upstream.
    uint32_t exp_idx = 0;
    uint32_t mantissa_dword_idx = 0;
    uint32_t fp32_element_index = 0;

    // Bit widths for the chosen BFP format - mirrors create_packed_bfp_packed_as_u32
    // but lets us inline the inner mantissa packing without an extra vector
    // allocation per dword.
    constexpr uint32_t MANTISSA_OUT_BITS = 32u / num_mantissas_in_dword;
    constexpr uint32_t MANTISSA_OUT_MASK = (1u << MANTISSA_OUT_BITS) - 1u;

    // Whether the AVX2/simde fast path is applicable for the current
    // (BfpFormat, T, face_W, is_exp_a) combination. Decided at compile time
    // for the format/type axes and at runtime for face_W and is_exp_a. The
    // TT_BFP_HOST_TILIZER_DISABLE_SIMD env var (set to anything non-empty)
    // forces the scalar path - useful for benchmarking or debugging.
    constexpr bool simd_eligible_format = (BfpFormat == tt::DataFormat::Bfp8_b) && kSimdEligibleType<T>;
    static const bool simd_disabled_by_env = []() {
        const char* env = std::getenv("TT_BFP_HOST_TILIZER_DISABLE_SIMD");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    const bool use_simd = simd_eligible_format && (face_W == 16) && !is_exp_a && !simd_disabled_by_env;

    // Helper: compute the start-of-row pointer in the input buffer for either
    // layout. `fp32_element_index` is captured by reference and advanced in
    // the non-row-major path.
    auto row_pointer = [&](uint32_t tr, uint32_t tc, uint32_t i) -> const T* {
        if (row_major_input) {
            const uint32_t row_base = (tr * face_H + i) * tile_W + (tc * face_W);
            return tile_in + row_base;
        }
        const T* p = tile_in + fp32_element_index;
        fp32_element_index += face_W;
        return p;
    };

    // ---- SIMD loop --------------------------------------------------------
    // Hoisted out of the per-row branch so the compiler sees a straight-line,
    // fully-vectorisable loop body (no `if (use_simd)` / `if constexpr` checks
    // inside the hot inner loop). Only entered when use_simd is true; we
    // verified that simd_eligible_format holds at compile time too.
    if constexpr (simd_eligible_format) {
        if (use_simd) {
            for (uint32_t tr = 0; tr < subtiles_in_tile_row; ++tr) {
                for (uint32_t tc = 0; tc < subtiles_in_tile_col; ++tc) {
                    for (uint32_t i = 0; i < face_H; ++i) {
                        const T* row_src = row_pointer(tr, tc, i);
                        gather_face_row_16_fp32<T>(row_src, single_row);
                        const uint8_t shared_exp = simd_get_max_exp_16_fp32_b(single_row);
                        exp_buf[exp_idx++] = shared_exp;
                        simd_pack_face_row_bfp8_b(single_row, shared_exp, mantissa_out + mantissa_dword_idx);
                        mantissa_dword_idx += 4;  // face_W=16, 4 mantissas/dword -> 4 dwords/row
                    }
                }
            }
        }
    }

    // ---- Scalar loop ------------------------------------------------------
    // Runs when the SIMD fast path is not applicable (BFP4/2, is_exp_a=true,
    // non-16 face_W, integer input types, or SIMD disabled by env var). Kept
    // as a single straight-line loop body for the same reason - no inner
    // branches over the dispatch decision.
    const bool run_scalar_loop = !(simd_eligible_format && use_simd);
    if (run_scalar_loop) {
        for (uint32_t tr = 0; tr < subtiles_in_tile_row; ++tr) {
            for (uint32_t tc = 0; tc < subtiles_in_tile_col; ++tc) {
                for (uint32_t i = 0; i < face_H; ++i) {
                    const T* row_src = row_pointer(tr, tc, i);

                    // Gather one face row into single_row.
                    for (uint32_t j = 0; j < face_W; ++j) {
                        single_row[j] = element_to_u32_bits<T>(row_src[j]);
                    }

                    // Compute shared exponent (scalar reduction).
                    uint32_t max_exp = 0;
                    for (uint32_t k = 0; k < face_W; ++k) {
                        uint32_t exp = (single_row[k] & 0x7f800000u) >> 23;
                        if (is_exp_a) {
                            int32_t se = static_cast<int32_t>(exp) - 127 + 15;
                            if (se > 31) {
                                se = 31;
                            } else if (se < 0) {
                                se = 0;
                            }
                            exp = static_cast<uint32_t>(se);
                        }
                        max_exp = std::max(exp, max_exp);
                    }
                    const uint8_t shared_exp = static_cast<uint8_t>(max_exp);
                    exp_buf[exp_idx++] = shared_exp;

                    // Pack mantissas. Bit ordering matches
                    // create_packed_bfp_packed_as_u32: the first input lands in
                    // the LSBs of the output dword.
                    for (uint32_t base = 0; base < face_W; base += num_mantissas_in_dword) {
                        uint32_t packed = 0;
                        for (int k = static_cast<int>(num_mantissas_in_dword) - 1; k >= 0; --k) {
                            const uint32_t conv =
                                convert_u32_to_bfp<BfpFormat, false>(single_row[base + k], shared_exp, is_exp_a);
                            packed = (packed << MANTISSA_OUT_BITS) | (conv & MANTISSA_OUT_MASK);
                        }
                        mantissa_out[mantissa_dword_idx++] = packed;
                    }
                }
            }
        }
    }

    // Emit exponent dwords for this tile. Two layouts (matching legacy):
    //   * exponent_padding == true : write all exp bytes, zero-pad up to the
    //     L1 alignment, then byte-pack into dwords.
    //   * exponent_padding == false: every 4 exponent bytes -> 1 dword.
    if (exponent_padding) {
        const uint32_t padded_count = static_cast<uint32_t>(tt::round_up(exp_idx, l1_alignment));
        // Zero out the padding tail in-place.
        for (uint32_t k = exp_idx; k < padded_count; ++k) {
            exp_buf[k] = 0;
        }
        pack_exp_bytes_to_dwords(exp_buf, padded_count, tile_out);
    } else {
        // The per-tile exponent count must be a multiple of 4 in this layout
        // (the caller guarantees this by selecting the optimized fast path).
        TT_ASSERT(
            exp_idx % 4 == 0,
            "Optimized BFP packer requires per-tile exponent count divisible by 4 in non-padded layout");
        pack_exp_bytes_to_dwords(exp_buf, exp_idx, tile_out);
    }
}

// Worker that packs a contiguous range of tiles. Used both for serial and
// for std::thread-based parallel execution.
template <tt::DataFormat BfpFormat, typename T>
void pack_tile_range(
    const T* input_base,
    bool row_major_input,
    bool is_exp_a,
    uint32_t tile_W,
    uint32_t face_H,
    uint32_t face_W,
    uint32_t subtiles_in_tile_row,
    uint32_t subtiles_in_tile_col,
    uint32_t num_exp_dwords,
    bool exponent_padding,
    uint32_t l1_alignment,
    uint32_t num_float_in_tile,
    uint32_t bfp_dwords_per_tile,
    uint32_t tile_begin,
    uint32_t tile_end,
    uint32_t* output_base) {
    for (uint32_t tile_idx = tile_begin; tile_idx < tile_end; ++tile_idx) {
        const T* tile_in = input_base + static_cast<size_t>(tile_idx) * num_float_in_tile;
        uint32_t* tile_out = output_base + static_cast<size_t>(tile_idx) * bfp_dwords_per_tile;
        pack_one_tile<BfpFormat, T>(
            tile_in,
            row_major_input,
            is_exp_a,
            tile_W,
            face_H,
            face_W,
            subtiles_in_tile_row,
            subtiles_in_tile_col,
            num_exp_dwords,
            exponent_padding,
            l1_alignment,
            tile_out);
    }
}

// Decide how many worker threads to use. We only thread when there is enough
// work to amortize the (small but non-zero) cost of std::thread startup. The
// thresholds are intentionally conservative so that small offline conversions
// stay single-threaded.
inline uint32_t pick_num_pack_threads(uint32_t num_tiles, uint32_t num_float_in_tile) {
    // Allow pinning thread count via env var (useful for debugging / benchmarking).
    // 0 disables threading; >=1 forces that many threads (capped by tile count).
    static const int env_threads = []() {
        const char* env = std::getenv("TT_BFP_HOST_TILIZER_THREADS");
        if (env == nullptr) {
            return -1;
        }
        return std::atoi(env);
    }();
    if (env_threads == 0) {
        return 1;
    }
    if (env_threads > 0) {
        return std::min<uint32_t>(static_cast<uint32_t>(env_threads), num_tiles);
    }

    // Heuristic: aim for ~4 tiles per thread minimum, cap by hardware concurrency,
    // require a minimum total work size to bother threading at all.
    constexpr uint64_t MIN_ELEMENTS_FOR_THREADING = 1u << 14;  // 16K input elements
    constexpr uint32_t MIN_TILES_PER_THREAD = 4;
    const uint64_t total_elements = static_cast<uint64_t>(num_tiles) * num_float_in_tile;
    if (total_elements < MIN_ELEMENTS_FOR_THREADING || num_tiles < 2 * MIN_TILES_PER_THREAD) {
        return 1;
    }
    uint32_t hw = std::thread::hardware_concurrency();
    if (hw == 0) {
        hw = 2;
    }
    // Cap threads so each gets at least MIN_TILES_PER_THREAD tiles.
    const uint32_t by_tiles = num_tiles / MIN_TILES_PER_THREAD;
    const uint32_t threads = std::min<uint32_t>(hw, by_tiles);
    return std::max<uint32_t>(threads, 1);
}

template <tt::DataFormat BfpFormat, typename T>
std::vector<uint32_t> pack_as_bfp_tiles(
    tt::stl::Span<const T> input_data,
    bool row_major_input,
    bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile) {
    ZoneScoped;

    TT_ASSERT(
        BfpFormat == tt::DataFormat::Bfp2 || BfpFormat == tt::DataFormat::Bfp4 || BfpFormat == tt::DataFormat::Bfp8 ||
        BfpFormat == tt::DataFormat::Bfp2_b || BfpFormat == tt::DataFormat::Bfp4_b ||
        BfpFormat == tt::DataFormat::Bfp8_b);

    const uint32_t tile_H = tile.has_value() ? tile->get_tile_shape()[0] : tt::constants::TILE_HEIGHT;
    const uint32_t tile_W = tile.has_value() ? tile->get_tile_shape()[1] : tt::constants::TILE_WIDTH;
    const uint32_t face_H = tile.has_value() ? tile->get_face_shape()[0] : tt::constants::FACE_HEIGHT;
    const uint32_t face_W = tile.has_value() ? tile->get_face_shape()[1] : tt::constants::FACE_WIDTH;
    const uint32_t tile_HW = tile_H * tile_W;
    const uint32_t subtiles_in_tile_row = tile_H / face_H;
    const uint32_t subtiles_in_tile_col = tile_W / face_W;

    const uint32_t l1_alignment =
        tt::tt_metal::MetalContext::instance().hal().get_alignment(tt::tt_metal::HalMemType::L1);
    const uint32_t num_exponents_per_tile = subtiles_in_tile_row * subtiles_in_tile_col * face_H;
    const bool exponent_padding = num_exponents_per_tile < l1_alignment;

    const uint32_t num_float_in_tile = tile_HW;
    TT_ASSERT(input_data.size() % num_float_in_tile == 0);
    const uint32_t num_tiles = input_data.size() / num_float_in_tile;

    constexpr uint32_t num_mantissas_in_dword = []() {
        if constexpr (BfpFormat == tt::DataFormat::Bfp2 || BfpFormat == tt::DataFormat::Bfp2_b) {
            return 16;
        } else if constexpr (BfpFormat == tt::DataFormat::Bfp4 || BfpFormat == tt::DataFormat::Bfp4_b) {
            return 8;
        } else {
            return 4;
        }
    }();

    // Output sizing - compute per-tile dword counts then allocate the whole
    // result up-front. Two cases mirror the legacy code's two layouts.
    uint32_t num_exp_dwords_per_tile;
    if (exponent_padding) {
        num_exp_dwords_per_tile = static_cast<uint32_t>(tt::round_up(num_exponents_per_tile, l1_alignment)) / 4;
    } else {
        num_exp_dwords_per_tile = num_exponents_per_tile / 4;
    }
    const uint32_t num_data_dwords_per_tile = tile_HW / num_mantissas_in_dword;
    const uint32_t bfp_dwords_per_tile = num_exp_dwords_per_tile + num_data_dwords_per_tile;

    // Conditions under which we can take the optimized per-tile fast path:
    //   * Each tile's exponent count packs cleanly into whole dwords (or we
    //     are using the padded layout, which always packs cleanly).
    //   * Each face row's mantissa count is a multiple of num_mantissas_in_dword
    //     so we can flush mantissa dwords without carry across rows.
    //   * The non-row-major (pre-tilized) path: order of writes per tile is
    //     identical to legacy, so no constraint there beyond divisibility.
    const bool exp_packs_cleanly = exponent_padding || (num_exponents_per_tile % 4 == 0);
    const bool mantissas_pack_cleanly = (face_W % num_mantissas_in_dword == 0);

    if (exp_packs_cleanly && mantissas_pack_cleanly) {
        std::vector<uint32_t> packed_result(static_cast<size_t>(num_tiles) * bfp_dwords_per_tile);
        if (num_tiles == 0) {
            return packed_result;
        }

        const T* input_base = input_data.data();
        uint32_t* output_base = packed_result.data();

        const uint32_t num_threads = pick_num_pack_threads(num_tiles, num_float_in_tile);

        if (num_threads <= 1) {
            pack_tile_range<BfpFormat, T>(
                input_base,
                row_major_input,
                is_exp_a,
                tile_W,
                face_H,
                face_W,
                subtiles_in_tile_row,
                subtiles_in_tile_col,
                num_exp_dwords_per_tile,
                exponent_padding,
                l1_alignment,
                num_float_in_tile,
                bfp_dwords_per_tile,
                0,
                num_tiles,
                output_base);
        } else {
            // Parallel path. Each worker owns a disjoint, contiguous range of
            // tiles - inputs and outputs do not overlap so no synchronization
            // is needed. We use std::thread directly (rather than the metal
            // device-bound thread pool) because the host tilizer is callable
            // without a device being present.
            std::vector<std::thread> workers;
            workers.reserve(num_threads - 1);

            const uint32_t tiles_per_thread = (num_tiles + num_threads - 1) / num_threads;
            for (uint32_t t = 1; t < num_threads; ++t) {
                const uint32_t begin = t * tiles_per_thread;
                if (begin >= num_tiles) {
                    break;
                }
                const uint32_t end = std::min(begin + tiles_per_thread, num_tiles);
                workers.emplace_back([=]() {
                    pack_tile_range<BfpFormat, T>(
                        input_base,
                        row_major_input,
                        is_exp_a,
                        tile_W,
                        face_H,
                        face_W,
                        subtiles_in_tile_row,
                        subtiles_in_tile_col,
                        num_exp_dwords_per_tile,
                        exponent_padding,
                        l1_alignment,
                        num_float_in_tile,
                        bfp_dwords_per_tile,
                        begin,
                        end,
                        output_base);
                });
            }
            // The main thread takes the first chunk while workers run.
            const uint32_t main_end = std::min(tiles_per_thread, num_tiles);
            pack_tile_range<BfpFormat, T>(
                input_base,
                row_major_input,
                is_exp_a,
                tile_W,
                face_H,
                face_W,
                subtiles_in_tile_row,
                subtiles_in_tile_col,
                num_exp_dwords_per_tile,
                exponent_padding,
                l1_alignment,
                num_float_in_tile,
                bfp_dwords_per_tile,
                0,
                main_end,
                output_base);

            for (auto& w : workers) {
                w.join();
            }
        }

        return packed_result;
    }

    // ------------------------------------------------------------------
    // Fallback: original unmodified algorithm. Triggered only by exotic
    // tile/face shapes where the per-tile divisibility invariants do not
    // hold (production formats - 32x32 tile, 16x16 face - always satisfy
    // both invariants and use the optimized path above).
    //
    // All scratch vectors used by this path are pre-sized once with
    // `resize()` and then written via indexed assignment instead of
    // `push_back`. `single_row` has a constant size (subtile_cols), and
    // the flush-threshold accumulators (`exponents`, `data`) and the
    // per-tile `packed_data` all have constant maximum sizes, so once
    // capacity is in place we never reallocate and we never have to
    // re-check size on every store. Counters (`exp_idx`, `data_idx`,
    // `packed_data_idx`) track logical fill level within each fixed-
    // size buffer. `packed_result` is reserved (not resized) to its
    // exact final size; we still use push_back / insert into it, but
    // the up-front reserve eliminates any grow-from-zero reallocations.
    // ------------------------------------------------------------------
    const uint32_t subtile_rows = face_H;
    const uint32_t subtile_cols = face_W;
    constexpr size_t num_exponents_in_dword = 4;

    std::vector<uint32_t> packed_result;
    packed_result.reserve(static_cast<size_t>(num_tiles) * bfp_dwords_per_tile);

    // Constant-size scratch buffers - sized once, indexed by position.
    std::vector<uint32_t> single_row(subtile_cols);
    std::vector<uint8_t> exponents(num_exponents_in_dword);
    std::vector<uint32_t> data(num_mantissas_in_dword);
    std::vector<uint32_t> packed_data(num_data_dwords_per_tile);

    // Padded-exponent scratch: per-tile size varies with padding, so we
    // still rely on push_back into a single hoisted, pre-reserved buffer.
    std::vector<uint8_t> exponents_with_padding;
    exponents_with_padding.reserve(l1_alignment * subtiles_in_tile_row * subtiles_in_tile_col);

    int fp32_element_index = 0;
    size_t exp_idx = 0;

    for (uint32_t tile_index = 0; tile_index < num_tiles; ++tile_index) {
        size_t packed_data_idx = 0;
        exponents_with_padding.clear();
        for (uint32_t tr = 0; tr < subtiles_in_tile_row; ++tr) {
            for (uint32_t tc = 0; tc < subtiles_in_tile_col; ++tc) {
                for (uint32_t i = 0; i < subtile_rows; ++i) {
                    for (uint32_t j = 0; j < subtile_cols; ++j) {
                        int data_index;
                        if (row_major_input) {
                            data_index = (tr * face_H + i) * tile_W + (tc * face_W + j) +
                                         (num_float_in_tile * tile_index);
                        } else {
                            data_index = fp32_element_index++;
                        }
                        float float_num = static_cast<float>(input_data[data_index]);
                        single_row[j] = std::bit_cast<uint32_t>(float_num);
                    }

                    uint8_t exp = get_max_exp(single_row, is_exp_a);

                    if (exponent_padding) {
                        exponents_with_padding.push_back(exp);
                    } else {
                        exponents[exp_idx++] = exp;
                        if (exp_idx == num_exponents_in_dword) {
                            packed_result.push_back(get_exp_dword(exponents));
                            exp_idx = 0;
                        }
                    }

                    size_t data_idx = 0;
                    for (uint32_t u32_datum : single_row) {
                        data[data_idx++] = u32_datum;
                        if (data_idx == num_mantissas_in_dword) {
                            packed_data[packed_data_idx++] =
                                create_packed_bfp_packed_as_u32<BfpFormat>(data, exp, is_exp_a);
                            data_idx = 0;
                        }
                    }
                }
            }
        }
        if (exponent_padding) {
            // Zero-pad in place instead of allocating a temporary `pads` vector
            // each tile just to copy zeros across.
            const size_t pad_count =
                tt::round_up(exponents_with_padding.size(), l1_alignment) - exponents_with_padding.size();
            exponents_with_padding.insert(exponents_with_padding.end(), pad_count, static_cast<uint8_t>(0));
            std::vector<uint32_t> packed = pack_exponents(exponents_with_padding, num_exponents_in_dword);
            packed_result.insert(packed_result.end(), packed.begin(), packed.end());
        }
        packed_result.insert(packed_result.end(), packed_data.begin(), packed_data.end());
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

template std::vector<uint32_t> pack_fp32_vec_as_bfp_tiles<tt::DataFormat::Bfp2>(tt::stl::Span<const float> fp32_vec, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_fp32_vec_as_bfp_tiles<tt::DataFormat::Bfp4>(tt::stl::Span<const float> fp32_vec, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_fp32_vec_as_bfp_tiles<tt::DataFormat::Bfp8>(tt::stl::Span<const float> fp32_vec, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_fp32_vec_as_bfp_tiles<tt::DataFormat::Bfp2_b>(tt::stl::Span<const float> fp32_vec, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_fp32_vec_as_bfp_tiles<tt::DataFormat::Bfp4_b>(tt::stl::Span<const float> fp32_vec, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_fp32_vec_as_bfp_tiles<tt::DataFormat::Bfp8_b>(tt::stl::Span<const float> fp32_vec, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);

template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2>(tt::stl::Span<const float> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4>(tt::stl::Span<const float> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8>(tt::stl::Span<const float> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2_b>(tt::stl::Span<const float> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4_b>(tt::stl::Span<const float> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8_b>(tt::stl::Span<const float> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);

template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2>(tt::stl::Span<const bfloat16> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4>(tt::stl::Span<const bfloat16> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8>(tt::stl::Span<const bfloat16> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2_b>(tt::stl::Span<const bfloat16> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4_b>(tt::stl::Span<const bfloat16> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8_b>(tt::stl::Span<const bfloat16> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);

template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2>(tt::stl::Span<const int32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4>(tt::stl::Span<const int32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8>(tt::stl::Span<const int32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2_b>(tt::stl::Span<const int32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4_b>(tt::stl::Span<const int32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8_b>(tt::stl::Span<const int32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);

template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2>(tt::stl::Span<const uint32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4>(tt::stl::Span<const uint32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8>(tt::stl::Span<const uint32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2_b>(tt::stl::Span<const uint32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4_b>(tt::stl::Span<const uint32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8_b>(tt::stl::Span<const uint32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);


template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2>(tt::stl::Span<const uint16_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4>(tt::stl::Span<const uint16_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8>(tt::stl::Span<const uint16_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2_b>(tt::stl::Span<const uint16_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4_b>(tt::stl::Span<const uint16_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8_b>(tt::stl::Span<const uint16_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);


template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2>(tt::stl::Span<const uint8_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4>(tt::stl::Span<const uint8_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8>(tt::stl::Span<const uint8_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2_b>(tt::stl::Span<const uint8_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4_b>(tt::stl::Span<const uint8_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8_b>(tt::stl::Span<const uint8_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);

// clang-format on
