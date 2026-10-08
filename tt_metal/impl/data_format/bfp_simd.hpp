// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <tt-metalium/bfloat16.hpp>
#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace tt::tt_metal::bfp_simd {
// BFP_B uses the maximum FP32 exponent of each 16-value face row.
// Match Metal's integer conversion exactly: flush exponent-zero inputs,
// align significands, round ties to even, saturate, suppress signed zero.
template <int bits, typename T>
static inline void row_scalar(const T* src, uint8_t* exp_out, uint8_t* out) {
    uint32_t words[16], codes[16];
    uint32_t shared = 0;
    for (int j = 0; j < 16; ++j) {
        float value = static_cast<float>(src[j]);
        std::memcpy(&words[j], &value, 4);
        shared = std::max(shared, (words[j] >> 23) & 255);
    }
    *exp_out = static_cast<uint8_t>(shared);
    const int shift = 24 - bits;
    const uint32_t limit = (1u << bits) - 1;
    for (int j = 0; j < 16; ++j) {
        uint32_t exponent = (words[j] >> 23) & 255;
        uint32_t diff = shared - exponent;
        uint32_t mantissa = (words[j] & 0x7fffff) | 0x800000;
        mantissa = diff >= 32 ? 0 : mantissa >> diff;
        mantissa = (mantissa + ((1u << (shift - 1)) - 1) + ((mantissa >> shift) & 1)) >> shift;
        mantissa = exponent ? std::min(mantissa, limit) : 0;
        codes[j] = mantissa ? mantissa | ((words[j] >> 31) << bits) : 0;
    }
    const int per_byte = 8 / (bits + 1);
    for (int j = 0; j < 16 / per_byte; ++j) {
        uint32_t value = 0;
        for (int k = 0; k < per_byte; ++k) {
            value |= codes[j * per_byte + k] << (k * (bits + 1));
        }
        out[j] = static_cast<uint8_t>(value);
    }
}

#if defined(__x86_64__)
__attribute__((target("avx2"))) static inline void store_codes(__m128i bytes, uint8_t* out, int bits) {
    if (bits == 7) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out), bytes);
    } else if (bits == 3) {
        auto pairs = _mm_maddubs_epi16(bytes, _mm_set1_epi16(0x1001));
        _mm_storel_epi64(reinterpret_cast<__m128i*>(out), _mm_packus_epi16(pairs, pairs));
    } else {
        auto pairs = _mm_maddubs_epi16(bytes, _mm_set1_epi16(0x0401));
        auto quads = _mm_madd_epi16(pairs, _mm_set1_epi32(0x00100001));
        auto shorts = _mm_packus_epi32(quads, quads);
        uint32_t value = static_cast<uint32_t>(_mm_cvtsi128_si32(_mm_packus_epi16(shorts, shorts)));
        std::memcpy(out, &value, 4);
    }
}

template <int bits, typename T>
__attribute__((target("avx2"), always_inline)) static inline void row_avx2(
    const T* src, uint8_t* exp_out, uint8_t* out) {
    __m256i words[2];
    if constexpr (sizeof(T) == 4) {
        words[0] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
        words[1] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 8));
    } else {
        words[0] = _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src))), 16);
        words[1] =
            _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 8))), 16);
    }
    auto mask = _mm256_set1_epi32(255);
    __m256i exps[2] = {
        _mm256_and_si256(_mm256_srli_epi32(words[0], 23), mask),
        _mm256_and_si256(_mm256_srli_epi32(words[1], 23), mask)};
    auto m = _mm256_max_epu32(exps[0], exps[1]);
    auto h = _mm_max_epu32(_mm256_castsi256_si128(m), _mm256_extracti128_si256(m, 1));
    h = _mm_max_epu32(h, _mm_shuffle_epi32(h, 0x4e));
    h = _mm_max_epu32(h, _mm_shuffle_epi32(h, 0xb1));
    uint32_t shared = static_cast<uint32_t>(_mm_cvtsi128_si32(h));
    *exp_out = static_cast<uint8_t>(shared);

    __m128i half[2];
    for (int j = 0; j < 2; ++j) {
        auto mantissa =
            _mm256_or_si256(_mm256_and_si256(words[j], _mm256_set1_epi32(0x7fffff)), _mm256_set1_epi32(0x800000));
        mantissa = _mm256_srlv_epi32(mantissa, _mm256_sub_epi32(_mm256_broadcastd_epi32(h), exps[j]));
        auto odd = _mm256_and_si256(_mm256_srli_epi32(mantissa, 24 - bits), _mm256_set1_epi32(1));
        mantissa = _mm256_srli_epi32(
            _mm256_add_epi32(mantissa, _mm256_add_epi32(odd, _mm256_set1_epi32((1u << (23 - bits)) - 1))), 24 - bits);
        mantissa = _mm256_min_epu32(mantissa, _mm256_set1_epi32((1 << bits) - 1));
        auto zero = _mm256_or_si256(
            _mm256_cmpeq_epi32(exps[j], _mm256_setzero_si256()), _mm256_cmpeq_epi32(mantissa, _mm256_setzero_si256()));
        auto sign = _mm256_slli_epi32(_mm256_srli_epi32(words[j], 31), bits);
        auto codes = _mm256_andnot_si256(zero, _mm256_or_si256(mantissa, sign));
        half[j] = _mm_packus_epi32(_mm256_castsi256_si128(codes), _mm256_extracti128_si256(codes, 1));
    }
    store_codes(_mm_packus_epi16(half[0], half[1]), out, bits);
}

template <int bits, typename T>
__attribute__((target("avx512f,avx512bw,avx512vl,avx2"), always_inline)) static inline void row_avx512(
    const T* src, uint8_t* exp_out, uint8_t* out) {
    __m512i words;
    if constexpr (sizeof(T) == 4) {
        words = _mm512_loadu_si512(src);
    } else {
        words = _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(src))), 16);
    }
    auto exps = _mm512_and_si512(_mm512_srli_epi32(words, 23), _mm512_set1_epi32(255));
    auto max8 = _mm256_max_epu32(_mm512_castsi512_si256(exps), _mm512_extracti64x4_epi64(exps, 1));
    auto max4 = _mm_max_epu32(_mm256_castsi256_si128(max8), _mm256_extracti128_si256(max8, 1));
    max4 = _mm_max_epu32(max4, _mm_shuffle_epi32(max4, 0x4e));
    max4 = _mm_max_epu32(max4, _mm_shuffle_epi32(max4, 0xb1));
    *exp_out = static_cast<uint8_t>(_mm_cvtsi128_si32(max4));
    auto mantissa = _mm512_ternarylogic_epi32(words, _mm512_set1_epi32(0x7fffff), _mm512_set1_epi32(0x800000), 0xea);
    mantissa = _mm512_srlv_epi32(mantissa, _mm512_sub_epi32(_mm512_broadcastd_epi32(max4), exps));

    auto odd = _mm512_and_si512(_mm512_srli_epi32(mantissa, 24 - bits), _mm512_set1_epi32(1));
    mantissa = _mm512_srli_epi32(
        _mm512_add_epi32(mantissa, _mm512_add_epi32(odd, _mm512_set1_epi32((1u << (23 - bits)) - 1))), 24 - bits);
    mantissa = _mm512_min_epu32(mantissa, _mm512_set1_epi32((1 << bits) - 1));
    auto valid = _mm512_cmpneq_epi32_mask(exps, _mm512_setzero_si512()) &
                 _mm512_cmpneq_epi32_mask(mantissa, _mm512_setzero_si512());
    auto sign = _mm512_slli_epi32(_mm512_srli_epi32(words, 31), bits);
    auto codes = _mm512_maskz_mov_epi32(valid, _mm512_or_si512(mantissa, sign));
    store_codes(_mm512_cvtepi32_epi8(codes), out, bits);
}
#endif

template <int bits, typename T>
using RowPacker = void (*)(const T*, uint8_t*, uint8_t*);

// isa is exposed only to internal CPU tests: 0=auto, 1=scalar, 2=AVX2, 3=AVX512.
// Null means the explicitly requested instruction set is unavailable.
template <int bits, typename T>
static RowPacker<bits, T> select_row_packer(int isa = 0) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, bfloat16>);
    if (isa == 1) {
        return row_scalar<bits, T>;
    }
#if defined(__x86_64__)
    if ((isa == 0 || isa == 3) && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx2")) {
        return row_avx512<bits, T>;
    }
    if ((isa == 0 || isa == 2) && __builtin_cpu_supports("avx2")) {
        return row_avx2<bits, T>;
    }
#endif
    return isa == 0 ? row_scalar<bits, T> : nullptr;
}
}  // namespace tt::tt_metal::bfp_simd
