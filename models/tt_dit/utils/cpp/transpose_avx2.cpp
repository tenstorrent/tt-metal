// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// SIMD byte-tile transposes backing the CHWT scatter in planar_concat.cpp.

#include "transpose_avx2.hpp"

#include <cstring>

#include <emmintrin.h>
#include <immintrin.h>
#include <tmmintrin.h>

namespace tt_dit_planar {

namespace {

// 16x16 byte transpose via the standard four-stage unpack network: each stage
// doubles the element width, so after epi8/16/32/64 every output lane holds one
// source column. This is the only primitive; the wider tiles compose from it.
inline void transpose_16x16_u8(const uint8_t* src, std::ptrdiff_t ss, uint8_t* dst, std::ptrdiff_t ds) {
    __m128i r[16];
    for (int i = 0; i < 16; ++i) {
        r[i] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i * ss));
    }

    __m128i a[16];
    for (int i = 0; i < 8; ++i) {
        a[2 * i] = _mm_unpacklo_epi8(r[2 * i], r[2 * i + 1]);
        a[2 * i + 1] = _mm_unpackhi_epi8(r[2 * i], r[2 * i + 1]);
    }

    __m128i b[16];
    for (int g = 0; g < 4; ++g) {
        const int s = g * 4;
        b[s + 0] = _mm_unpacklo_epi16(a[s + 0], a[s + 2]);
        b[s + 1] = _mm_unpackhi_epi16(a[s + 0], a[s + 2]);
        b[s + 2] = _mm_unpacklo_epi16(a[s + 1], a[s + 3]);
        b[s + 3] = _mm_unpackhi_epi16(a[s + 1], a[s + 3]);
    }

    __m128i c[16];
    for (int g = 0; g < 2; ++g) {
        const int s = g * 8;
        for (int i = 0; i < 4; ++i) {
            c[s + 2 * i] = _mm_unpacklo_epi32(b[s + i], b[s + 4 + i]);
            c[s + 2 * i + 1] = _mm_unpackhi_epi32(b[s + i], b[s + 4 + i]);
        }
    }

    for (int i = 0; i < 8; ++i) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + (2 * i) * ds), _mm_unpacklo_epi64(c[i], c[8 + i]));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + (2 * i + 1) * ds), _mm_unpackhi_epi64(c[i], c[8 + i]));
    }
}

}  // namespace

void transpose_32x32_u8(const uint8_t* src, std::ptrdiff_t src_stride, uint8_t* dst, std::ptrdiff_t dst_stride) {
    // Off-diagonal quadrants swap position under transposition.
    transpose_16x16_u8(src, src_stride, dst, dst_stride);
    transpose_16x16_u8(src + 16 * src_stride, src_stride, dst + 16, dst_stride);
    transpose_16x16_u8(src + 16, src_stride, dst + 16 * dst_stride, dst_stride);
    transpose_16x16_u8(src + 16 * src_stride + 16, src_stride, dst + 16 * dst_stride + 16, dst_stride);
}

void transpose_32xN_u8(const uint8_t* src, std::ptrdiff_t src_stride, uint8_t* dst, std::ptrdiff_t dst_stride, int n) {
    if (n <= 0) {
        return;
    }
    // Staging keeps the partial tile off the strided source/destination; the
    // padded lanes are never copied back out.
    alignas(32) uint8_t tmp_src[32 * 32];
    alignas(32) uint8_t tmp_dst[32 * 32];
    std::memset(tmp_src, 0, sizeof(tmp_src));
    for (int i = 0; i < 32; ++i) {
        std::memcpy(tmp_src + i * 32, src + i * src_stride, static_cast<size_t>(n));
    }
    transpose_32x32_u8(tmp_src, 32, tmp_dst, 32);
    for (int i = 0; i < n; ++i) {
        std::memcpy(dst + i * dst_stride, tmp_dst + i * 32, 32);
    }
}

void transpose_32x16_u8(const uint8_t* src, std::ptrdiff_t src_stride, uint8_t* dst, std::ptrdiff_t dst_stride) {
    // 16 source rows x 32 source cols: the two 16x16 halves stack vertically in dst.
    transpose_16x16_u8(src, src_stride, dst, dst_stride);
    transpose_16x16_u8(src + 16, src_stride, dst + 16 * dst_stride, dst_stride);
}

}  // namespace tt_dit_planar
