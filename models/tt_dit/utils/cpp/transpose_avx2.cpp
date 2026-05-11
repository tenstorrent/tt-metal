// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// AVX2/SSE2 byte tile transposes used by the CHWT planar concat path.
//
// The "32×32" transpose is composed from four independent 16×16 SSE2
// transposes plus an implicit TR↔BL swap (achieved by the call sites
// addressing the four sub-blocks).  Every 16×16 transpose uses the standard
// 4-stage byte/word/dword/qword unpack pattern.

#include "transpose_avx2.hpp"

#include <cstring>
#include <emmintrin.h>  // SSE2
#include <immintrin.h>  // AVX2

namespace tt_dit_planar {

// ---------------------------------------------------------------------------
// 16×16 byte transpose (SSE2 only).
//
// Standard recursive transpose: 4 stages of unpack-{lo,hi} at byte / word /
// dword / qword granularity.  After all 4 stages, output[i] = column i of
// the input (treating each 128-bit register as a row of 16 bytes).
// ---------------------------------------------------------------------------

static inline void transpose16x16_u8_sse(const uint8_t* src, std::ptrdiff_t s, uint8_t* dst, std::ptrdiff_t d) {
    __m128i v[16];
    for (int i = 0; i < 16; ++i) {
        v[i] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i * s));
    }

    __m128i a[16];

    // Stage 1 — stride-1 byte unpack.
    for (int i = 0; i < 8; ++i) {
        a[2 * i + 0] = _mm_unpacklo_epi8(v[2 * i + 0], v[2 * i + 1]);
        a[2 * i + 1] = _mm_unpackhi_epi8(v[2 * i + 0], v[2 * i + 1]);
    }

    // Stage 2 — stride-2 word (16-bit) unpack.
    // Pairs: (a[0],a[2]), (a[1],a[3]), (a[4],a[6]), (a[5],a[7]), …
    for (int g = 0; g < 4; ++g) {
        for (int j = 0; j < 2; ++j) {
            v[4 * g + j + 0] = _mm_unpacklo_epi16(a[4 * g + j], a[4 * g + j + 2]);
            v[4 * g + j + 2] = _mm_unpackhi_epi16(a[4 * g + j], a[4 * g + j + 2]);
        }
    }

    // Stage 3 — stride-4 dword (32-bit) unpack.
    // Pairs: (v[0],v[4]), (v[1],v[5]), (v[2],v[6]), (v[3],v[7]), (v[8],v[12]), …
    for (int g = 0; g < 2; ++g) {
        for (int j = 0; j < 4; ++j) {
            a[8 * g + j + 0] = _mm_unpacklo_epi32(v[8 * g + j], v[8 * g + j + 4]);
            a[8 * g + j + 4] = _mm_unpackhi_epi32(v[8 * g + j], v[8 * g + j + 4]);
        }
    }

    // Stage 4 — stride-8 qword (64-bit) unpack.
    // Pairs: (a[0],a[8]), (a[1],a[9]), …
    for (int j = 0; j < 8; ++j) {
        v[j + 0] = _mm_unpacklo_epi64(a[j], a[j + 8]);
        v[j + 8] = _mm_unpackhi_epi64(a[j], a[j + 8]);
    }

    // The 4-stage stride-1/2/4/8 unpack produces outputs in bit-reversed
    // order: v[i] holds the column at index `bit_reverse4(i)` of the input.
    // Store with a fixed permutation table to undo the bit-reversal.
    static constexpr int kBitRev4[16] = {
        0,
        8,
        4,
        12,
        2,
        10,
        6,
        14,
        1,
        9,
        5,
        13,
        3,
        11,
        7,
        15,
    };
    for (int i = 0; i < 16; ++i) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + kBitRev4[i] * d), v[i]);
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void transpose_32x32_u8(const uint8_t* src, std::ptrdiff_t src_stride, uint8_t* dst, std::ptrdiff_t dst_stride) {
    // Decompose into four independent 16×16 transposes with TR↔BL swap.
    // - input[0..15][0..15]  → output[0..15][0..15]   (TL → TL transposed)
    // - input[0..15][16..31] → output[16..31][0..15]  (TR → BL transposed)
    // - input[16..31][0..15] → output[0..15][16..31]  (BL → TR transposed)
    // - input[16..31][16..31]→ output[16..31][16..31] (BR → BR transposed)
    transpose16x16_u8_sse(src + 0 * src_stride + 0, src_stride, dst + 0 * dst_stride + 0, dst_stride);
    transpose16x16_u8_sse(src + 0 * src_stride + 16, src_stride, dst + 16 * dst_stride + 0, dst_stride);
    transpose16x16_u8_sse(src + 16 * src_stride + 0, src_stride, dst + 0 * dst_stride + 16, dst_stride);
    transpose16x16_u8_sse(src + 16 * src_stride + 16, src_stride, dst + 16 * dst_stride + 16, dst_stride);
}

void transpose_16x32_u8(const uint8_t* src, std::ptrdiff_t src_stride, uint8_t* dst, std::ptrdiff_t dst_stride) {
    // 32 source rows × 16 source cols → 16 dest rows × 32 dest cols.
    // - source rows [0..15], cols [0..15]  → dest rows [0..15], cols [0..15]
    // - source rows [16..31], cols [0..15] → dest rows [0..15], cols [16..31]
    transpose16x16_u8_sse(src + 0 * src_stride, src_stride, dst + 0, dst_stride);
    transpose16x16_u8_sse(src + 16 * src_stride, src_stride, dst + 16, dst_stride);
}

void transpose_32x16_u8(const uint8_t* src, std::ptrdiff_t src_stride, uint8_t* dst, std::ptrdiff_t dst_stride) {
    // 16 source rows × 32 source cols → 32 dest rows × 16 dest cols.
    // - source rows [0..15], cols [0..15]  → dest rows [0..15], cols [0..15]
    // - source rows [0..15], cols [16..31] → dest rows [16..31], cols [0..15]
    transpose16x16_u8_sse(src + 0, src_stride, dst + 0 * dst_stride, dst_stride);
    transpose16x16_u8_sse(src + 16, src_stride, dst + 16 * dst_stride, dst_stride);
}

void transpose_32xN_u8(
    const uint8_t* src, std::ptrdiff_t src_stride, uint8_t* dst, std::ptrdiff_t dst_stride, int n_dst_rows) {
    // 32 source rows × `n_dst_rows` valid source cols (= T values).  Output
    // is `n_dst_rows` dst rows × 32 dst cols.  Cols past `n_dst_rows` in
    // source may be out-of-bounds (the next pixel's bytes), so we copy the
    // 32 × `n_dst_rows` valid region into a 32×32 temp buffer with zero
    // padding before transposing — safe and compact (1 KiB stack).
    if (n_dst_rows <= 0) {
        return;
    }
    if (n_dst_rows >= 32) {
        transpose_32x32_u8(src, src_stride, dst, dst_stride);
        return;
    }

    alignas(32) uint8_t tmp_in[32 * 32] = {0};
    for (int i = 0; i < 32; ++i) {
        std::memcpy(tmp_in + i * 32, src + i * src_stride, n_dst_rows);
    }

    alignas(32) uint8_t tmp_out[32 * 32];
    transpose_32x32_u8(tmp_in, 32, tmp_out, 32);

    for (int i = 0; i < n_dst_rows; ++i) {
        std::memcpy(dst + i * dst_stride, tmp_out + i * 32, 32);
    }
}

}  // namespace tt_dit_planar
