// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace tt_dit_planar {

// Transpose a 32-row x 32-col tile of uint8.
//
// `src` is the top-left of a 32x32 byte block where consecutive rows are
// `src_stride` bytes apart and bytes within a row are contiguous (stride 1).
// `dst` is the top-left of the transposed tile, with rows spaced by
// `dst_stride` bytes (and inner stride 1).
//
// AVX2 only — uses _mm256_unpack* + _mm256_permute2x128_si256.
void transpose_32x32_u8(const uint8_t* src, std::ptrdiff_t src_stride, uint8_t* dst, std::ptrdiff_t dst_stride);

// Transpose a 32-row x N-col tile (N < 32) of uint8.  Used to handle the
// trailing T values when the T dimension isn't a multiple of 32.
//
// Reads 32 source rows but only writes the top `n_cols` destination rows
// (i.e. the trailing N values become destination rows; the lower 32-N
// destination rows of the 32x32 block are skipped).
//
// Internally pads to 32 cols (loads 32 source bytes, ignores the upper part)
// then writes only the first `n_cols` rows of the transposed result.
void transpose_32xN_u8(
    const uint8_t* src, std::ptrdiff_t src_stride, uint8_t* dst, std::ptrdiff_t dst_stride, int n_dst_rows);

// Transpose a 32-row x 16-col tile of uint8.  Used for the W tail when
// `w_per` is 16 mod 32 (e.g. UV plane: w_per_uv=80 = 2*32+16).
//
// `src` covers 32 source rows of 16 contiguous columns each.  `dst` writes
// 16 destination rows of 32 contiguous columns each.
void transpose_16x32_u8(const uint8_t* src, std::ptrdiff_t src_stride, uint8_t* dst, std::ptrdiff_t dst_stride);

// Transpose a 16-row x 32-col tile of uint8 (the inverse shape of
// transpose_16x32_u8).  Source has 16 rows of 32 contiguous columns each;
// destination has 32 rows of 16 contiguous columns each.
//
// Used by the CHWT scatter when w_tail is exactly 16: 16 W-source rows ×
// 32 T-source cols → 32 T-dest rows × 16 W-dest cols.  Replaces the prior
// 32x32 zero-padded bounce path for that case.
void transpose_32x16_u8(const uint8_t* src, std::ptrdiff_t src_stride, uint8_t* dst, std::ptrdiff_t dst_stride);

}  // namespace tt_dit_planar
