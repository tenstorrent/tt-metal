// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace tt_dit_planar {

// All three routines write dst[i][j] = src[j][i]; `src` is indexed by the CHWT
// source's W axis and `dst` by its T axis, so the caller's "rows" swap meaning
// across the call.

// 32 source rows x 32 source cols.
void transpose_32x32_u8(const uint8_t* src, std::ptrdiff_t src_stride, uint8_t* dst, std::ptrdiff_t dst_stride);

// 32 source rows x `n` source cols, n in [1, 32]. Reads only `n` bytes per
// source row, so a short trailing source is never over-read.
void transpose_32xN_u8(const uint8_t* src, std::ptrdiff_t src_stride, uint8_t* dst, std::ptrdiff_t dst_stride, int n);

// 16 source rows x 32 source cols.
void transpose_32x16_u8(const uint8_t* src, std::ptrdiff_t src_stride, uint8_t* dst, std::ptrdiff_t dst_stride);

}  // namespace tt_dit_planar
