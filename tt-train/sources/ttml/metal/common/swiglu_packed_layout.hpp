// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The [gate | up] tile layout the swiglu_packed kernels share. Token-major: a packed row is 2*Wt tiles, gate in
// [0, Wt) and up in [Wt, 2*Wt). Transposed (iteration 39, packed^T [2I, T]): a row of one half is Ht tiles, the
// gate rows come first and the up rows Wt*Ht tiles later. Work is handed out in blocks of block_size tiles along a
// row of one half; the factories pass the row stride and the gate-to-up offset of the layout in use.

#pragma once

#include <cstdint>

inline uint32_t swiglu_packed_block_tile(
    uint32_t b, uint32_t blocks_per_row, uint32_t row_stride, uint32_t block_size) {
    return (b / blocks_per_row) * row_stride + (b % blocks_per_row) * block_size;
}
