// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Circular-buffer map shared by the skinny_matmul program factories and kernels (both phases).
namespace skinny_matmul_cb {

// Phase A
constexpr uint32_t x = 0;            // [2*block] bf16 x tiles of one row, streamed along K
constexpr uint32_t w = 1;            // [slice_tiles*Nt] bf16 W slice, resident for the item (tile (k, n) at k*Nt + n)
constexpr uint32_t partial_out = 2;  // [2*Nt] fp32 partial products of one row

// Phase B
constexpr uint32_t partials = 3;  // [2*S] fp32 the slices of one (row, n)
constexpr uint32_t out = 4;       // [2] bf16
constexpr uint32_t zero = 5;      // [1] bf16 zero tile (REDUCE writer, tail columns past N)

}  // namespace skinny_matmul_cb
