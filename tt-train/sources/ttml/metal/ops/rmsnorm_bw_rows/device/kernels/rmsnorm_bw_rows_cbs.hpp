// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace rmsnorm_bw_rows_cb {

constexpr uint32_t a = 0;            // [St] bf16 input slice, resident (pushed in groups of `block` as it arrives)
constexpr uint32_t gamma = 1;        // [St] bf16 gamma slice, resident (same)
constexpr uint32_t dy = 2;           // [St] bf16 dL_dout slice, resident (same)
constexpr uint32_t zero = 3;         // [1] bf16 zeros (bcast FPU ops accumulate into DEST)
constexpr uint32_t partial_out = 4;  // [1] fp32 this core's row dot product over its slice
constexpr uint32_t rms = 5;          // [1] bf16 rms tile of the row (column 0 valid)
constexpr uint32_t partials = 6;     // [S] fp32 the row's S partial tiles (slot s written by the row's core s)
constexpr uint32_t ones = 7;         // [1] bf16 all-ones
constexpr uint32_t ones_row0 = 8;    // [1] bf16 ones in row 0
constexpr uint32_t inv = 9;          // [1] fp32 1/rms broadcast along columns
constexpr uint32_t acc = 10;         // [1] fp32 sum of the partial tiles
constexpr uint32_t t = 11;           // [1] fp32 P / (C * rms^2) broadcast along columns
constexpr uint32_t dx = 12;          // [2*block] bf16 dL_da
constexpr uint32_t dres = 13;        // [2*block] bf16 residual_grad (ADD_RESIDUAL_GRAD), streamed
constexpr uint32_t dx_copy = 14;  // [2*block] bf16 dL_da packed a second time, the transpose's source (DX_TRANSPOSED)
constexpr uint32_t dx_t = 15;     // [2*block] bf16 dL_da^T tiles (DX_TRANSPOSED)
constexpr uint32_t acc1 = 16;     // [1] fp32 pass 1's running dot product between groups (iteration 41)

}  // namespace rmsnorm_bw_rows_cb
