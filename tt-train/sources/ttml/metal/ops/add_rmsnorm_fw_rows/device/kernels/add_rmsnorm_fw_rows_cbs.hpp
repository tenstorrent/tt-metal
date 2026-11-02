// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace add_rmsnorm_fw_rows_cb {

constexpr uint32_t residual = 0;     // [2*group] bf16 residual, streamed in groups of four tiles
constexpr uint32_t delta = 1;        // [2*group] bf16 delta, streamed in groups of four tiles
constexpr uint32_t gamma = 2;        // [St] bf16 gamma slice, resident
constexpr uint32_t sum = 3;          // [St] bf16 residual + delta: written to DRAM by the writer, re-read in pass 2
constexpr uint32_t zero = 4;         // [1] bf16 zeros (bcast FPU ops accumulate into DEST)
constexpr uint32_t ones = 5;         // [1] bf16 all-ones: X @ ones == row-sum broadcast along columns
constexpr uint32_t acc = 6;          // [1] fp32 running sum of squares between groups of pass 1
constexpr uint32_t partial_out = 7;  // [1] fp32 this core's sum of squares over its slice
constexpr uint32_t partials = 8;     // [S] fp32 the row's S partial tiles (slot s written by the row's core s)
constexpr uint32_t row_acc = 9;      // [1] fp32 sum of the partial tiles
constexpr uint32_t rms_tmp = 10;     // [1] fp32 rms broadcast along columns
constexpr uint32_t rms_out = 11;     // [1] bf16 rms tile (written by slice 0)
constexpr uint32_t inv = 12;         // [1] fp32 1 / rms broadcast along columns
constexpr uint32_t out = 13;         // [2*block] bf16 sum * gamma / rms
constexpr uint32_t delta_t =
    14;  // [2*group] bf16 DELTA_TRANSPOSED: delta's tiles transposed through DEST (iteration 41)

}  // namespace add_rmsnorm_fw_rows_cb
