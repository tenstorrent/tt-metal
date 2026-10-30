// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Circular-buffer map shared by the add_rmsnorm_fw program factories and kernels (both phases).
namespace add_rmsnorm_fw_cb {

// Phase A (sum and partial sums of squares)
constexpr uint32_t residual = 0;     // [2*block] bf16
constexpr uint32_t delta = 1;        // [2*block] bf16
constexpr uint32_t sum_out = 2;      // [2*block] bf16 residual + delta
constexpr uint32_t acc = 3;          // [1] fp32 running sum of squares between blocks of one item
constexpr uint32_t partial_out = 4;  // [2] fp32 sum of squares over the slice

// Phase B (rms and output)
constexpr uint32_t sum = 5;       // [2*block] bf16 phase-A sum
constexpr uint32_t gamma = 6;     // [2*block] bf16
constexpr uint32_t partials = 7;  // [2*S] fp32 phase-A outputs of the row
constexpr uint32_t ones = 8;      // [1] bf16 all-ones: X @ ones == row-sum broadcast along columns
constexpr uint32_t zero = 9;      // [1] bf16 all-zeros (bcast FPU ops accumulate into DEST, so DEST is zeroed first)
constexpr uint32_t row_acc = 10;  // [1] fp32 sum of the partial tiles
constexpr uint32_t rms_tmp = 11;  // [1] fp32 rms broadcast along columns
constexpr uint32_t rms_out = 12;  // [2] bf16 rms tile (written for slice 0)
constexpr uint32_t inv = 13;      // [1] fp32 1 / rms broadcast along columns
constexpr uint32_t out = 14;      // [2*block] bf16 sum * gamma / rms

}  // namespace add_rmsnorm_fw_cb
