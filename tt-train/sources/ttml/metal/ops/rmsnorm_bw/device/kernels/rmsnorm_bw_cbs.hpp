// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Circular-buffer map shared by the rmsnorm_bw program factories and kernels (both phases).
namespace rmsnorm_bw_cb {

// Streamed inputs (both phases), `block` tiles at a time
constexpr uint32_t a = 0;      // [2*block] bf16
constexpr uint32_t gamma = 1;  // [2*block] bf16
constexpr uint32_t dy = 2;     // [2*block] bf16
constexpr uint32_t zero = 3;   // [1] bf16 all-zeros (bcast FPU ops accumulate into DEST, so DEST is zeroed first)

// Phase A (partial sums)
constexpr uint32_t mask = 4;         // [1] bf16 column mask for the last tile when C % 32 != 0
constexpr uint32_t partial_out = 5;  // [2] fp32 sum over the slice of a * gamma * dy

// Phase B (gradients)
constexpr uint32_t rms = 6;        // [2] bf16 rms tile of the row (column 0 valid)
constexpr uint32_t partials = 7;   // [2*S] fp32 phase-A outputs for the row
constexpr uint32_t ones = 8;       // [1] bf16 all-ones: X @ ones == row-sum broadcast along columns
constexpr uint32_t ones_row0 = 9;  // [1] bf16 ones in row 0: X @ ones_row0 == column 0 of X broadcast along columns
constexpr uint32_t inv = 10;       // [1] fp32 1/rms broadcast along columns
constexpr uint32_t acc = 11;       // [1] fp32 sum of the partial tiles
constexpr uint32_t t = 12;         // [1] fp32 P / (C * rms^2) broadcast along columns
constexpr uint32_t dx = 13;        // [2*block] bf16 dL_da
constexpr uint32_t dgamma = 14;    // [2*block] bf16 dL_dgamma components

}  // namespace rmsnorm_bw_cb
