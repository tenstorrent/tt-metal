// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Circular-buffer map shared by the rmsnorm_fw program factories and kernels (both phases).
namespace rmsnorm_fw_cb {

// Streamed input (both phases), `block` tiles at a time
constexpr uint32_t x = 0;     // [2*block] bf16
constexpr uint32_t zero = 1;  // [1] bf16 all-zeros (bcast FPU ops accumulate into DEST, so DEST is zeroed first)

// Phase A (partial sums of squares)
constexpr uint32_t mask = 2;         // [1] bf16 column mask for the last tile when C % 32 != 0
constexpr uint32_t partial_out = 3;  // [2] fp32 sum over the slice of x^2

// Phase B (rms + output)
constexpr uint32_t gamma = 4;     // [2*block] bf16, streamed alongside x
constexpr uint32_t partials = 5;  // [2*S] fp32 phase-A outputs for the row
constexpr uint32_t ones = 6;      // [1] bf16 all-ones: X @ ones == row-sum broadcast along columns
constexpr uint32_t acc = 7;       // [1] fp32 sum of the partial tiles
constexpr uint32_t rms = 8;       // [1] fp32 sqrt(rowsum(acc) / C + eps) broadcast along columns
constexpr uint32_t inv = 9;       // [1] fp32 1 / rms broadcast along columns
constexpr uint32_t y = 10;        // [2*block] bf16 output x * gamma * inv
constexpr uint32_t rms_out = 11;  // [2] bf16 rms tile written once per tile-row (RETURN_RMS)

}  // namespace rmsnorm_fw_cb
