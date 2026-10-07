// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Circular-buffer map shared by the gated_rmsnorm forward/backward program factories and kernels.
// All activations are bf16; the reduction scratch tiles are fp32.
namespace gated_rmsnorm_cb {

// Inputs (shared fw/bw)
constexpr uint32_t x = 0;        // [2*Gt] bf16   input tiles of the current (row, group)
constexpr uint32_t gate = 1;     // [2*Gt] bf16
constexpr uint32_t gamma = 2;    // [Gt]   bf16   raw gamma tiles (row 0 valid)
constexpr uint32_t gamma_b = 3;  // [Gt]   bf16   gamma broadcast down all 32 rows (built by the reader)
constexpr uint32_t ones = 4;     // [1]    bf16   all-ones tile: X @ ones == row-sum broadcast to every column
constexpr uint32_t sq = 5;       // [1]    fp32   sum_j x_j^2 (elementwise over the group's tiles)
constexpr uint32_t inv = 6;      // [1]    fp32   rsqrt(mean(x^2) + eps), broadcast to all columns
constexpr uint32_t out = 7;      // [2*Gt] bf16   fw: output; bw: dL/dinput

// Backward only
constexpr uint32_t dy = 8;       // [2*Gt] bf16
constexpr uint32_t u = 9;        // [Gt]   fp32   x * gamma * inv (the un-gated normalized output)
constexpr uint32_t du = 10;      // [Gt]   fp32   dy * silu(gate)
constexpr uint32_t prod = 11;    // [Gt]   fp32   u * du
constexpr uint32_t acc = 12;     // [1]    fp32   sum_j prod_j
constexpr uint32_t t = 13;       // [1]    fp32   (sum_c u*du) * inv / group, broadcast to all columns
constexpr uint32_t dgate = 14;   // [2*Gt] bf16   dL/dgate
constexpr uint32_t dgamma = 15;  // [2*Gt] bf16   dL/dgamma components (x * inv * du)

}  // namespace gated_rmsnorm_cb
