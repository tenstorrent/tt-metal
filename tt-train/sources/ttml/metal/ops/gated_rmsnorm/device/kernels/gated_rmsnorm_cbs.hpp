// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Circular-buffer map shared by the gated_rmsnorm program factories and kernels. Gt = V / 32 tiles
// per head; every push/pop on a streamed CB is a whole group of Gt tiles.

#pragma once

#include <cstdint>

namespace ttml_gated_rmsnorm_cb {

constexpr uint32_t x = 0U;        // bf16, 2*Gt: input tiles of the current item
constexpr uint32_t gate = 1U;     // bf16, 2*Gt
constexpr uint32_t gamma_b = 3U;  // bf16, Gt: gamma with row 0 broadcast to all 32 rows
constexpr uint32_t ones = 4U;     // bf16, 1: all-ones tile, X @ ones = row-sum of X in every column
constexpr uint32_t sq = 5U;       // f32, 1: sum_j x_j * x_j
constexpr uint32_t inv = 6U;      // f32, 1: rsqrt(mean(x^2) + eps), broadcast to all columns
constexpr uint32_t out = 7U;      // bf16, 2*Gt: fw out, bw dx
constexpr uint32_t dy = 8U;       // bf16, 2*Gt: bw only
constexpr uint32_t u = 9U;        // f32, Gt: x * gamma * inv
constexpr uint32_t du = 10U;      // f32, Gt: dy * silu(gate)
constexpr uint32_t prod = 11U;    // f32, Gt: u * du
constexpr uint32_t acc = 12U;     // f32, 1: sum_j prod_j
constexpr uint32_t t = 13U;       // f32, 1: rowsum(acc) * inv / V, broadcast
constexpr uint32_t dgate = 14U;   // bf16, 2*Gt: bw only
constexpr uint32_t dgamma = 15U;  // bf16, 2*Gt: bw + COMPUTE_DGAMMA only

}  // namespace ttml_gated_rmsnorm_cb
