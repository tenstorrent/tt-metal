// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// out = x @ W for a skinny W (N <= 64 columns): W = w^T with transpose_b (w is [N, K]), else w is [K, N].
// x is [B, C, M, K], w [1, 1, N, K] / [1, 1, K, N], all bf16 TILE in DRAM, K a multiple of 32; out is
// [B, C, M, N] bf16. HiFi4, fp32 accumulation.
//
// ttnn's matmul parallelises over output tiles, so an [M, K] x [K, 16] product uses M/32 cores (33 for a
// 1024-token LoRA `h = x @ A^T`). This op splits K: phase A computes per-(row-group, K-slice) fp32 partial
// products with the W slice resident in L1 and the x rows streamed, phase B sums the slices per output tile.
// x may be wider than w's K: only its first K columns enter the product (the wide LoRA input buffer). With
// `tail_out` (same rows as x) the output is also written in place at tile column `tail_col` of it, followed by
// zero tiles up to `tail_tiles` columns: the LoRA h written straight into the buffer's tail (iteration 24).
ttnn::Tensor skinny_matmul(
    const ttnn::Tensor& x,
    const ttnn::Tensor& w,
    bool transpose_b = false,
    const std::optional<ttnn::Tensor>& tail_out = std::nullopt,
    uint32_t tail_col = 0,
    uint32_t tail_tiles = 0,
    // iteration 40: x stored transposed ([1, 1, K', M]: the transposed wide LoRA buffer) and / or the output
    // transposed ([1, 1, N, M]; the tail target then takes the tiles in its rows tail_col .. tail_col + tail_tiles)
    bool x_transposed = false,
    bool transpose_out = false);

}  // namespace ttml::metal
