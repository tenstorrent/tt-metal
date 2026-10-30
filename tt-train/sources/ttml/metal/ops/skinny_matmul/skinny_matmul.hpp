// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// out = x @ W for a skinny W (N <= 64 columns): W = w^T with transpose_b (w is [N, K]), else w is [K, N].
// x is [B, C, M, K], w [1, 1, N, K] / [1, 1, K, N], all bf16 TILE in DRAM, K a multiple of 32; out is
// [B, C, M, N] bf16. HiFi4, fp32 accumulation.
//
// ttnn's matmul parallelises over output tiles, so an [M, K] x [K, 16] product uses M/32 cores (33 for a
// 1024-token LoRA `h = x @ A^T`). This op splits K: phase A computes per-(row-group, K-slice) fp32 partial
// products with the W slice resident in L1 and the x rows streamed, phase B sums the slices per output tile.
ttnn::Tensor skinny_matmul(const ttnn::Tensor& x, const ttnn::Tensor& w, bool transpose_b = false);

}  // namespace ttml::metal
