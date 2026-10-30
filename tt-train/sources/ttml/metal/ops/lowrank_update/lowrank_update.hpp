// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// out = acc + scale * (a @ b), with b = w^T when transpose_b (w is [1, 1, N, K]), else w is [1, 1, K, N].
// acc is [B, C, M, N], a [B, C, M, K] with K <= 32 (one tile), all bf16 TILE in DRAM; out has acc's shape.
// The rank-16 LoRA products `base + s * h @ B^T` and `dx + dh @ A` in one pass: the product is formed per output
// tile in fp32 DEST (HiFi4), the acc tile added there and the sum rounded once to bf16, instead of a 33-core
// matmul writing a [M, N] bf16 intermediate that a separate add reads back.
ttnn::Tensor lowrank_update(
    const ttnn::Tensor& acc,
    const ttnn::Tensor& a,
    const ttnn::Tensor& w,
    float scale = 1.0F,
    bool transpose_b = false);

}  // namespace ttml::metal
