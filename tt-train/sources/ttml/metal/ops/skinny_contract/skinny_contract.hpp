// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// out = scale * small^T @ big, contracting the row (token) dimension: small is [1, 1, T, r] with r <= 32,
// big [1, 1, T, N], both bf16 TILE in DRAM, T a multiple of 32; out is [1, 1, r, N] bf16, or its transpose
// [1, 1, N, r] with transpose_out. The LoRA gradients dB = s g^T h and dA = dh^T x: one work item per output
// column tile, the transposed small operand resident in L1 (T/32 tiles), the big operand streamed once in all,
// HiFi4 with fp32 accumulation; ttnn's matmul with transpose_a runs them on 33-88 cores.
// big_transposed (iteration 38): big is [1, 1, N, T]; small_transposed: small is already [1, 1, r, T].
ttnn::Tensor skinny_contract(
    const ttnn::Tensor& small,
    const ttnn::Tensor& big,
    float scale = 1.0F,
    bool transpose_out = false,
    bool big_transposed = false,
    bool small_transposed = false,
    uint32_t grid_rows = 0U);

}  // namespace ttml::metal
