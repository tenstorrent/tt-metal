// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// {dx, tail} = {input[:rows, :]^T, tail_scale * input[rows:rows + tail_rows, :]^T} for a bf16 TILE input
// [1, 1, R, T] with rows % 32 == 0, tail_rows <= 32 and R >= rows + tail_rows: dx is [1, 1, T, rows], tail
// [1, 1, T, tail_rows]. One pass over the input (one tile transpose per output tile) in place of the two slices
// and two transposes of the fused LoRA backward's `[in + TAIL, T]` product.
std::vector<ttnn::Tensor> transpose_split(
    const ttnn::Tensor& input, uint32_t rows, uint32_t tail_rows, float tail_scale = 1.0F, uint32_t dx_cols = 0);

}  // namespace ttml::metal
