// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Row statistics of a [B, 1, S, V] bf16 TILE shard for the vocab-parallel cross-entropy forward (iteration 32 of the
// Mistral loop). mode 0: per (row, K slice) partial row max, the value broadcast over the tile's 32 columns; mode 1:
// per (row, K slice) partial sum of exp(x - max_col) in column 0 (the other columns 0), `max_col` [B, 1, S, 1] TILE.
// Output [B, 1, S, slices * 32] fp32 TILE; reduce it over the last dim with ttnn::max (mode 0) or ttnn::sum (mode 1).
ttnn::Tensor vp_ce_row_stats(
    const ttnn::Tensor& logits, uint32_t mode, const std::optional<ttnn::Tensor>& max_col = std::nullopt);

}  // namespace ttml::metal
