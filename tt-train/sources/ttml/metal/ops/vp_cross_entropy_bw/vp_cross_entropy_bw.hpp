// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Backward of the vocab-parallel cross-entropy in one pass over the local logits shard:
//   grad[r, c] = scale * g * (exp(logits[r, c] - log_normalizer[r]) - [c + first_v == target[r]])
// logits [B, 1, S, local_V] bf16 TILE; log_normalizer [B, 1, S, 1] fp32 TILE (the global log-sum-exp per row);
// target [B, S] ROW_MAJOR uint32 global indices; grad one bf16 / fp32 tile holding g (the loss's upstream
// gradient) in every element;
// first_v the shard's first global column (plus rank * local_V along cluster_axis on a mesh).
ttnn::Tensor vp_cross_entropy_bw(
    const ttnn::Tensor& logits,
    const ttnn::Tensor& log_normalizer,
    const ttnn::Tensor& target,
    const ttnn::Tensor& grad,
    uint32_t local_V,
    std::optional<uint32_t> cluster_axis = std::nullopt,
    uint32_t first_v = 0U,
    float scale = 1.0F);

}  // namespace ttml::metal
