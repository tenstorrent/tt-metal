// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <core/ttnn_all_includes.hpp>

#include "core/tt_tensor_utils.hpp"

namespace ttml::ttnn_fixed {

ttnn::Tensor sum_over_dim(const ttnn::Tensor& t, uint32_t dim);
ttnn::Tensor sum_over_batch(const ttnn::Tensor& t);
ttnn::Tensor log_softmax(const ttnn::Tensor& t, int dim);
ttnn::Tensor softmax(const ttnn::Tensor& t, int dim);
ttnn::Tensor divide(const ttnn::Tensor& a, const ttnn::Tensor& b);

ttnn::Tensor mean_moreh(const ttnn::Tensor& t, int dim, bool keep_dim);
ttnn::Tensor mean_ttnn(const ttnn::Tensor& t, int dim, bool keep_dim);

ttnn::Tensor sum_moreh(const ttnn::Tensor& t, int dim, bool keep_dim);
ttnn::Tensor sum_ttnn(const ttnn::Tensor& t, int dim, bool keep_dim);

// `seed_axes` (optional) lists the mesh axes across which the logits hold DISTINCT data and must
// therefore be seeded uniquely (e.g. dp / fsdp). Axes NOT listed are treated as replicated (e.g. tp)
// and get identical noise. When std::nullopt (the DEFAULT) NO axis is seeded uniquely: every device
// draws the same noise from `seed` (the original single-seed behavior). Callers needing distinct
// per-device noise (e.g. GRPO) MUST pass the sharded axes explicitly.
// `positions` (optional): per-batch-row token position as a [B, 1, 1, 1] UINT32 ROW_MAJOR tensor,
// sharded with the SAME mapper as the batch. std::nullopt (default) samples every position and
// returns [B, 1, tokens, 1]; supplying it returns [B, 1, 1, 1] and reads only the tiles those rows
// live in.
ttnn::Tensor sample(
    const ttnn::Tensor& t,
    float temperature,
    uint32_t seed,
    std::optional<ttnn::Tensor> logits_mask = std::nullopt,
    std::optional<std::vector<uint32_t>> seed_axes = std::nullopt,
    std::optional<ttnn::Tensor> positions = std::nullopt);

ttnn::Tensor to_l1_interleaved(const ttnn::Tensor& t);
ttnn::Tensor to_dram_interleaved(const ttnn::Tensor& t);

}  // namespace ttml::ttnn_fixed
