// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <core/ttnn_all_includes.hpp>

#include "core/tt_tensor_utils.hpp"

namespace ttml::ttnn_fixed {

tt::tt_metal::Tensor sum_over_dim(const tt::tt_metal::Tensor& t, uint32_t dim);
tt::tt_metal::Tensor sum_over_batch(const tt::tt_metal::Tensor& t);
tt::tt_metal::Tensor log_softmax(const tt::tt_metal::Tensor& t, int dim);
tt::tt_metal::Tensor softmax(const tt::tt_metal::Tensor& t, int dim);
tt::tt_metal::Tensor divide(const tt::tt_metal::Tensor& a, const tt::tt_metal::Tensor& b);

tt::tt_metal::Tensor mean_moreh(const tt::tt_metal::Tensor& t, int dim, bool keep_dim);
tt::tt_metal::Tensor mean_ttnn(const tt::tt_metal::Tensor& t, int dim, bool keep_dim);

tt::tt_metal::Tensor sum_moreh(const tt::tt_metal::Tensor& t, int dim, bool keep_dim);
tt::tt_metal::Tensor sum_ttnn(const tt::tt_metal::Tensor& t, int dim, bool keep_dim);

// `seed_axes` (optional) lists the mesh axes across which the logits hold DISTINCT data and must
// therefore be seeded uniquely (e.g. dp / fsdp). Axes NOT listed are treated as replicated (e.g. tp)
// and get identical noise. When std::nullopt (the DEFAULT) NO axis is seeded uniquely: every device
// draws the same noise from `seed` (the original single-seed behavior). Callers needing distinct
// per-device noise (e.g. GRPO) MUST pass the sharded axes explicitly.
tt::tt_metal::Tensor sample(
    const tt::tt_metal::Tensor& t,
    float temperature,
    uint32_t seed,
    std::optional<tt::tt_metal::Tensor> logits_padding_mask = std::nullopt,
    std::optional<std::vector<uint32_t>> seed_axes = std::nullopt);

tt::tt_metal::Tensor to_l1_interleaved(const tt::tt_metal::Tensor& t);
tt::tt_metal::Tensor to_dram_interleaved(const tt::tt_metal::Tensor& t);

}  // namespace ttml::ttnn_fixed
