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

ttnn::Tensor sample(
    const ttnn::Tensor& t,
    float temperature,
    uint32_t seed,
    std::optional<ttnn::Tensor> logits_padding_mask = std::nullopt);

ttnn::Tensor to_l1_interleaved(const ttnn::Tensor& t);
ttnn::Tensor to_dram_interleaved(const ttnn::Tensor& t);

}  // namespace ttml::ttnn_fixed
