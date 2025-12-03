// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <ttnn/tensor/tensor.hpp>

namespace ttml::metal::optimizers::adamw_full_precision::device {

struct operation_attributes_t {
    float lr{};
    float beta1{};
    float beta2{};
    float beta1_pow{};
    float beta2_pow{};
    float epsilon{};
    float weight_decay{};
    bool amsgrad{false};
    uint32_t step{};
};

struct tensor_args_t {
    const ttnn::Tensor& param;  // fp32 - master weights
    const ttnn::Tensor& grad;   // bf16 - gradients

    const ttnn::Tensor& exp_avg;                                // fp32 - first moment
    const ttnn::Tensor& exp_avg_sq;                             // fp32 - second moment
    std::optional<ttnn::Tensor> max_exp_avg_sq = std::nullopt;  // fp32 - for amsgrad
};

using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = ttnn::TensorSpec;

}  // namespace ttml::metal::optimizers::adamw_full_precision::device
