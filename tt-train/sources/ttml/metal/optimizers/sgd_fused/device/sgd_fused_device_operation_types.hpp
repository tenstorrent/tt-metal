// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <ttnn/tensor/tensor.hpp>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::optimizers::sgd_fused::device {

struct operation_attributes_t {
    float lr{};
    float momentum{0.0F};
    float dampening{0.0F};
    float weight_decay{0.0F};
    bool nesterov{false};
};

struct tensor_args_t {
    const ttnn::Tensor& param;
    const ttnn::Tensor& grad;
    std::optional<ttnn::Tensor> momentum_buffer = std::nullopt;
};

using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = ttnn::TensorSpec;

}  // namespace ttml::metal::optimizers::sgd_fused::device
