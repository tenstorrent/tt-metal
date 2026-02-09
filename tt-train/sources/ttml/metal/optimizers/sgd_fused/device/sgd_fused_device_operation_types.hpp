// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <ttnn/tensor/tensor.hpp>
#include <tuple>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::optimizers::sgd_fused::device {

struct operation_attributes_t {
    float lr{};
    float momentum{0.0F};
    float dampening{0.0F};
    float weight_decay{0.0F};
    bool nesterov{false};

    static constexpr auto attribute_names =
        std::forward_as_tuple("lr", "momentum", "dampening", "weight_decay", "nesterov");
    auto attribute_values() const {
        return std::forward_as_tuple(lr, momentum, dampening, weight_decay, nesterov);
    }
};

struct tensor_args_t {
    const ttnn::Tensor& param;
    const ttnn::Tensor& grad;
    std::optional<ttnn::Tensor> momentum_buffer = std::nullopt;

    static constexpr auto attribute_names = std::forward_as_tuple("param", "grad", "momentum_buffer");
    auto attribute_values() const {
        return std::forward_as_tuple(param, grad, momentum_buffer);
    }
};

using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = ttnn::TensorSpec;

}  // namespace ttml::metal::optimizers::sgd_fused::device
