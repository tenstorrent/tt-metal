// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <ttnn/tensor/tensor.hpp>
#include <tuple>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::optimizers::sgd::device {

struct operation_attributes_t {
    float lr{};
    float momentum{0.0F};
    float dampening{0.0F};
    float weight_decay{0.0F};
    bool nesterov{false};

    static constexpr auto attribute_names = std::forward_as_tuple("nesterov");
    auto attribute_values() const {
        return std::forward_as_tuple(nesterov);
    }
};

struct tensor_args_t {
    const ttnn::Tensor& param;
    const ttnn::Tensor& grad;
    std::optional<ttnn::Tensor> momentum_buffer = std::nullopt;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "momentum_initialized", "param_dtype", "param_logical_shape", "param", "grad", "momentum_buffer");
    auto attribute_values() const {
        return std::make_tuple(
            momentum_buffer.has_value(),
            param.dtype(),
            std::cref(param.logical_shape()),
            std::cref(param),
            std::cref(grad),
            momentum_buffer);
    }
};

using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = ttnn::TensorSpec;

}  // namespace ttml::metal::optimizers::sgd::device
