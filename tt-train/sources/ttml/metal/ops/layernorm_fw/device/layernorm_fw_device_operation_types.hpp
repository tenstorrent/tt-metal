// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::layernorm_fw::device {

// Attributes for the forward operation
struct operation_attributes_t {
    float epsilon;          // epsilon for numerical stability
    bool return_mean_rstd;  // whether to return mean and rstd for backward pass

    static constexpr auto attribute_names = std::forward_as_tuple("epsilon", "return_mean_rstd");
    auto attribute_values() const {
        return std::forward_as_tuple(epsilon, return_mean_rstd);
    }
};

// Tensors required for forward
struct tensor_args_t {
    ttnn::Tensor input;
    ttnn::Tensor gamma;  // scale parameter (learnable weight)
    ttnn::Tensor beta;   // shift parameter (learnable weight)
    std::optional<ttnn::Tensor> preallocated_output = std::nullopt;
    std::optional<ttnn::Tensor> preallocated_mean = std::nullopt;
    std::optional<ttnn::Tensor> preallocated_rstd = std::nullopt;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "input", "gamma", "beta", "preallocated_output", "preallocated_mean", "preallocated_rstd");
    auto attribute_values() const {
        return std::forward_as_tuple(input, gamma, beta, preallocated_output, preallocated_mean, preallocated_rstd);
    }
};

// Output tensor specs and tensors
using spec_return_value_t = std::vector<ttnn::TensorSpec>;
using tensor_return_value_t = std::vector<std::optional<ttnn::Tensor>>;

}  // namespace ttml::metal::ops::layernorm_fw::device
