// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::layernorm_fw::device {

// Attributes for the forward operation
struct LayerNormForwardParams {
    float epsilon;          // epsilon for numerical stability
    bool return_mean_rstd;  // whether to return mean and rstd for backward pass
};

// Tensors required for forward
struct LayerNormForwardInputs {
    ttnn::Tensor input;
    ttnn::Tensor gamma;  // scale parameter (learnable weight)
    ttnn::Tensor beta;   // shift parameter (learnable weight)
    std::optional<ttnn::Tensor> preallocated_output = std::nullopt;
    std::optional<ttnn::Tensor> preallocated_mean = std::nullopt;
    std::optional<ttnn::Tensor> preallocated_rstd = std::nullopt;
};

using operation_attributes_t = LayerNormForwardParams;
using tensor_args_t = LayerNormForwardInputs;

// Output tensor specs and tensors
using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
using tensor_return_value_t = std::vector<std::optional<ttnn::Tensor>>;

}  // namespace ttml::metal::ops::layernorm_fw::device
