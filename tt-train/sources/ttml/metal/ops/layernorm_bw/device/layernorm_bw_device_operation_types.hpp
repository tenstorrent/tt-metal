// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::layernorm_bw::device {

// Attributes for the backward operation (add more if needed)
struct operation_attributes_t {};

// Tensors required for backward
struct tensor_args_t {
    ttnn::Tensor input;
    ttnn::Tensor gamma;                                          // scale parameter (learnable weight)
    ttnn::Tensor mean;                                           // mean from forward pass
    ttnn::Tensor rstd;                                           // reciprocal std from forward pass
    ttnn::Tensor dL_dout;                                        // gradient from upstream
    std::optional<ttnn::Tensor> preallocated_dx = std::nullopt;  // dx (input gradient)
    std::optional<ttnn::Tensor> preallocated_dgamma_components = std::nullopt;  // dgamma components
    std::optional<ttnn::Tensor> preallocated_dbeta_components = std::nullopt;   // dbeta components
};

// Output tensor specs and tensors
using spec_return_value_t = std::vector<ttnn::TensorSpec>;
using tensor_return_value_t = std::vector<ttnn::Tensor>;

}  // namespace ttml::metal::ops::layernorm_bw::device
