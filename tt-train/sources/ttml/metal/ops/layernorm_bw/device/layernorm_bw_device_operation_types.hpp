// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::layernorm_bw::device {

// Attributes for the backward operation (add more if needed)
struct LayerNormBackwardParams {
    // Which grads to compute. A skipped grad has no output and its kernel work is skipped.
    bool compute_grad_input = true;
    bool compute_grad_gamma = true;
    bool compute_grad_beta = true;
};

// Tensors required for backward
struct LayerNormBackwardInputs {
    ttnn::Tensor input;
    ttnn::Tensor gamma;                                          // scale parameter (learnable weight)
    ttnn::Tensor mean;                                           // mean from forward pass
    ttnn::Tensor rstd;                                           // reciprocal std from forward pass
    ttnn::Tensor dL_dout;                                        // gradient from upstream
    std::optional<ttnn::Tensor> preallocated_dx = std::nullopt;  // dx (input gradient)
    std::optional<ttnn::Tensor> preallocated_dgamma_components = std::nullopt;  // dgamma components
    std::optional<ttnn::Tensor> preallocated_dbeta_components = std::nullopt;   // dbeta components
};

using operation_attributes_t = LayerNormBackwardParams;
using tensor_args_t = LayerNormBackwardInputs;

// Output tensor specs and tensors: [dx, dgamma_components, dbeta_components]. A skipped grad is nullopt.
using spec_return_value_t = std::vector<std::optional<tt::tt_metal::TensorSpec>>;
using tensor_return_value_t = std::vector<std::optional<ttnn::Tensor>>;

}  // namespace ttml::metal::ops::layernorm_bw::device
