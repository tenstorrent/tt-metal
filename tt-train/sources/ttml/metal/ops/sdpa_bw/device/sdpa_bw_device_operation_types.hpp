// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::sdpa_bw::device {

struct operation_attributes_t {
    bool fp32_dest_acc_en{true};
    float dropout_probability{0.0F};
};

struct tensor_args_t {
    const ttnn::Tensor& grad_output;      // Gradient w.r.t. output
    const ttnn::Tensor& query;            // Original Q (needed for gradients)
    const ttnn::Tensor& key;              // Original K (needed for gradients)
    const ttnn::Tensor& value;            // Original V (needed for gradients)
    const std::optional<ttnn::Tensor>& attn_mask;  // attention mask
    const ttnn::Tensor& intermediates;    // From forward pass (1/sum_exp values)

    // Preallocated gradient tensors (optional)
    std::optional<ttnn::Tensor> preallocated_grad_query;
    std::optional<ttnn::Tensor> preallocated_grad_key;
    std::optional<ttnn::Tensor> preallocated_grad_value;
};

using tensor_return_value_t = std::vector<ttnn::Tensor>;  // [grad_Q, grad_K, grad_V]

using spec_return_value_t = std::vector<ttnn::TensorSpec>;

}  // namespace ttml::metal::ops::sdpa_bw::device
