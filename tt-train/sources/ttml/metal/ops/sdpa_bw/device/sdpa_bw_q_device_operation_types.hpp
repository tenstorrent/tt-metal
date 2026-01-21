// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::sdpa_bw::device::q {

struct operation_attributes_t {
    bool fp32_dest_acc_en{true};
    float dropout_probability{0.0F};
};

struct tensor_args_t {
    const ttnn::Tensor& grad_output;               // Gradient w.r.t. output
    const ttnn::Tensor& attn_output;               // sdap forward output (needed for gradients)
    const ttnn::Tensor& query;                     // Input Q (needed for gradients)
    const ttnn::Tensor& key;                       // Input K (needed for gradients)
    const ttnn::Tensor& value;                     // Input V (needed for gradients)
    const std::optional<ttnn::Tensor>& attn_mask;  // attention mask
    const ttnn::Tensor& intermediates;             // From forward pass (max_val,1/sum_exp values)

    // Preallocated gradient tensor (optional)
    std::optional<ttnn::Tensor> preallocated_grad_query;
};

using tensor_return_value_t = ttnn::Tensor;  // [grad_Q]

using spec_return_value_t = ttnn::TensorSpec;

}  // namespace ttml::metal::ops::sdpa_bw::device::q
