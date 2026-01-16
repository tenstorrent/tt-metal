// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::sdpa_bw {

struct SDPABackwardOperation {
    // Returns [grad_Q, grad_K, grad_V]
    static std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> invoke(
        const ttnn::Tensor& grad_output,               // Gradient w.r.t. output
        const ttnn::Tensor& attn_output,               // sdap forward output (needed for gradients)
        const ttnn::Tensor& query,                     // input Q (needed for gradients)
        const ttnn::Tensor& key,                       // input K (needed for gradients)
        const ttnn::Tensor& value,                     // input V (needed for gradients)
        const std::optional<ttnn::Tensor>& attn_mask,  // attention mask
        const ttnn::Tensor& intermediates,             // From forward pass (attention weights, etc.)
        const float dropout_probability = 0.0F,
        const bool fp32_dest_acc_en = true);
};

}  // namespace ttml::metal::ops::sdpa_bw
