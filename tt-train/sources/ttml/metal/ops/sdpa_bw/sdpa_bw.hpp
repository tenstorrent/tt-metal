// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::sdpa_bw {

struct SDPABackwardOperation {
    static std::vector<ttnn::Tensor> invoke(
        const ttnn::Tensor& grad_output,     // Gradient w.r.t. output
        const ttnn::Tensor& query,           // Original Q (needed for gradients)
        const ttnn::Tensor& key,             // Original K (needed for gradients)  
        const ttnn::Tensor& value,           // Original V (needed for gradients)
        const std::optional<ttnn::Tensor>& attn_mask,  // attn mask
        const ttnn::Tensor& intermediates,   // From forward pass (attention weights, etc.)
        const float dropout_probability = 0.0F,
        const bool fp32_dest_acc_en = true);
};

}  // namespace ttml::metal::ops::sdpa_bw
