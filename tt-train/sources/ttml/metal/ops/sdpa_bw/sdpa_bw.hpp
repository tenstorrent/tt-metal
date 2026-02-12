// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/common/const_utils.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Returns [grad_Q, grad_K, grad_V]
std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> sdpa_bw(
    const ttnn::Tensor& grad_output,                              // Gradient w.r.t. output
    const ttnn::Tensor& attn_output,                              // sdpa forward output (needed for gradients)
    const ttnn::Tensor& query,                                    // input Q (needed for gradients)
    const ttnn::Tensor& key,                                      // input K (needed for gradients)
    const ttnn::Tensor& value,                                    // input V (needed for gradients)
    const ttnn::Tensor& intermediates,                            // From forward pass (max_val, 1/sum_exp values)
    AttentionMaskType mask_type = AttentionMaskType::Arbitrary,   // Mask type (None, Causal, or Arbitrary)
    const std::optional<ttnn::Tensor>& attn_mask = std::nullopt,  // attention mask (only for Arbitrary)
    const float dropout_probability = 0.0F);

}  // namespace ttml::metal
