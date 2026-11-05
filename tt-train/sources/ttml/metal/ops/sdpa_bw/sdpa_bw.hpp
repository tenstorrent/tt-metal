// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/common/const_utils.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Which grads sdpa_bw computes. A field set to false drops that grad's kernel work; grad_gate also needs a gate.
struct SDPABackwardGrads {
    bool query = true;
    bool key = true;
    bool value = true;
    bool gate = true;
};

// Returns [grad_Q, grad_K, grad_V, grad_gate]; a grad is nullopt when it was not requested (or, for grad_gate,
// without a gate). With a gate, attn_output must be the gated forward output Y = sdpa(Q, K, V) * sigmoid(gate).
std::tuple<
    std::optional<ttnn::Tensor>,
    std::optional<ttnn::Tensor>,
    std::optional<ttnn::Tensor>,
    std::optional<ttnn::Tensor>>
sdpa_bw(
    const ttnn::Tensor& grad_output,                              // Gradient w.r.t. output
    const ttnn::Tensor& attn_output,                              // sdpa forward output (needed for gradients)
    const ttnn::Tensor& query,                                    // input Q (needed for gradients)
    const ttnn::Tensor& key,                                      // input K (needed for gradients)
    const ttnn::Tensor& value,                                    // input V (needed for gradients)
    const ttnn::Tensor& intermediates,                            // From forward pass (max_val, 1/sum_exp values)
    AttentionMaskType mask_type = AttentionMaskType::Arbitrary,   // Mask type (None, Causal, or Arbitrary)
    const std::optional<ttnn::Tensor>& attn_mask = std::nullopt,  // attention mask (only for Arbitrary)
    const float dropout_probability = 0.0F,
    const std::optional<ttnn::Tensor>& gate =
        std::nullopt,  // out = sdpa(Q, K, V) * sigmoid(gate); gate shape (B, H, S, Dv)
    const std::optional<ttnn::Tensor>& preallocated_grad_gate =
        std::nullopt,  // grad_gate output buffer, shape (B, H, S, Dv); may alias attn_output; needs grads.gate
    const SDPABackwardGrads& grads = {});

}  // namespace ttml::metal
