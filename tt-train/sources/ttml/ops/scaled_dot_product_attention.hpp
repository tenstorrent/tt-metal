// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"

namespace ttml::ops {

// Result struct for SDPA with scale values (needed for ring attention)
struct SDPAResult {
    autograd::TensorPtr output;
    autograd::TensorPtr sum_exp;  // sum(exp(qk_scaled - max)), shape (B, H, S, 1)
    autograd::TensorPtr exp_max;  // exp(max(qk_scaled)), shape (B, H, S, 1)
};

autograd::TensorPtr scaled_dot_product_attention(
    const autograd::TensorPtr& query,
    const autograd::TensorPtr& key,
    const autograd::TensorPtr& value,
    const std::optional<autograd::TensorPtr>& mask = std::nullopt);

// SDPA variant that returns scale values for ring attention online softmax combination.
// Returns {output, sum_exp, exp_max} where:
//   - output: attention output (B, H, S, D)
//   - sum_exp: sum(exp(qk_scaled - max)) per row (B, H, S, 1)
//   - exp_max: exp(max(qk_scaled)) per row (B, H, S, 1)
// The total unnormalized scale is sum_exp * exp_max = sum(exp(qk_scaled))
SDPAResult scaled_dot_product_attention_with_intermediates(
    const autograd::TensorPtr& query,
    const autograd::TensorPtr& key,
    const autograd::TensorPtr& value,
    const std::optional<autograd::TensorPtr>& mask = std::nullopt);

// Fused implementation using custom sdpa_fw and sdpa_bw kernels
// More efficient than composite implementation for training
autograd::TensorPtr scaled_dot_product_attention_fused(
    const autograd::TensorPtr& query,
    const autograd::TensorPtr& key,
    const autograd::TensorPtr& value,
    const std::optional<autograd::TensorPtr>& mask = std::nullopt,
    float dropout_probability = 0.0F);

autograd::TensorPtr scaled_sigmoid_dot_product_attention(
    const autograd::TensorPtr& query,
    const autograd::TensorPtr& key,
    const autograd::TensorPtr& value,
    const std::optional<autograd::TensorPtr>& mask = std::nullopt);

}  // namespace ttml::ops
