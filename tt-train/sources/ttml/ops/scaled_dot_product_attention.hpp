// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"

namespace ttml::ops {

// Fused implementation using custom sdpa_fw and sdpa_bw kernels (default)
// When no mask is provided, uses on-device causal mask generation
// When a gate is provided, the output is sdpa(Q, K, V) * sigmoid(gate); gate has the output's
// shape (B, H, S, Dv) and receives its own gradient in backward.
autograd::TensorPtr scaled_dot_product_attention(
    const autograd::TensorPtr& query,
    const autograd::TensorPtr& key,
    const autograd::TensorPtr& value,
    const std::optional<autograd::TensorPtr>& mask = std::nullopt,
    float dropout_probability = 0.0F,
    const std::optional<autograd::TensorPtr>& gate = std::nullopt);

// Composite implementation using individual TTNN ops (fallback)
autograd::TensorPtr scaled_dot_product_attention_composite(
    const autograd::TensorPtr& query,
    const autograd::TensorPtr& key,
    const autograd::TensorPtr& value,
    const std::optional<autograd::TensorPtr>& mask = std::nullopt);

autograd::TensorPtr scaled_sigmoid_dot_product_attention(
    const autograd::TensorPtr& query,
    const autograd::TensorPtr& key,
    const autograd::TensorPtr& value,
    const std::optional<autograd::TensorPtr>& mask = std::nullopt);

}  // namespace ttml::ops
