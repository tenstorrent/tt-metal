// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "metal/common/const_utils.hpp"

namespace ttml::ops {

// Fused implementation using custom sdpa_fw and sdpa_bw kernels (default)
// When no mask is provided, uses on-device causal mask generation
// mask_type overrides this default; pass AttentionMaskType::None for unmasked attention.
autograd::TensorPtr scaled_dot_product_attention(
    const autograd::TensorPtr& query,
    const autograd::TensorPtr& key,
    const autograd::TensorPtr& value,
    const std::optional<autograd::TensorPtr>& mask = std::nullopt,
    float dropout_probability = 0.0F,
    std::optional<ttml::metal::AttentionMaskType> mask_type = std::nullopt);

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
