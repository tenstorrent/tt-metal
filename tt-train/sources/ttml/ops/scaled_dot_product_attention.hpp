// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"

namespace ttml::ops {

// Uses fused sdpa_fw/sdpa_bw kernels for square attention. Rectangular attention requires an explicit mask and uses
// the composite fallback. For square attention with no mask, the fused kernels generate a causal mask on-device.
autograd::TensorPtr scaled_dot_product_attention(
    const autograd::TensorPtr& query,
    const autograd::TensorPtr& key,
    const autograd::TensorPtr& value,
    const std::optional<autograd::TensorPtr>& mask = std::nullopt,
    float dropout_probability = 0.0F);

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
