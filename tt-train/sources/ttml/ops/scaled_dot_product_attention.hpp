// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"

namespace ttml::ops {

// Fused implementation using custom sdpa_fw and sdpa_bw kernels (default)
// When no mask is provided and no_mask is false, uses on-device causal mask
// generation. When no_mask is true, the mask argument is ignored and
// AttentionMaskType::None is used instead: the fused kernels skip mask-bias
// application entirely (cheaper than an explicit all-ones "Arbitrary" mask,
// which computes and adds a zero bias for no semantic reason). Use this for
// plain bidirectional attention where every position is always valid, e.g.
// no static causal or padding structure exists.
autograd::TensorPtr scaled_dot_product_attention(
    const autograd::TensorPtr& query,
    const autograd::TensorPtr& key,
    const autograd::TensorPtr& value,
    const std::optional<autograd::TensorPtr>& mask = std::nullopt,
    float dropout_probability = 0.0F,
    bool no_mask = false);

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
