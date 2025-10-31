// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "autograd/tensor.hpp"

namespace ttml::ops {

// LayerNorm operation with configurable epsilon and hardware clamping control
// Default eps=1e-5F is safe for most cases; BERT uses 1e-12F (see bert.hpp)
// enable_hardware_clamp: When true, applies max(eps, min_safe_eps) for BFLOAT16 safety
// min_safe_eps: Configurable threshold for clamping (default 1e-4F for bfloat16)
autograd::TensorPtr layernorm(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& gamma,
    const autograd::TensorPtr& beta,
    float eps = 1e-5F,
    bool enable_hardware_clamp = true,
    float min_safe_eps = 1e-4F);

// Composite LayerNorm (manual implementation) with configurable epsilon and hardware clamping
autograd::TensorPtr composite_layernorm(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& gamma,
    const autograd::TensorPtr& beta,
    float eps = 1e-5F,
    bool enable_hardware_clamp = true,
    float min_safe_eps = 1e-4F);

}  // namespace ttml::ops
