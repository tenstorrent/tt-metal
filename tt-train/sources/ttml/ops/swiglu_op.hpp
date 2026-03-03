// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"

namespace ttml::ops {

// Baseline SwiGLU: weights are [1,1,in,out], uses swiglu_fw kernel for forward.
autograd::TensorPtr swiglu(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& w1,
    const autograd::TensorPtr& w2,
    const autograd::TensorPtr& w3);

// Optimized SwiGLU: weights use LinearLayer convention [out_features, in_features].
// Composite forward (ttnn matmul + fused silu*multiply), fused swiglu_grad kernel
// for backward, in-place ops, 3 saved tensors. Drop-in replacement for LlamaMLP.
autograd::TensorPtr swiglu_optimized(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& w1,
    const autograd::TensorPtr& w2,
    const autograd::TensorPtr& w3,
    float dropout_prob = 0.0F);

}  // namespace ttml::ops
