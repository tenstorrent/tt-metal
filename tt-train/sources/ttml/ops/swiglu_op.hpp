// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"

namespace ttml::ops {

// Canonical SwiGLU: weights use LinearLayer convention [out_features, in_features].
// Composite forward (ttnn matmul + fused silu*multiply), fused swiglu_elemwise_bw kernel
// for backward, in-place ops, 3 saved tensors. Drop-in replacement for LlamaMLP.
autograd::TensorPtr swiglu(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& w1,
    const autograd::TensorPtr& w2,
    const autograd::TensorPtr& w3,
    float dropout_prob = 0.0F);

// Baseline composite SwiGLU path (matmul + silu + mul + matmul + dropout).
// Keep this only as a benchmark/reference baseline for A/B measurements.
// Production model paths should call fused `swiglu(...)`.
autograd::TensorPtr swiglu_composite(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& w1,
    const autograd::TensorPtr& w2,
    const autograd::TensorPtr& w3,
    float dropout_prob = 0.0F);

// Backward-compatible alias for in-flight branches.
inline autograd::TensorPtr swiglu_optimized(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& w1,
    const autograd::TensorPtr& w2,
    const autograd::TensorPtr& w3,
    float dropout_prob = 0.0F) {
    return swiglu(tensor, w1, w2, w3, dropout_prob);
}

}  // namespace ttml::ops
