// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"

namespace ttml::ops {

autograd::TensorPtr swiglu(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& w1,
    const autograd::TensorPtr& w2,
    const autograd::TensorPtr& w3);

autograd::TensorPtr swiglu_optimized(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& w1,
    const autograd::TensorPtr& w2,
    const autograd::TensorPtr& w3);

// Same as swiglu_optimized but weights use LinearLayer convention: [out_features, in_features]
// (matmuls use transpose_b=true). Drop-in replacement for LlamaMLP.
autograd::TensorPtr swiglu_fused(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& w1,
    const autograd::TensorPtr& w2,
    const autograd::TensorPtr& w3,
    float dropout_prob = 0.0F);

}  // namespace ttml::ops
