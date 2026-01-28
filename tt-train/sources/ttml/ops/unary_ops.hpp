// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"

namespace ttml::ops {

// Activation functions
autograd::TensorPtr relu(const autograd::TensorPtr& tensor);
autograd::TensorPtr gelu(const autograd::TensorPtr& tensor);
autograd::TensorPtr silu(const autograd::TensorPtr& tensor, bool use_composite_bw = false);

// Element-wise math ops
autograd::TensorPtr exp(const autograd::TensorPtr& tensor);
autograd::TensorPtr reciprocal(const autograd::TensorPtr& tensor);
autograd::TensorPtr neg(const autograd::TensorPtr& tensor);

// Reduction ops
autograd::TensorPtr mean(const autograd::TensorPtr& tensor);
autograd::TensorPtr sum(const autograd::TensorPtr& tensor, int dim, bool keepdim = true);
autograd::TensorPtr max(const autograd::TensorPtr& tensor, int dim, bool keepdim = true);

// Utility ops
autograd::TensorPtr broadcast_batch(const autograd::TensorPtr& tensor, uint32_t new_batch_dim);
autograd::TensorPtr log_softmax(const autograd::TensorPtr& tensor, int dim);
autograd::TensorPtr log_softmax_moreh(const autograd::TensorPtr& tensor, int dim);

// Softmax using moreh fused kernels
autograd::TensorPtr softmax(const autograd::TensorPtr& tensor, int dim);

}  // namespace ttml::ops
