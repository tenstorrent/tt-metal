// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

namespace ttml::ops {

// Map to ttnn.GeluVariant.
using GeluVariant = ttnn::operations::unary::GeluVariant;

autograd::TensorPtr relu(const autograd::TensorPtr& tensor);
autograd::TensorPtr gelu(const autograd::TensorPtr& tensor, GeluVariant variant = GeluVariant::ACCURATE);
autograd::TensorPtr silu(const autograd::TensorPtr& tensor, bool use_composite_bw = false);
autograd::TensorPtr mean(const autograd::TensorPtr& tensor);
// autograd::TensorPtr sum(const autograd::TensorPtr& tensor);
autograd::TensorPtr broadcast_batch(const autograd::TensorPtr& tensor, uint32_t new_batch_dim);
autograd::TensorPtr log_softmax(const autograd::TensorPtr& tensor, int dim);
autograd::TensorPtr log_softmax_moreh(const autograd::TensorPtr& tensor, int dim);
autograd::TensorPtr exp(const autograd::TensorPtr& tensor);
autograd::TensorPtr clip(const autograd::TensorPtr& tensor, float lo, float hi);

autograd::TensorPtr sigmoid(const autograd::TensorPtr& tensor);

// Sum along a single axis, keeping the reduced dim. Backward broadcasts the
// gradient back over that axis.
autograd::TensorPtr sum_over_dim(const autograd::TensorPtr& tensor, int dim);

// Cumulative sum along `dim`. The backward of a forward cumsum is a reverse
// cumsum (grad_i = sum_{j>=i} g_j), which ttnn::cumsum provides directly via
// its reverse_order flag.
autograd::TensorPtr cumsum(const autograd::TensorPtr& tensor, int dim);

// softplus(x) = log(1 + exp(beta * x)) / beta, linear above `threshold`.
autograd::TensorPtr softplus(const autograd::TensorPtr& tensor, float beta = 1.0F, float threshold = 20.0F);

// L2-normalize along the last dim: y = x * rsqrt(sum(x^2) + epsilon).
autograd::TensorPtr l2_norm(const autograd::TensorPtr& tensor, float epsilon = 1e-6F);

// Shift forward along `dim` by `shift`, zero-filling the vacated front:
// out[t] = tensor[t - shift] for t >= shift, else 0. This is the building block
// of a causal depthwise conv: a kernel-K causal conv is
// sum_j w_j * shift_along_dim(x, dim, K - 1 - j).
autograd::TensorPtr shift_along_dim(const autograd::TensorPtr& tensor, int dim, int shift);

// Swap two dims. Backward applies the same swap to the gradient.
autograd::TensorPtr transpose(const autograd::TensorPtr& tensor, int dim0, int dim1);
}  // namespace ttml::ops
