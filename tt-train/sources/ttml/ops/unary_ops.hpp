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
// Preserves small log-probabilities and dominant-class gradients (arXiv:2610.31642, Sec. 2).
// Computes in FP32 from the native input; output and input gradient retain the native dtype.
// Sec. 3's teacher/student cancellation fix requires a loss accepting both sets of logits.
autograd::TensorPtr log_softmax(const autograd::TensorPtr& tensor, int dim);
autograd::TensorPtr log_softmax_moreh(const autograd::TensorPtr& tensor, int dim);
autograd::TensorPtr exp(const autograd::TensorPtr& tensor);
autograd::TensorPtr clip(const autograd::TensorPtr& tensor, float lo, float hi);
}  // namespace ttml::ops
