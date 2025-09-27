// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"

namespace ttml::ops {

autograd::TensorPtr relu(const autograd::TensorPtr& tensor);

// GELU with explicit approximation control.
// fast_and_approximate = true  → "tanh" approximation
// fast_and_approximate = false → exact/erf form
autograd::TensorPtr gelu(const autograd::TensorPtr& tensor, bool fast_and_approximate);

// Default GELU uses the fast (tanh) approximation to match device forward.
autograd::TensorPtr gelu(const autograd::TensorPtr& tensor);

autograd::TensorPtr silu(const autograd::TensorPtr& tensor, bool use_composite_bw = false);
autograd::TensorPtr tanh(const autograd::TensorPtr& tensor);
autograd::TensorPtr mean(const autograd::TensorPtr& tensor);
autograd::TensorPtr sum(const autograd::TensorPtr& tensor);
autograd::TensorPtr broadcast_batch(const autograd::TensorPtr& tensor, uint32_t new_batch_dim);
autograd::TensorPtr log_softmax(const autograd::TensorPtr& tensor, int dim);
autograd::TensorPtr log_softmax_moreh(const autograd::TensorPtr& tensor, int dim);

}  // namespace ttml::ops
