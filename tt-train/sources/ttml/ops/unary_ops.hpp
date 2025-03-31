// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"

namespace ttml::ops {

autograd::TensorPtr relu(const autograd::TensorPtr& tensor);
autograd::TensorPtr gelu(const autograd::TensorPtr& tensor);
autograd::TensorPtr silu(const autograd::TensorPtr& tensor);
autograd::TensorPtr mean(const autograd::TensorPtr& tensor);
autograd::TensorPtr sum(const autograd::TensorPtr& tensor);
autograd::TensorPtr broadcast_batch(const autograd::TensorPtr& tensor, uint32_t new_batch_dim);
autograd::TensorPtr log_softmax(const autograd::TensorPtr& tensor, int dim);
autograd::TensorPtr log_softmax_moreh(const autograd::TensorPtr& tensor, int dim);
}  // namespace ttml::ops
