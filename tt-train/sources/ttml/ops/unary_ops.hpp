// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>

#include "autograd/tensor.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

namespace ttml::ops {

using GeluVariant = ttnn::operations::unary::GeluVariant;

// Parses a GELU variant name. Accepted spellings (lowercase, exact):
//   "none", "accurate" -> ACCURATE   ("none" mirrors torch's and gelu_bw's `approximate=` vocabulary)
//   "tanh"             -> TANH
//   "fast_lut"         -> FAST_LUT
// Throws std::invalid_argument otherwise; nanobind maps that to a Python ValueError.
GeluVariant gelu_variant_from_string(std::string_view name);

autograd::TensorPtr relu(const autograd::TensorPtr& tensor);
// ACCURATE and TANH each pair their forward with a matching backward kernel. FAST_LUT is
// forward-only -- ttnn has no LUT backward kernel -- so it throws std::invalid_argument if the input
// requires grad while gradient mode is enabled; use it only for inference.
autograd::TensorPtr gelu(const autograd::TensorPtr& tensor, GeluVariant variant = GeluVariant::ACCURATE);
autograd::TensorPtr silu(const autograd::TensorPtr& tensor, bool use_composite_bw = false);
autograd::TensorPtr mean(const autograd::TensorPtr& tensor);
// autograd::TensorPtr sum(const autograd::TensorPtr& tensor);
autograd::TensorPtr broadcast_batch(const autograd::TensorPtr& tensor, uint32_t new_batch_dim);
autograd::TensorPtr log_softmax(const autograd::TensorPtr& tensor, int dim);
autograd::TensorPtr log_softmax_moreh(const autograd::TensorPtr& tensor, int dim);
autograd::TensorPtr exp(const autograd::TensorPtr& tensor);
autograd::TensorPtr clip(const autograd::TensorPtr& tensor, float lo, float hi);
}  // namespace ttml::ops
