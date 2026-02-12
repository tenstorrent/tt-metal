// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttml::metal {

/** Softmax backward: output = y * (grad - sum(y * grad, dim, keepdim=True)). Supports last dimension only. */
ttnn::Tensor softmax_backward(const ttnn::Tensor& softmax_output, const ttnn::Tensor& grad, int32_t dim);

}  // namespace ttml::metal
