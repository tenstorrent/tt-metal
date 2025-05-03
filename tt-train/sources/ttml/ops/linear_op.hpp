// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"

namespace ttml::ops {

autograd::TensorPtr linear_op(
    const autograd::TensorPtr& tensor, const autograd::TensorPtr& weight, const autograd::TensorPtr& bias);

void ttnn_linear_backward(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& weight,
    const autograd::TensorPtr& bias,
    const autograd::TensorPtr& out);

void moreh_linear_backward(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& weight,
    const autograd::TensorPtr& bias,
    const autograd::TensorPtr& out);

}  // namespace ttml::ops
