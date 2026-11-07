// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "autograd/tensor.hpp"

namespace ttml::ops {

autograd::TensorPtr dropout(const autograd::TensorPtr& tensor, float probability);

// Whether dropout on `value` draws a seed per device rather than one seed for every device, read from its label.
bool dropout_seed_per_device(const ttnn::Tensor& value);

}  // namespace ttml::ops
