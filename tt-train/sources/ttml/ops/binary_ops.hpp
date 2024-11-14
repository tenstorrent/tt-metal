// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
namespace ttml::ops {

autograd::TensorPtr operator+(const autograd::TensorPtr& a, const autograd::TensorPtr& b);
autograd::TensorPtr operator*(const autograd::TensorPtr& a, const autograd::TensorPtr& b);
autograd::TensorPtr operator-(const autograd::TensorPtr& a, const autograd::TensorPtr& b);
autograd::TensorPtr operator/(const autograd::TensorPtr& a, const autograd::TensorPtr& b);

autograd::TensorPtr add(const autograd::TensorPtr& a, const autograd::TensorPtr& b);
autograd::TensorPtr sub(const autograd::TensorPtr& a, const autograd::TensorPtr& b);
autograd::TensorPtr mul(const autograd::TensorPtr& a, const autograd::TensorPtr& b);
autograd::TensorPtr div(const autograd::TensorPtr& a, const autograd::TensorPtr& b);

}  // namespace ttml::ops
