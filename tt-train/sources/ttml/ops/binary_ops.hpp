// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/autocast_tensor.hpp"
#include "autograd/tensor.hpp"
namespace ttml::ops {

autograd::TensorPtr operator+(const autograd::TensorPtr& a, const autograd::AutocastTensor& b);
autograd::TensorPtr operator+(const autograd::TensorPtr& a, const autograd::TensorPtr& b);
autograd::TensorPtr operator*(const autograd::TensorPtr& a, const autograd::TensorPtr& b);
autograd::TensorPtr operator*(const autograd::TensorPtr& a, float b);
autograd::TensorPtr operator-(const autograd::TensorPtr& a, const autograd::TensorPtr& b);
autograd::TensorPtr operator/(const autograd::TensorPtr& a, const autograd::TensorPtr& b);

autograd::TensorPtr add(const autograd::TensorPtr& a, const autograd::AutocastTensor& b);
autograd::TensorPtr add(const autograd::TensorPtr& a, const autograd::TensorPtr& b);
autograd::TensorPtr sub(const autograd::TensorPtr& a, const autograd::TensorPtr& b);
autograd::TensorPtr mul(const autograd::TensorPtr& a, const autograd::TensorPtr& b);
autograd::TensorPtr mul(const autograd::TensorPtr& a, float b);
autograd::TensorPtr div(const autograd::TensorPtr& a, const autograd::TensorPtr& b);

}  // namespace ttml::ops

// operators should be in the same namespace as the TensorPtr
namespace ttml::autograd {
using ops::operator*;
using ops::operator+;
using ops::operator-;
using ops::operator/;

}  // namespace ttml::autograd
