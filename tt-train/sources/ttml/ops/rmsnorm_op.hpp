// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "autograd/tensor.hpp"

namespace ttml::ops {

autograd::TensorPtr rmsnorm(const autograd::TensorPtr& tensor, const autograd::TensorPtr& gamma, float epsilon);
autograd::TensorPtr rmsnorm_composite(
    const autograd::TensorPtr& tensor, const autograd::TensorPtr& gamma, float epsilon);

}  // namespace ttml::ops
