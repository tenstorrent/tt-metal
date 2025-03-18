// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"

namespace ttml::ops::distributed {

autograd::TensorPtr scatter(const autograd::TensorPtr& tensor, int dim);
autograd::TensorPtr all_reduce(const autograd::TensorPtr& tensor);
autograd::TensorPtr all_gather(const autograd::TensorPtr& tensor, int dim);
autograd::TensorPtr broadcast(const autograd::TensorPtr& tensor);

}  // namespace ttml::ops::distributed
