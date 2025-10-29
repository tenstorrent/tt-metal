// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"

namespace ttml::ops {

enum ReduceType : uint8_t { NONE = 0, MEAN = 1, SUM = 2 };

autograd::TensorPtr mse_loss(
    const autograd::TensorPtr& prediction, const autograd::TensorPtr& target, ReduceType reduce = ReduceType::MEAN);

autograd::TensorPtr cross_entropy_loss(
    const autograd::TensorPtr& prediction, const autograd::TensorPtr& target, ReduceType reduce = ReduceType::MEAN);

autograd::TensorPtr nll_loss(
    const autograd::TensorPtr& prediction, const autograd::TensorPtr& target, ReduceType reduce = ReduceType::MEAN);

}  // namespace ttml::ops
