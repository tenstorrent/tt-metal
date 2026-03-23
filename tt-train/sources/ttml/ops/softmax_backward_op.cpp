// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/softmax_backward_op.hpp"

#include "autograd/tensor.hpp"
#include "metal/operations.hpp"

namespace ttml::ops {

autograd::TensorPtr softmax_backward(
    const autograd::TensorPtr& softmax_output,
    const autograd::TensorPtr& grad,
    int dim,
    std::optional<tt::tt_metal::CoreRangeSet> sub_core_grids) {
    auto result = ttml::metal::softmax_backward(softmax_output->get_value(), grad->get_value(), dim, sub_core_grids);
    return autograd::create_tensor(result);
}

}  // namespace ttml::ops
