// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/softmax_backward_op.hpp"

namespace ttml::ops {

autograd::TensorPtr softmax_backward(
    const autograd::TensorPtr& softmax_output,
    const autograd::TensorPtr& grad,
    int dim,
    std::optional<tt::tt_metal::CoreRangeSet> sub_core_grids) {
    throw std::runtime_error("softmax_backward operation has been removed");
}

}  // namespace ttml::ops
