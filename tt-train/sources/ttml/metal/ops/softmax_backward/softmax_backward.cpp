// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_backward.hpp"

#include <tt_stl/assert.hpp>

#include "device/softmax_backward_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor softmax_backward(const ttnn::Tensor& softmax_output, const ttnn::Tensor& grad, int32_t dim) {
    const auto rank = static_cast<int32_t>(softmax_output.logical_shape().rank());
    int32_t normalized_dim = dim >= 0 ? dim : rank + dim;
    TT_FATAL(
        normalized_dim >= 0 && normalized_dim < rank,
        "Dimension {} is out of bounds for tensor with rank {}",
        dim,
        rank);
    return ttnn::prim::ttml_softmax_backward(softmax_output, grad, static_cast<uint32_t>(normalized_dim));
}

}  // namespace ttml::metal
