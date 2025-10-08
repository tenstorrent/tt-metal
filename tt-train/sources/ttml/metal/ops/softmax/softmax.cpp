// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax.hpp"

#include "device/softmax_device_operation.hpp"

namespace ttml::metal::ops::softmax {

ttnn::Tensor SoftmaxOperation::invoke(const ttnn::Tensor& input_tensor, int32_t dim) {
    return ttnn::prim::ttml_softmax(input_tensor, dim);
}
}  // namespace ttml::metal::ops::softmax
