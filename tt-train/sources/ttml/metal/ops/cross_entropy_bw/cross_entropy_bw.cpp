// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cross_entropy_bw.hpp"

#include "device/cross_entropy_bw_device_operation.hpp"

namespace ttml::metal::ops::cross_entropy_bw {

ttnn::Tensor CrossEntropyBackwardOperation::invoke(
    const ttnn::Tensor& input_tensor, const ttnn::Tensor& target_tensor, const ttnn::Tensor& grad, float scaler) {
    return ttnn::multiply(ttnn::prim::ttml_cross_entropy_bw(input_tensor, target_tensor, scaler), grad);
}
}  // namespace ttml::metal::ops::cross_entropy_bw
