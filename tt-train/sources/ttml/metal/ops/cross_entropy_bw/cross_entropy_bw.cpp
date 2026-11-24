// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cross_entropy_bw.hpp"

#include "device/cross_entropy_bw_device_operation.hpp"
#include "metal/ops/log_softmax/device/row_softmax_program.hpp"

namespace ttml::metal {

ttnn::Tensor cross_entropy_bw(
    const ttnn::Tensor& input_tensor, const ttnn::Tensor& target_tensor, const ttnn::Tensor& grad, float scaler) {
    if (ops::row_softmax::can_fuse_upstream(input_tensor, grad)) {
        return ttnn::prim::ttml_cross_entropy_bw(input_tensor, target_tensor, scaler, std::nullopt, grad);
    }
    return ttnn::multiply(ttnn::prim::ttml_cross_entropy_bw(input_tensor, target_tensor, scaler), grad);
}

}  // namespace ttml::metal
