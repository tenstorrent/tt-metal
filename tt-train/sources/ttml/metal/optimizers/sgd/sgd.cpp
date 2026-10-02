// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sgd.hpp"

#include "autograd/autocast_tensor.hpp"
#include "device/sgd_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor sgd(
    const autograd::MutableTensorView& param,
    const ttnn::Tensor& grad,
    const float lr,
    const float momentum,
    const float dampening,
    const float weight_decay,
    const bool nesterov,
    const autograd::MutableTensorView* momentum_buffer) {
    const auto momentum_buffer_tensor =
        momentum_buffer != nullptr ? std::optional<ttnn::Tensor>(momentum_buffer->tensor()) : std::nullopt;
    return ttnn::prim::sgd(
        param.tensor(), grad, lr, momentum, dampening, weight_decay, nesterov, momentum_buffer_tensor);
}

}  // namespace ttml::metal
