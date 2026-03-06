// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sgd.hpp"

#include "device/sgd_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor sgd(
    const ttnn::Tensor& param,
    const ttnn::Tensor& grad,
    const float lr,
    const float momentum,
    const float dampening,
    const float weight_decay,
    const bool nesterov,
    const std::optional<ttnn::Tensor>& momentum_buffer) {
    return ttnn::prim::sgd(param, grad, lr, momentum, dampening, weight_decay, nesterov, momentum_buffer);
}

}  // namespace ttml::metal
