// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sgd_fused.hpp"

#include "device/sgd_fused_device_operation.hpp"

namespace ttml::metal::optimizers::sgd_fused {

ttnn::Tensor SGDFusedOptimizer::invoke(
    const ttnn::Tensor& param,
    const ttnn::Tensor& grad,
    const float lr,
    const float momentum,
    const float dampening,
    const float weight_decay,
    const bool nesterov,
    const std::optional<ttnn::Tensor>& momentum_buffer) {
    return ttnn::prim::ttml_sgd_fused(param, grad, lr, momentum, dampening, weight_decay, nesterov, momentum_buffer);
}
}  // namespace ttml::metal::optimizers::sgd_fused
