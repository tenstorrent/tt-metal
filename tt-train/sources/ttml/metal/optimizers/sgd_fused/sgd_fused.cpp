// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sgd_fused.hpp"

#include "device/sgd_fused_device_operation.hpp"

namespace ttml::metal::optimizers::sgd_fused {

ttnn::Tensor SGDFusedOptimizer::invoke(
    const ttnn::Tensor& param_in, const ttnn::Tensor& grad, float lr, const std::optional<ttnn::Tensor>& param_out) {
    return ttnn::prim::ttml_sgd_fused(param_in, grad, lr, param_out);
}
}  // namespace ttml::metal::optimizers::sgd_fused
