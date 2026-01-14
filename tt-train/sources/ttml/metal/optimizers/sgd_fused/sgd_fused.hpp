// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::optimizers::sgd_fused {

struct SGDFusedOptimizer {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& param_in,
        const ttnn::Tensor& grad,
        const float lr,
        const float momentum,
        const float dampening,
        const float weight_decay,
        const bool nesterov,
        const std::optional<ttnn::Tensor>& momentum_buffer);
};
}  // namespace ttml::metal::optimizers::sgd_fused
