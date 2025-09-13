// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::optimizers::sgd_fused {

struct SGDFusedOptimizer {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& param_in, const ttnn::Tensor& grad, float lr, const std::optional<ttnn::Tensor>& param_out);
};
}  // namespace ttml::metal::optimizers::sgd_fused
