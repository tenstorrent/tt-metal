// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <core/ttnn_all_includes.hpp>

namespace ttml::ttnn_fixed {
ttnn::Tensor matmul(
    const ttnn::Tensor& a,
    const ttnn::Tensor& b,
    bool transpose_a = false,
    bool transpose_b = false,
    std::optional<ttnn::Tensor> output_tensor = std::nullopt);

struct MatmulBackwardGrads {
    bool a = true;
    bool b = true;
};

std::pair<std::optional<ttnn::Tensor>, std::optional<ttnn::Tensor>> matmul_backward(
    const ttnn::Tensor& a,
    const ttnn::Tensor& b,
    const ttnn::Tensor& out_grad,
    bool transpose_a = false,
    bool transpose_b = false,
    MatmulBackwardGrads grads = {});
}  // namespace ttml::ttnn_fixed
