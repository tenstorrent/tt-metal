// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Depthwise 4-tap conv1d along T of a [1, 1, T, C] ROW_MAJOR bf16 input; taps are [1, 1, 1, C] TILE bf16.
// Returns [1, 1, T, C] TILE bf16. With silu_grad, returns silu_grad * silu'(conv(input)).
ttnn::Tensor depthwise_conv1d_k4(
    const ttnn::Tensor& input,
    const ttnn::Tensor& tap0,
    const ttnn::Tensor& tap1,
    const ttnn::Tensor& tap2,
    const ttnn::Tensor& tap3,
    bool anti_causal = false,
    const std::optional<ttnn::Tensor>& silu_grad = std::nullopt);

}  // namespace ttml::metal
