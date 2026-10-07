// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::depthwise_conv1d_k4::device {

struct DepthwiseConv1dK4Params {
    // false: out[t] = sum_j tap_j * x[t + j - 3]   (causal)
    // true:  out[t] = sum_j tap_j * x[t + 3 - j]   (anti-causal; the input-gradient of the causal conv)
    bool anti_causal = false;
};

struct DepthwiseConv1dK4Inputs {
    ttnn::Tensor input;  // [1, 1, T, C] ROW_MAJOR bf16
    ttnn::Tensor tap0;   // [1, 1, 1, C] TILE bf16
    ttnn::Tensor tap1;
    ttnn::Tensor tap2;
    ttnn::Tensor tap3;
    // When present, the output is silu_grad * silu'(conv(input)) instead of conv(input).
    std::optional<ttnn::Tensor> silu_grad = std::nullopt;  // [1, 1, T, C] TILE bf16
};

using operation_attributes_t = DepthwiseConv1dK4Params;
using tensor_args_t = DepthwiseConv1dK4Inputs;

using spec_return_value_t = tt::tt_metal::TensorSpec;
using tensor_return_value_t = ttnn::Tensor;

}  // namespace ttml::metal::ops::depthwise_conv1d_k4::device
