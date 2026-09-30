// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "autograd/tensor.hpp"

namespace ttml::ops {

using Conv3dDims = std::array<uint32_t, 3>;

// 3D convolution with autograd for input, weight and (optional) bias, on top of ttnn::experimental::conv3d.
autograd::TensorPtr conv3d(
    const autograd::TensorPtr& input,
    const autograd::TensorPtr& weight,
    const autograd::TensorPtr& bias = nullptr,
    const Conv3dDims& stride = {1, 1, 1},
    const Conv3dDims& padding = {0, 0, 0},
    const Conv3dDims& dilation = {1, 1, 1},
    uint32_t groups = 1,
    const std::string& padding_mode = "zeros");

struct Conv3dPreparedWeight {
    ttnn::Shape weight_shape;
    uint32_t groups = 1;
    uint32_t c_in_block = 0;
    std::vector<ttnn::Tensor> forward;
    std::vector<ttnn::Tensor> transposed;
};

// with_transposed=false skips the input-gradient form (only the forward pass and dW/db use the forward form).
Conv3dPreparedWeight prepare_conv3d_weight(
    const ttnn::Tensor& weight, uint32_t groups = 1, bool with_transposed = true);

// Same op with caller-provided prepared weights; `weight` is still the autograd parameter that receives dW. The
// forward forms are required; if `prepared.transposed` is empty, the input-gradient form is built on the fly.
autograd::TensorPtr conv3d(
    const autograd::TensorPtr& input,
    const autograd::TensorPtr& weight,
    const autograd::TensorPtr& bias,
    const Conv3dPreparedWeight& prepared,
    const Conv3dDims& stride = {1, 1, 1},
    const Conv3dDims& padding = {0, 0, 0},
    const Conv3dDims& dilation = {1, 1, 1},
    uint32_t groups = 1,
    const std::string& padding_mode = "zeros");

}  // namespace ttml::ops
