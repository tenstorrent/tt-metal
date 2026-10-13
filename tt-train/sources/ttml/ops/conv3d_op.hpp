// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include "autograd/tensor.hpp"

namespace ttml::ops {

using Conv3dDims = std::array<uint32_t, 3>;

// Unpadded, dense 3D convolution with autograd for input, weight and (optional) bias, on top of
// ttnn::experimental::conv3d.
autograd::TensorPtr conv3d(
    const autograd::TensorPtr& input,
    const autograd::TensorPtr& weight,
    const autograd::TensorPtr& bias = nullptr,
    const Conv3dDims& stride = {1, 1, 1},
    const Conv3dDims& dilation = {1, 1, 1});

struct Conv3dPreparedWeight {
    ttnn::Shape weight_shape;
    uint32_t c_in_block = 0;
    ttnn::Tensor forward;
    std::optional<ttnn::Tensor> transposed;
};

// with_transposed=false skips the input-gradient form (only the forward pass and dW/db use the forward form).
// The forms are bfloat16, the precision conv3d reads the autograd weight at; a float32 weight is cast first.
Conv3dPreparedWeight prepare_conv3d_weight(const ttnn::Tensor& weight, bool with_transposed = true);

// Same op with a caller-provided prepared weight; `weight` is still the autograd parameter that receives dW. The
// forward form is required; without `prepared.transposed`, the input-gradient form is built on the fly.
autograd::TensorPtr conv3d(
    const autograd::TensorPtr& input,
    const autograd::TensorPtr& weight,
    const autograd::TensorPtr& bias,
    const Conv3dPreparedWeight& prepared,
    const Conv3dDims& stride = {1, 1, 1},
    const Conv3dDims& dilation = {1, 1, 1});

}  // namespace ttml::ops
