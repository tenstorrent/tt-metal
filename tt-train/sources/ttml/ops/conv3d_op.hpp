// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include "autograd/tensor.hpp"

namespace ttml::ops {

using Conv3dDims = std::array<uint32_t, 3>;

// Dense, zero-padded 3D convolution with autograd for input, weight and (optional) bias, on top of
// ttnn::experimental::conv3d.
autograd::TensorPtr conv3d(
    const autograd::TensorPtr& input,
    const autograd::TensorPtr& weight,
    const autograd::TensorPtr& bias = nullptr,
    const Conv3dDims& stride = {1, 1, 1},
    const Conv3dDims& padding = {0, 0, 0},
    const Conv3dDims& dilation = {1, 1, 1});

}  // namespace ttml::ops
