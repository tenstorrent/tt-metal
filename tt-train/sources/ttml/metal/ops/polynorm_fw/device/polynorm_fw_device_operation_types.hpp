// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::polynorm3_fw::device {

struct PolyNorm3FWAttributes {
    float epsilon{1e-5F};
};

struct PolyNorm3FWTensorArgs {
    const ttnn::Tensor& input;
    const ttnn::Tensor& weight;
    const ttnn::Tensor& bias;
    std::optional<ttnn::Tensor> preallocated_output = std::nullopt;
};

using PolyNorm3FWSpecReturn = std::vector<ttnn::TensorSpec>;
using PolyNorm3FWTensorReturn = ttnn::Tensor;

}  // namespace ttml::metal::ops::polynorm3_fw::device
