// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::softmax::device {

struct SoftmaxParams {
    const int32_t dim{3U};  // Use last dimension by default
};

struct SoftmaxInputs {
    const ttnn::Tensor& input;

    std::optional<ttnn::Tensor> preallocated_output;
};

using operation_attributes_t = SoftmaxParams;
using tensor_args_t = SoftmaxInputs;

using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = ttnn::TensorSpec;

}  // namespace ttml::metal::ops::softmax::device
