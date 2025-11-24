// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::softmax::device {

struct operation_attributes_t {
    const int32_t dim{3U};  // Use last dimension by default
};

struct tensor_args_t {
    const ttnn::Tensor& input;

    std::optional<ttnn::Tensor> preallocated_output;
};

using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = ttnn::TensorSpec;

}  // namespace ttml::metal::ops::softmax::device
