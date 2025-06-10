// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::cross_entropy_bw::device {

struct operation_attributes_t {
    const float scaler{1.0F};
};

struct tensor_args_t {
    const ttnn::Tensor& input;
    const ttnn::Tensor& target;

    std::optional<ttnn::Tensor> preallocated_output;
};

using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = ttnn::TensorSpec;

}  // namespace ttml::metal::ops::cross_entropy_bw::device
