// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::cross_entropy_fw::device {

struct operation_attributes_t {};

struct tensor_args_t {
    const ttnn::Tensor& input;
    const ttnn::Tensor& target;

    std::optional<ttnn::Tensor> preallocated_output;
};

using tensor_return_value_t = ttnn::Tensor;  // return loss: tensor with shape (N, 1, H, 1)
using spec_return_value_t = ttnn::TensorSpec;

}  // namespace ttml::metal::ops::cross_entropy_fw::device
