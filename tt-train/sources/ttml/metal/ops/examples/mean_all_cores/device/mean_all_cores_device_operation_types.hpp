// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::examples::mean_all_cores::device {

struct operation_attributes_t {};

struct tensor_args_t {
    const ttnn::Tensor& input;
    std::optional<ttnn::Tensor> preallocated_output;
};

using tensor_return_value_t = ttnn::Tensor;  // return mean: tensor with shape (1, 1, 1, 1)
using spec_return_value_t = ttnn::TensorSpec;

}  // namespace ttml::metal::ops::examples::mean_all_cores::device

