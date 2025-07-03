// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::profiler_no_op::device {

struct operation_attributes_t {};

struct tensor_args_t {
    const ttnn::Tensor& input;

    std::optional<ttnn::Tensor> preallocated_output;
};

using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = ttnn::TensorSpec;

}  // namespace ttml::metal::ops::profiler_no_op::device
