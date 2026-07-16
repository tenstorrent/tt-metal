// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::profiler_no_op::device {

struct ProfilerNoOpParams {
    std::string identifier = "profiler_no_op";

    static constexpr auto attribute_names = std::forward_as_tuple("identifier");
    auto attribute_values() const {
        return std::forward_as_tuple(identifier);
    }
};

struct ProfilerNoOpInputs {
    const ttnn::Tensor& input;

    std::optional<ttnn::Tensor> preallocated_output;

    static constexpr auto attribute_names = std::forward_as_tuple("input_dtype", "input_logical_shape");
    auto attribute_values() const {
        return std::make_tuple(input.dtype(), std::cref(input.logical_shape()));
    }
};

using operation_attributes_t = ProfilerNoOpParams;
using tensor_args_t = ProfilerNoOpInputs;

using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = ttnn::TensorSpec;

}  // namespace ttml::metal::ops::profiler_no_op::device
