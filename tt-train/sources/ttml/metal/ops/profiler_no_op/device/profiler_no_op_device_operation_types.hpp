// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::profiler_no_op::device {

struct operation_attributes_t {
    std::string identifier = "profiler_no_op";

    static constexpr auto attribute_names = std::forward_as_tuple("identifier");
    auto attribute_values() const {
        return std::forward_as_tuple(identifier);
    }
};

struct tensor_args_t {
    const ttnn::Tensor& input;

    std::optional<ttnn::Tensor> preallocated_output;

    static constexpr auto attribute_names = std::forward_as_tuple("input", "preallocated_output");
    auto attribute_values() const {
        return std::forward_as_tuple(input, preallocated_output);
    }
};

using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = ttnn::TensorSpec;

}  // namespace ttml::metal::ops::profiler_no_op::device
