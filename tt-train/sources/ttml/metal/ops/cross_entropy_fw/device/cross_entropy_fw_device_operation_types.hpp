// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::cross_entropy_fw::device {

struct operation_attributes_t {
    static constexpr auto attribute_names = std::make_tuple();
    auto attribute_values() const {
        return std::make_tuple();
    }
};

struct tensor_args_t {
    const ttnn::Tensor& input;
    const ttnn::Tensor& target;

    std::optional<ttnn::Tensor> preallocated_output;

    static constexpr auto attribute_names = std::forward_as_tuple("input", "target", "preallocated_output");
    auto attribute_values() const {
        return std::forward_as_tuple(input, target, preallocated_output);
    }
};

using tensor_return_value_t = ttnn::Tensor;  // return loss: tensor with shape (N, 1, H, 1)
using spec_return_value_t = ttnn::TensorSpec;

}  // namespace ttml::metal::ops::cross_entropy_fw::device
