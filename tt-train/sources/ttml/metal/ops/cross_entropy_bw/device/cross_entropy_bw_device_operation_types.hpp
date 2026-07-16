// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::cross_entropy_bw::device {

struct CrossEntropyBackwardParams {
    const float scaler{1.0F};

    static constexpr auto attribute_names = std::forward_as_tuple("scaler");
    auto attribute_values() const {
        return std::forward_as_tuple(scaler);
    }
};

struct CrossEntropyBackwardInputs {
    const ttnn::Tensor& input;
    const ttnn::Tensor& target;

    std::optional<ttnn::Tensor> preallocated_output;

    CrossEntropyBackwardInputs(
        const ttnn::Tensor& input,
        const ttnn::Tensor& target,
        std::optional<ttnn::Tensor> preallocated_output = std::nullopt) :
        input(input), target(target), preallocated_output(std::move(preallocated_output)) {
    }

    static constexpr auto attribute_names = std::forward_as_tuple("input", "target");
    auto attribute_values() const {
        return std::forward_as_tuple(input, target);
    }
};

using operation_attributes_t = CrossEntropyBackwardParams;
using tensor_args_t = CrossEntropyBackwardInputs;

using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = ttnn::TensorSpec;

}  // namespace ttml::metal::ops::cross_entropy_bw::device
