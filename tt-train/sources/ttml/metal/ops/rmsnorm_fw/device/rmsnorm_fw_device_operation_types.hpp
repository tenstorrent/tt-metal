// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::rmsnorm_fw::device {

struct operation_attributes_t {
    bool return_intermediates{false};
    float epsilon{1e-6F};

    static constexpr auto attribute_names = std::forward_as_tuple("return_intermediates", "epsilon");
    auto attribute_values() const {
        return std::forward_as_tuple(return_intermediates, epsilon);
    }
};

struct tensor_args_t {
    const ttnn::Tensor& input;
    const ttnn::Tensor& gamma;

    std::optional<ttnn::Tensor> preallocated_rms;
    std::optional<ttnn::Tensor> preallocated_output;

    static constexpr auto attribute_names =
        std::forward_as_tuple("input", "gamma", "preallocated_rms", "preallocated_output");
    auto attribute_values() const {
        return std::forward_as_tuple(input, gamma, preallocated_rms, preallocated_output);
    }
};

using tensor_return_value_t = std::vector<ttnn::Tensor>;

using spec_return_value_t = std::vector<ttnn::TensorSpec>;

}  // namespace ttml::metal::ops::rmsnorm_fw::device
