// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::rmsnorm_fw::device {

struct RMSNormForwardParams {
    bool return_intermediates{false};
    float epsilon{1e-6F};

    static constexpr auto attribute_names = std::forward_as_tuple("return_intermediates", "epsilon");
    auto attribute_values() const {
        return std::forward_as_tuple(return_intermediates, epsilon);
    }
};

struct RMSNormForwardInputs {
    const ttnn::Tensor& input;
    const ttnn::Tensor& gamma;

    std::optional<ttnn::Tensor> preallocated_rms;
    std::optional<ttnn::Tensor> preallocated_output;

    static constexpr auto attribute_names = std::forward_as_tuple("input_dtype", "input_logical_shape", "input");
    auto attribute_values() const {
        return std::make_tuple(input.dtype(), std::cref(input.logical_shape()), std::cref(input));
    }
};

using operation_attributes_t = RMSNormForwardParams;
using tensor_args_t = RMSNormForwardInputs;

using tensor_return_value_t = std::vector<ttnn::Tensor>;

using spec_return_value_t = std::vector<ttnn::TensorSpec>;

}  // namespace ttml::metal::ops::rmsnorm_fw::device
