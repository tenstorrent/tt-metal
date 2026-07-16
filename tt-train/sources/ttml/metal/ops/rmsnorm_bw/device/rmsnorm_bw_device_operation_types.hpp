// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::rmsnorm_bw::device {

// Attributes for the backward operation (add more if needed)
struct RMSNormBackwardParams {
    float epsilon = 1e-6F;

    static constexpr auto attribute_names = std::forward_as_tuple("epsilon");
    auto attribute_values() const {
        return std::forward_as_tuple(epsilon);
    }
};

// Tensors required for backward
struct RMSNormBackwardInputs {
    ttnn::Tensor input;
    ttnn::Tensor gamma;
    ttnn::Tensor rms;
    ttnn::Tensor dL_dout;
    std::optional<ttnn::Tensor> preallocated_da = std::nullopt;
    std::optional<ttnn::Tensor> preallocated_dgamma_components = std::nullopt;

    static constexpr auto attribute_names = std::forward_as_tuple("input_dtype", "input_logical_shape", "input");
    auto attribute_values() const {
        return std::make_tuple(input.dtype(), std::cref(input.logical_shape()), std::cref(input));
    }
};

using operation_attributes_t = RMSNormBackwardParams;
using tensor_args_t = RMSNormBackwardInputs;

// Output tensor specs and tensors
using spec_return_value_t = std::vector<ttnn::TensorSpec>;
using tensor_return_value_t = std::vector<ttnn::Tensor>;

}  // namespace ttml::metal::ops::rmsnorm_bw::device
