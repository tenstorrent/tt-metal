// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::silu_bw::device {

// Attributes for the backward operation (add more if needed)
struct SiLUBackwardParams {};

// Tensors required for backward
struct SiLUBackwardInputs {
    ttnn::Tensor input;
    ttnn::Tensor dL_dout;
    std::optional<ttnn::Tensor> preallocated_da = std::nullopt;

    static constexpr auto attribute_names = std::forward_as_tuple("input_dtype", "input_logical_shape");
    auto attribute_values() const {
        return std::make_tuple(input.dtype(), std::cref(input.logical_shape()));
    }
};

using operation_attributes_t = SiLUBackwardParams;
using tensor_args_t = SiLUBackwardInputs;

// Output tensor specs and tensors
using spec_return_value_t = std::vector<ttnn::TensorSpec>;
using tensor_return_value_t = ttnn::Tensor;

}  // namespace ttml::metal::ops::silu_bw::device
