// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::silu_bw::device {

// Attributes for the backward operation (add more if needed)
struct operation_attributes_t {
    static constexpr auto attribute_names = std::make_tuple();
    auto attribute_values() const {
        return std::make_tuple();
    }
};

// Tensors required for backward
struct tensor_args_t {
    ttnn::Tensor input;
    ttnn::Tensor dL_dout;
    std::optional<ttnn::Tensor> preallocated_da = std::nullopt;

    static constexpr auto attribute_names = std::forward_as_tuple("input", "dL_dout", "preallocated_da");
    auto attribute_values() const {
        return std::forward_as_tuple(input, dL_dout, preallocated_da);
    }
};

// Output tensor specs and tensors
using spec_return_value_t = std::vector<ttnn::TensorSpec>;
using tensor_return_value_t = ttnn::Tensor;

}  // namespace ttml::metal::ops::silu_bw::device
