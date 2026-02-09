// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::swiglu_fw::device {

struct operation_attributes_t {
    static constexpr auto attribute_names = std::make_tuple();
    auto attribute_values() const {
        return std::make_tuple();
    }
};

struct tensor_args_t {
    ttnn::Tensor input;
    ttnn::Tensor w1;
    ttnn::Tensor w2;
    ttnn::Tensor w3;
    std::optional<ttnn::Tensor> preallocated_swiglu = std::nullopt;

    static constexpr auto attribute_names = std::forward_as_tuple("input", "w1", "w2", "w3", "preallocated_swiglu");
    auto attribute_values() const {
        return std::forward_as_tuple(input, w1, w2, w3, preallocated_swiglu);
    }
};

// Output tensor specs and tensors
using spec_return_value_t = std::vector<ttnn::TensorSpec>;

using tensor_return_value_t = ttnn::Tensor;

}  // namespace ttml::metal::ops::swiglu_fw::device
