// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::swiglu_fw::device {

struct operation_attributes_t {};

struct tensor_args_t {
    ttnn::Tensor input;
    ttnn::Tensor w1;
    ttnn::Tensor w2;
    ttnn::Tensor w3;
    std::optional<ttnn::Tensor> preallocated_swiglu = std::nullopt;
};

// Output tensor specs and tensors
using spec_return_value_t = std::vector<ttnn::TensorSpec>;

using tensor_return_value_t = ttnn::Tensor;

}  // namespace ttml::metal::ops::swiglu_fw::device
