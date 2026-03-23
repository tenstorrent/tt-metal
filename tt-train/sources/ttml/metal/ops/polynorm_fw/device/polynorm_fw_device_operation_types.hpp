// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::polynorm3_fw::device {

struct operation_attributes_t {
    float epsilon{1e-5F};
};

struct tensor_args_t {
    const ttnn::Tensor& input;
    const ttnn::Tensor& weight;
    const ttnn::Tensor& bias;
    std::optional<ttnn::Tensor> preallocated_output = std::nullopt;
};

using spec_return_value_t = std::vector<ttnn::TensorSpec>;
using tensor_return_value_t = ttnn::Tensor;

}  // namespace ttml::metal::ops::polynorm3_fw::device
