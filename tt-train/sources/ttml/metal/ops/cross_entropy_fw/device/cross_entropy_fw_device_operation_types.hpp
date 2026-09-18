// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::cross_entropy_fw::device {

struct CrossEntropyForwardParams {};

struct CrossEntropyForwardInputs {
    const ttnn::Tensor& input;
    const ttnn::Tensor& target;

    std::optional<ttnn::Tensor> preallocated_output;
};

using operation_attributes_t = CrossEntropyForwardParams;
using tensor_args_t = CrossEntropyForwardInputs;

using tensor_return_value_t = ttnn::Tensor;  // return loss: tensor with shape (N, 1, H, 1)
using spec_return_value_t = tt::tt_metal::TensorSpec;

}  // namespace ttml::metal::ops::cross_entropy_fw::device
