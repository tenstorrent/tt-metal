
// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::cross_entropy_fw::device {

struct operation_attributes_t {};  // add reduction attribute to reduce loss to (1, 1, 1, 1)

struct tensor_args_t {
    const ttnn::Tensor& input;
    const ttnn::Tensor& target;

    std::optional<ttnn::Tensor> preallocated_output;  // loss
};

using tensor_return_value_t = ttnn::Tensor;  // return loss: tensor with shape (B, 1, S, 1)
// todo: change loss shape to (1, 1, 1, 1) - we will reduce it later

using spec_return_value_t = ttnn::TensorSpec;

}  // namespace ttml::metal::ops::cross_entropy_fw::device
