// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::frobenius_normalize::device {

struct operation_attributes_t {
    float epsilon{1e-7F};
};

struct tensor_args_t {
    const ttnn::Tensor& input;
    std::optional<ttnn::Tensor> preallocated_output;
};

using tensor_return_value_t = std::vector<ttnn::Tensor>;

using spec_return_value_t = std::vector<ttnn::TensorSpec>;

}  // namespace ttml::metal::ops::frobenius_normalize::device
