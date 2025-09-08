// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::optimizers::sgd_fused::device {

struct operation_attributes_t {
    float lr{};
};

struct tensor_args_t {
    const ttnn::Tensor& param_in;
    const ttnn::Tensor& grad;
    std::optional<ttnn::Tensor> param_out = std::nullopt;
};

using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = ttnn::TensorSpec;

}  // namespace ttml::metal::optimizers::sgd_fused::device
