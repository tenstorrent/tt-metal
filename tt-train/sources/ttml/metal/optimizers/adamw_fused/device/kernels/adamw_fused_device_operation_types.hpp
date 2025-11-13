// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn/tensor/tensor.hpp>

namespace ttml::metal::optimizers::adamw_fused::device {

struct operation_attributes_t {
    float lr{};
    float beta1{};
    float beta2{};
    float epsilon{};
    float weight_decay{};
    uint32_t step{};
};

struct tensor_args_t {
    const ttnn::Tensor& param;
    const ttnn::Tensor& grad;

    const ttnn::Tensor& first_moment;
    const ttnn::Tensor& second_moment;
};

using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = ttnn::TensorSpec;

}  // namespace ttml::metal::optimizers::adamw_fused::device
