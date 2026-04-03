// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn/tensor/tensor.hpp>

#include "metal/common/const_utils.hpp"

namespace ttml::metal::ops::gram_polynomial::device {

struct operation_attributes_t {
    float b = 0.0f;
    float c = 1.0f;
    ttml::metal::OutputMode output_mode = ttml::metal::OutputMode::UpperTriangle;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
};

struct tensor_args_t {
    ttnn::Tensor input_tensor;
    std::optional<ttnn::Tensor> preallocated_output = std::nullopt;
};

using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = ttnn::TensorSpec;

}  // namespace ttml::metal::ops::gram_polynomial::device
