// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::polynorm_bw::device {

struct operation_attributes_t {
    float epsilon{1e-5F};
    float w0{0.0F};
    float w1{0.0F};
    float w2{0.0F};
};

struct tensor_args_t {
    const ttnn::Tensor& input;
    const ttnn::Tensor& dL_dout;
    std::optional<ttnn::Tensor> preallocated_dL_dx = std::nullopt;
};

using spec_return_value_t = std::vector<ttnn::TensorSpec>;
using tensor_return_value_t = ttnn::Tensor;

}  // namespace ttml::metal::ops::polynorm_bw::device
