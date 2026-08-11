// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::cross_entropy_bw::device {

struct CrossEntropyBackwardParams {
    const float scaler{1.0F};
};

struct CrossEntropyBackwardInputs {
    const ttnn::Tensor& input;
    const ttnn::Tensor& target;

    std::optional<ttnn::Tensor> preallocated_output;
};

using operation_attributes_t = CrossEntropyBackwardParams;
using tensor_args_t = CrossEntropyBackwardInputs;

using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = tt::tt_metal::TensorSpec;

}  // namespace ttml::metal::ops::cross_entropy_bw::device
