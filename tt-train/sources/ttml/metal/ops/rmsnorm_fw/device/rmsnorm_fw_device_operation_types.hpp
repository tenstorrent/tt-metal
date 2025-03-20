// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::rmsnorm_fw::device {

struct operation_attributes_t {
    bool return_intermediates{false};
    float epsilon{1e-6F};
};

struct tensor_args_t {
    const ttnn::Tensor& input;
    const ttnn::Tensor& gamma;

    std::optional<ttnn::Tensor> preallocated_rms;
    std::optional<ttnn::Tensor> preallocated_output;
};

using tensor_return_value_t = std::vector<ttnn::Tensor>;

using spec_return_value_t = std::vector<ttnn::TensorSpec>;

}  // namespace ttml::metal::ops::rmsnorm_fw::device
