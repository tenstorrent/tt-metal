// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::silu_bw::device {

// Attributes for the backward operation (add more if needed)
struct operation_attributes_t {};

// Tensors required for backward
struct tensor_args_t {
    ttnn::Tensor input;
    ttnn::Tensor dL_dout;
    std::optional<ttnn::Tensor> preallocated_da = std::nullopt;
};

// Output tensor specs and tensors
using spec_return_value_t = std::vector<ttnn::TensorSpec>;
using tensor_return_value_t = std::vector<ttnn::Tensor>;

}  // namespace ttml::metal::ops::silu_bw::device
