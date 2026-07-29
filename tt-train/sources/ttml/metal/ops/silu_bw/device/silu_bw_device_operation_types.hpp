// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::silu_bw::device {

// Attributes for the backward operation (add more if needed)
struct SiLUBackwardParams {};

// Tensors required for backward
struct SiLUBackwardInputs {
    ttnn::Tensor input;
    ttnn::Tensor dL_dout;
    std::optional<ttnn::Tensor> preallocated_da = std::nullopt;
};

using operation_attributes_t = SiLUBackwardParams;
using tensor_args_t = SiLUBackwardInputs;

// Output tensor specs and tensors
using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
using tensor_return_value_t = ttnn::Tensor;

}  // namespace ttml::metal::ops::silu_bw::device
