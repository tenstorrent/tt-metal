// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::subtract_at_target::device {

struct operation_attributes_t {
    uint32_t first_v{0U};
    uint32_t last_v{std::numeric_limits<uint32_t>::max()};
    float subtract_value{1.0F};
};

struct tensor_args_t {
    const ttnn::Tensor& input;   // [N, 1, H, local_V] TILE BFLOAT16
    const ttnn::Tensor& target;  // [N, H]              ROW_MAJOR UINT32  (global indices)

    std::optional<ttnn::Tensor> preallocated_output;
};

// output: same shape as input [N, 1, H, local_V] TILE BFLOAT16
// output[n, 0, h, c] = input[n, 0, h, c] - subtract_value   if c + first_v == target[n, h] && target[n, h] in [first_v,
// last_v)
//                     = input[n, 0, h, c]                     otherwise
using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = ttnn::TensorSpec;

}  // namespace ttml::metal::ops::subtract_at_target::device
