// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::gather_logit::device {

struct operation_attributes_t {
    uint32_t first_v{0U};
    uint32_t last_v{std::numeric_limits<uint32_t>::max()};
};

struct tensor_args_t {
    const ttnn::Tensor& logit;   // [N, 1, H, V] TILE BFLOAT16
    const ttnn::Tensor& target;  // [N, H]       ROW_MAJOR UINT32

    std::optional<ttnn::Tensor> preallocated_output;
};

// output: [N, 1, H, 1] TILE BFLOAT16
// output[n, 0, h, 0] = logit[n, 0, h, target[n, h]]   if target[n, h] in [first_v, last_v)
//                     = 0.0                             otherwise
using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = ttnn::TensorSpec;

}  // namespace ttml::metal::ops::gather_logit::device
