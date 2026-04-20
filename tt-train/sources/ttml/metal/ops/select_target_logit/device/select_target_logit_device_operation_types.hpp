// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::select_target_logit::device {

struct operation_attributes_t {
    uint32_t first_v{0U};
    uint32_t last_v{std::numeric_limits<uint32_t>::max()};
};

struct tensor_args_t {
    const ttnn::Tensor& logit;   // [N, 1, H, local_V] TILE BFLOAT16  (local_V = last_v - first_v)
    const ttnn::Tensor& target;  // [N, H]              ROW_MAJOR UINT32  (global indices)

    std::optional<ttnn::Tensor> preallocated_output;
};

// output: [N, 1, H, 1] TILE BFLOAT16 or FLOAT32 (matches preallocated_output dtype, or logit dtype)
// output[n, 0, h, 0] = logit[n, 0, h, target[n, h] - first_v]   if target[n, h] in [first_v, last_v)
//                     = 0.0                                        otherwise
using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = ttnn::TensorSpec;

}  // namespace ttml::metal::ops::select_target_logit::device
