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
    // Upstream gradient folded into the per-row scale: one value ([1, 1, 1, 1]) or one per input row
    // ([N, C, H, 1]); see upstream_kind().
    std::optional<ttnn::Tensor> upstream;
};

// 0: no upstream gradient, 1: one value for every row, 2: one value per row.
inline uint32_t upstream_kind(const CrossEntropyBackwardInputs& args) {
    if (!args.upstream.has_value()) {
        return 0;
    }
    return args.upstream->logical_volume() == 1 ? 1 : 2;
}

using operation_attributes_t = CrossEntropyBackwardParams;
using tensor_args_t = CrossEntropyBackwardInputs;

using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = tt::tt_metal::TensorSpec;

}  // namespace ttml::metal::ops::cross_entropy_bw::device
