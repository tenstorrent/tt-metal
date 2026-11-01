// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::vp_ce_row_stats::device {

struct Params {
    uint32_t mode{0U};  // 0: row max, 1: row sum of exp(x - max_col)
};

struct Inputs {
    const ttnn::Tensor& logits;             // [B, 1, S, V] TILE bf16
    std::optional<ttnn::Tensor> max_col{};  // [B, 1, S, 1] TILE bf16 / fp32 (mode 1)
};

using operation_attributes_t = Params;
using tensor_args_t = Inputs;
using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = tt::tt_metal::TensorSpec;

}  // namespace ttml::metal::ops::vp_ce_row_stats::device
