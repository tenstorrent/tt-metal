// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::vp_cross_entropy_bw::device {

struct Params {
    uint32_t first_v{0U};
    uint32_t local_V{0U};
    std::optional<uint32_t> cluster_axis{};
    float scale{1.0F};
};

struct Inputs {
    const ttnn::Tensor& logits;          // [B, 1, S, local_V] TILE bf16
    const ttnn::Tensor& log_normalizer;  // [B, 1, S, 1] TILE fp32
    const ttnn::Tensor& target;          // [B, S] ROW_MAJOR uint32 (global indices)
    const ttnn::Tensor& grad;            // one TILE [1, 1, 32, 32] holding the loss's upstream gradient g everywhere
};

using operation_attributes_t = Params;
using tensor_args_t = Inputs;
using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = tt::tt_metal::TensorSpec;

}  // namespace ttml::metal::ops::vp_cross_entropy_bw::device
