// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::lowrank_update::device {

struct Params {
    float scale = 1.0F;
    bool transpose_b = false;
};

struct Inputs {
    ttnn::Tensor acc;  // [B, C, M, N] bf16 TILE
    ttnn::Tensor a;    // [B, C, M, K] bf16 TILE, K <= 32
    ttnn::Tensor w;    // [1, 1, N, K] (transpose_b) or [1, 1, K, N] bf16 TILE
};

using operation_attributes_t = Params;
using tensor_args_t = Inputs;
using spec_return_value_t = tt::tt_metal::TensorSpec;
using tensor_return_value_t = ttnn::Tensor;

}  // namespace ttml::metal::ops::lowrank_update::device
