// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::skinny_contract::device {

struct Params {
    float scale = 1.0F;
    bool transpose_out = false;
    bool big_transposed =
        false;  // iteration 38: big is stored [1, 1, N, T] (g^T, x^T), its tile (t, n) at page n * Kt + t
};

struct Inputs {
    ttnn::Tensor small_t;  // [1, 1, r, T] bf16 TILE (r <= 32: one tile row), the transposed small operand
    ttnn::Tensor big;      // [1, 1, T, N] bf16 TILE
};

using operation_attributes_t = Params;
using tensor_args_t = Inputs;
using spec_return_value_t = tt::tt_metal::TensorSpec;
using tensor_return_value_t = ttnn::Tensor;

}  // namespace ttml::metal::ops::skinny_contract::device
