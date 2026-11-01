// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::transpose_split::device {

struct Params {
    uint32_t rows = 0;       // rows of the input that become dx's columns (multiple of 32)
    uint32_t tail_rows = 0;  // rows after them that become tail's columns (<= 32)
    float tail_scale = 1.0F;
    uint32_t dx_cols = 0;  // dx's width (>= rows, <= the input rows; 0 = rows): the extra input rows are transposed
                           // into dx's tail columns unscaled (the wide LoRA input buffer's gradient, iteration 24)
};

struct Inputs {
    ttnn::Tensor input;  // [1, 1, R, T] bf16 TILE
    // Fused low-rank update (iteration 25): dx += tail^T @ w with w [1, 1, tail_rows, rows] bf16 (the LoRA A), the
    // tail transposed and scaled as the tail output is; replaces a separate lowrank_update pass over dx.
    std::optional<ttnn::Tensor> w = std::nullopt;
};

using operation_attributes_t = Params;
using tensor_args_t = Inputs;
using spec_return_value_t = std::tuple<tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec>;
using tensor_return_value_t = std::tuple<ttnn::Tensor, ttnn::Tensor>;

}  // namespace ttml::metal::ops::transpose_split::device
