// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::rmsnorm_bw_rows::device {

struct Params {
    uint32_t num_slices = 1;   // S: cores per tile-row
    uint32_t slice_tiles = 1;  // column tiles per core (the last slice may be shorter)
    bool transposed_grad = false;  // iteration 37: also emit dL_da^T [B, N, C, S] (the transposed w2 output's gradient)
};

struct Inputs {
    ttnn::Tensor input;                         // [B, N, S, C] bf16 TILE
    ttnn::Tensor gamma;                         // [1, 1, 1, C] bf16 TILE
    ttnn::Tensor rms;                           // [B, N, S, 1] bf16 TILE (column 0 of each tile)
    ttnn::Tensor dL_dout;                       // [B, N, S, C] bf16 TILE
    std::optional<ttnn::Tensor> residual_grad;  // [B, N, S, C] bf16 TILE, added to dL_da
};

using operation_attributes_t = Params;
using tensor_args_t = Inputs;
// {dL_da [B, N, S, C]} or {dL_da, dL_da^T [B, N, C, S]} with transposed_grad
using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
using tensor_return_value_t = std::vector<ttnn::Tensor>;

}  // namespace ttml::metal::ops::rmsnorm_bw_rows::device
