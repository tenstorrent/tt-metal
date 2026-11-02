// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::add_rmsnorm_fw_rows::device {

struct Params {
    uint32_t num_slices = 1;   // S: cores per tile-row
    uint32_t slice_tiles = 1;  // column tiles per core (the last slice may be shorter)
    float epsilon = 1e-6F;
    bool delta_transposed = false;  // iteration 37: delta arrives as [B, N, C, S] (the transposed w2 output)
};

struct Inputs {
    ttnn::Tensor residual;  // [B, N, S, C] bf16 TILE
    ttnn::Tensor delta;     // [B, N, S, C] bf16 TILE
    ttnn::Tensor gamma;     // [1, 1, 1, C] bf16 TILE
};

using operation_attributes_t = Params;
using tensor_args_t = Inputs;
// {sum [B, N, S, C] bf16, out [B, N, S, C] bf16, rms [B, N, S, 1] bf16 (column 0 of each tile)}
using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
using tensor_return_value_t = std::vector<ttnn::Tensor>;

}  // namespace ttml::metal::ops::add_rmsnorm_fw_rows::device
