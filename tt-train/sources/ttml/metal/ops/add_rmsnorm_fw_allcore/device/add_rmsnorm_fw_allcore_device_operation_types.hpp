// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::add_rmsnorm_fw_allcore::device {

// Work item = (tile-row r, slice s), slice s covering tiles [s*slice_tiles, min(Wt, (s+1)*slice_tiles)), the split
// chosen by rmsnorm_bw_allcore's choose_work_split.
//
//   phase A (partial): sum = residual + delta (written, bf16); P_s = sum_{c in slice s} sum^2  (one fp32 tile)
//   phase B (apply):   rms = sqrt(rowsum(sum_s P_s) / C + eps); out = sum * gamma / rms; rms written for slice 0

namespace partial {

struct Params {
    uint32_t num_slices = 1;
    uint32_t slice_tiles = 1;
};

struct Inputs {
    ttnn::Tensor residual;  // [B, N, S, C] bf16 TILE
    ttnn::Tensor delta;     // [B, N, S, C] bf16 TILE
};

using operation_attributes_t = Params;
using tensor_args_t = Inputs;
// {sum [B, N, S, C] bf16, partials [1, 1, rows*32, num_slices*32] fp32}
using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
using tensor_return_value_t = std::vector<ttnn::Tensor>;

}  // namespace partial

struct ApplyParams {
    float epsilon = 1e-6F;
    uint32_t num_slices = 1;
    uint32_t slice_tiles = 1;
};

struct ApplyInputs {
    ttnn::Tensor sum;       // phase-A output
    ttnn::Tensor gamma;     // [1, 1, 1, C] bf16 TILE
    ttnn::Tensor partials;  // phase-A output
};

using operation_attributes_t = ApplyParams;
using tensor_args_t = ApplyInputs;
// {out [B, N, S, C] bf16, rms [B, N, S, 1] bf16}
using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
using tensor_return_value_t = std::vector<ttnn::Tensor>;

}  // namespace ttml::metal::ops::add_rmsnorm_fw_allcore::device
