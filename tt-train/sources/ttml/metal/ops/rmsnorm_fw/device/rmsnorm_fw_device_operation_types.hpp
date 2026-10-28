// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "metal/common/row_slice_work_split.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::rmsnorm_fw::device {

// RMSNorm forward runs as two device operations so that a single row can be spread over many cores:
//
//   partial:  Q_s = sum_{c in slice s} x^2                      (one fp32 tile per (tile-row, slice))
//   apply:    Q   = rowsum(sum_s Q_s)
//             rms = sqrt(Q / C + eps)                           (optionally returned as [B, N, S, 1])
//             y   = x * gamma / rms                             (for the item's slice)
//
// A work item is (tile-row r, slice s); slice s covers tiles [s*slice_tiles, min(Wt, (s+1)*slice_tiles)).
// WorkSplit / choose_work_split live in metal/common/row_slice_work_split.hpp (shared with rmsnorm_bw).
using ttml::metal::choose_work_split;
using ttml::metal::WorkSplit;

// ---------------------------------------------------------------------------------------------------------------
// Phase A: partial sums of squares
// ---------------------------------------------------------------------------------------------------------------
namespace partial {

struct Params {
    uint32_t num_slices = 1;
    uint32_t slice_tiles = 1;
};

struct Inputs {
    ttnn::Tensor input;  // [B, N, S, C] bf16 TILE
};

using operation_attributes_t = Params;
using tensor_args_t = Inputs;
using spec_return_value_t = tt::tt_metal::TensorSpec;
using tensor_return_value_t = ttnn::Tensor;  // [1, 1, rows*32, num_slices*32] fp32 TILE

}  // namespace partial

// ---------------------------------------------------------------------------------------------------------------
// Phase B: rms + normalised output
// ---------------------------------------------------------------------------------------------------------------
struct RMSNormForwardParams {
    bool return_intermediates = false;
    float epsilon = 1e-6F;
    uint32_t num_slices = 1;
    uint32_t slice_tiles = 1;
};

struct RMSNormForwardInputs {
    ttnn::Tensor input;     // [B, N, S, C] bf16 TILE
    ttnn::Tensor gamma;     // [1, 1, 1, C] bf16 TILE
    ttnn::Tensor partials;  // output of phase A
};

using operation_attributes_t = RMSNormForwardParams;
using tensor_args_t = RMSNormForwardInputs;

// {output, rms (only when return_intermediates)}
using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
using tensor_return_value_t = std::vector<ttnn::Tensor>;

}  // namespace ttml::metal::ops::rmsnorm_fw::device
