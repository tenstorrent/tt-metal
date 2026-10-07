// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <cstdint>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::rmsnorm_bw::device {

// RMSNorm backward runs as two device operations so that a single row can be spread over many cores:
//
//   partial:  P_s = sum_{c in slice s} a * gamma * dL_dout           (one fp32 tile per (tile-row, slice))
//   apply:    P   = sum_s P_s (row-reduced), scale = P / rms
//             dL_da = (gamma * dL_dout - a * P / (C * rms^2)) / rms
//             dL_dgamma_components = a * dL_dout / rms              (optional; reduced over rows on host)
//
// A work item is (tile-row r, slice s); slice s covers tiles [s*slice_tiles, min(Wt, (s+1)*slice_tiles)).

struct WorkSplit {
    uint32_t num_slices = 1;   // S
    uint32_t slice_tiles = 1;  // tiles per slice (last slice may be shorter)
};

// Picks the number of slices per row so that rows * num_slices work items keep the core grid busy
// without making the per-item partial-sum fan-in (num_slices fp32 tiles) or the fixed per-item cost dominate.
inline WorkSplit choose_work_split(uint32_t rows, uint32_t Wt, uint32_t num_cores) {
    constexpr uint32_t kMaxSlices = 32U;
    constexpr double kPartialLoadCost = 0.25;  // relative to one tile of main work
    constexpr double kFixedItemCost = 3.0;
    WorkSplit best;
    double best_cost = -1.0;
    const uint32_t max_slices = Wt < kMaxSlices ? Wt : kMaxSlices;
    for (uint32_t S = 1; S <= max_slices; ++S) {
        const uint32_t slice_tiles = (Wt + S - 1U) / S;
        const uint32_t S_eff = (Wt + slice_tiles - 1U) / slice_tiles;  // no empty slices
        const uint64_t items = static_cast<uint64_t>(rows) * S_eff;
        const double waves = static_cast<double>((items + num_cores - 1U) / num_cores);
        const double cost = waves * (slice_tiles + kPartialLoadCost * S_eff + kFixedItemCost);
        if (best_cost < 0.0 || cost < best_cost) {
            best_cost = cost;
            best = WorkSplit{.num_slices = S_eff, .slice_tiles = slice_tiles};
        }
    }
    return best;
}

// ---------------------------------------------------------------------------------------------------------------
// Phase A: partial sums
// ---------------------------------------------------------------------------------------------------------------
namespace partial {

struct Params {
    uint32_t num_slices = 1;
    uint32_t slice_tiles = 1;
};

struct Inputs {
    ttnn::Tensor input;    // [B, N, S, C] bf16 TILE
    ttnn::Tensor gamma;    // [1, 1, 1, C] bf16 TILE
    ttnn::Tensor dL_dout;  // [B, N, S, C] bf16 TILE
};

using operation_attributes_t = Params;
using tensor_args_t = Inputs;
using spec_return_value_t = tt::tt_metal::TensorSpec;
using tensor_return_value_t = ttnn::Tensor;  // [1, 1, rows*32, num_slices*32] fp32 TILE

}  // namespace partial

// ---------------------------------------------------------------------------------------------------------------
// Phase B: gradients
// ---------------------------------------------------------------------------------------------------------------
struct RMSNormBackwardParams {
    float epsilon = 1e-6F;
    bool compute_dgamma = true;
    uint32_t num_slices = 1;
    uint32_t slice_tiles = 1;
};

struct RMSNormBackwardInputs {
    ttnn::Tensor input;
    ttnn::Tensor gamma;
    ttnn::Tensor rms;  // [B, N, S, 1] bf16 TILE (column 0 of each tile)
    ttnn::Tensor dL_dout;
    ttnn::Tensor partials;  // output of phase A
};

using operation_attributes_t = RMSNormBackwardParams;
using tensor_args_t = RMSNormBackwardInputs;

// {dL_da, dL_dgamma_components (only when compute_dgamma)}
using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
using tensor_return_value_t = std::vector<ttnn::Tensor>;

}  // namespace ttml::metal::ops::rmsnorm_bw::device
