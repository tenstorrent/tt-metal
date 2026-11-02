// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::skinny_matmul::device {

// Phase A work item = (row group g, K-slice s): rows [g*rows_per_item, min(rows, (g+1)*rows_per_item)) of x
// against W's K-slice [s*slice_tiles, min(Kt, (s+1)*slice_tiles)), the W slice read once per item into L1.
// It emits one fp32 partial tile per (row, n) of the item: partials[r, s*Nt + n] (tile units).
// Phase B work item = (row r, output column tile n): out[r, n] = sum_s partials[r, s*Nt + n].
struct SkinnySplit {
    uint32_t rows_per_item = 4;
    uint32_t num_slices = 1;
    uint32_t slice_tiles = 1;
};

// Row groups of `rows_per_item` tile-rows; the K split keeps groups * slices work items filling the core grid
// without making the per-item W load (slice_tiles * Nt tiles) or the phase-B fan-in (num_slices tiles) dominate.
inline SkinnySplit choose_skinny_split(uint32_t rows, uint32_t Kt, uint32_t num_cores) {
    constexpr uint32_t kRowsPerItem = 4U;
    constexpr uint32_t kMaxSlices = 32U;
    constexpr double kWLoadCost = 1.0;      // one W tile read per slice tile, amortised over rows_per_item rows
    constexpr double kPartialCost = 0.5;    // per slice: one fp32 partial write (A) and read (B) per row
    constexpr double kFixedItemCost = 4.0;  // dispatch of an item
    SkinnySplit best{.rows_per_item = kRowsPerItem};
    const uint32_t groups = (rows + kRowsPerItem - 1U) / kRowsPerItem;
    const double rows_per_group = static_cast<double>(std::min(rows, kRowsPerItem));
    double best_cost = -1.0;
    const uint32_t max_slices = Kt < kMaxSlices ? Kt : kMaxSlices;
    for (uint32_t S = 1; S <= max_slices; ++S) {
        const uint32_t slice_tiles = (Kt + S - 1U) / S;
        const uint32_t S_eff = (Kt + slice_tiles - 1U) / slice_tiles;
        const uint64_t items = static_cast<uint64_t>(groups) * S_eff;
        const double waves = static_cast<double>((items + num_cores - 1U) / num_cores);
        const double cost =
            waves * (rows_per_group * (slice_tiles + kPartialCost) + kWLoadCost * slice_tiles + kFixedItemCost);
        if (best_cost < 0.0 || cost < best_cost) {
            best_cost = cost;
            best.num_slices = S_eff;
            best.slice_tiles = slice_tiles;
        }
    }
    return best;
}

struct Params {
    bool transpose_b = false;
    uint32_t rows_per_item = 4;
    uint32_t num_slices = 1;
    uint32_t slice_tiles = 1;
    // REDUCE with a tail target: the output tiles are also written into `tail_out` at tile column tail_col
    // (and zero tiles fill columns tail_col + Nt .. tail_col + tail_tiles): the fused LoRA h into the wide input
    // buffer's tail (iteration 24).
    uint32_t tail_col = 0;
    uint32_t tail_tiles = 0;
    // iteration 40: x stored transposed ([1, 1, K', M], its tile (m, k) at page k * Mt + m: the transposed wide
    // LoRA buffer), and the output transposed ([1, 1, N, M]; with a tail target the tiles go to the tile ROWS
    // tail_col .. of it, the zero tiles to the rows up to tail_col + tail_tiles)
    bool x_transposed = false;
    bool transpose_out = false;
};

struct PartialInputs {
    ttnn::Tensor x;  // [B, C, M, K] bf16 TILE
    ttnn::Tensor w;  // [1, 1, N, K] (transpose_b) or [1, 1, K, N] bf16 TILE
};

struct ReduceInputs {
    ttnn::Tensor x;  // shape / memory config of the output
    ttnn::Tensor w;
    ttnn::Tensor partials;  // phase-A output [1, 1, rows*32, num_slices*Nt*32] fp32
    std::optional<ttnn::Tensor> tail_out = std::nullopt;  // written in place at tile column tail_col (see Params)
};

namespace partial {
using operation_attributes_t = Params;
using tensor_args_t = PartialInputs;
using spec_return_value_t = tt::tt_metal::TensorSpec;
using tensor_return_value_t = ttnn::Tensor;
}  // namespace partial

namespace reduce {
using operation_attributes_t = Params;
using tensor_args_t = ReduceInputs;
using spec_return_value_t = tt::tt_metal::TensorSpec;
using tensor_return_value_t = ttnn::Tensor;
}  // namespace reduce

}  // namespace ttml::metal::ops::skinny_matmul::device
