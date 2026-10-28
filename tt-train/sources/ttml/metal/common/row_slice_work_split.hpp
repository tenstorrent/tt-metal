// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttml::metal {

// Work decomposition for row-wise normalisation ops that spread one tile-row over several cores.
//
// A work item is (tile-row r, slice s); slice s covers tiles [s*slice_tiles, min(Wt, (s+1)*slice_tiles)).
// Phase A emits one partial-reduction tile per item, phase B folds the row's `num_slices` partials into the
// row statistic and applies it to the slice. Both phases run over rows * num_slices items, so a [1, 1, T, C]
// tensor is not limited to T/32 cores.
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

}  // namespace ttml::metal
