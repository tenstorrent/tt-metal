// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>

namespace ttml::metal::moe_ungroup {

// Per-core contiguous slice of `total` items across `num_cores` cores.
// Returns this core's [start, start+count). Cores whose start lands past
// `total` get an empty slice (count = 0). Used both for the writer's
// pre-zero work distribution (total = D*B*S rows of `ungrouped`) and for
// the per-expert worker split (total = expert's tile-row count).
struct CoreSlice {
    uint32_t start;
    uint32_t count;
};

inline CoreSlice slice_for_core(uint32_t total, uint32_t num_cores, uint32_t core_idx) {
    uint32_t per_core = (total + num_cores - 1U) / num_cores;
    uint32_t start = core_idx * per_core;
    uint32_t end = start + per_core;
    end = std::min(end, total);
    start = std::min(start, total);
    return {start, end - start};
}

// Per-(expert × core) work bounds derived from the offsets table. TILE_HEIGHT
// is the tile-row stride: offsets are stored in source rows, both kernels work
// in 32-row tile-rows. Identical in moe_ungroup_reader and moe_ungroup_rmw_writer.
struct ExpertCoreSlice {
    uint32_t expert_start_tr;     // absolute tile-row of expert e in grouped layout
    uint32_t my_start_tr_global;  // absolute tile-row where this core starts
    uint32_t my_count;            // tile-rows owned by this core
};

template <typename Uint32Ptr>
inline ExpertCoreSlice expert_slice_for_core(
    Uint32Ptr offsets, uint32_t e, uint32_t tile_h, uint32_t num_cores, uint32_t core_idx) {
    uint32_t expert_start_tr = offsets[e] / tile_h;
    uint32_t expert_total_tr = (offsets[e + 1U] - offsets[e]) / tile_h;
    auto s = slice_for_core(expert_total_tr, num_cores, core_idx);
    return {expert_start_tr, expert_start_tr + s.start, s.count};
}

}  // namespace ttml::metal::moe_ungroup
