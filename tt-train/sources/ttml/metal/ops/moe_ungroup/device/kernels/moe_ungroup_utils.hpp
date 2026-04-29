// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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

}  // namespace ttml::metal::moe_ungroup
