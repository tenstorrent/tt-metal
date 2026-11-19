// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Experimental and subject to change: this header carries no API-stability guarantee.
namespace tt::tt_metal::experimental {

// How split_work_to_cores divides units of work over cores, without building its CoreRangeSets.
// The units may not divide evenly by the core count, so the first cores of the walk take one extra unit each
// until the remainder is used up. For example, 100 units over 64 cores: the first 36 cores get 2 units, the other 28
// get 1. When the units divide evenly, all cores get the same amount and there are no trailing cores.
struct WorkSplitCounts {
    uint32_t num_cores = 0;          // cores that get any work: min(units, available cores)
    uint32_t num_leading_cores = 0;  // cores that take the extra unit (all cores when the split is even)
    uint32_t units_per_leading_core = 0;
    uint32_t units_per_trailing_core = 0;  // one less than a leading core; 0 when there are no trailing cores
};

// max_num_cores is the number of cores to split over: grid_size.x * grid_size.y, or core_grid.num_cores().
WorkSplitCounts split_work_to_cores_counts(uint32_t max_num_cores, uint32_t units_to_divide);

}  // namespace tt::tt_metal::experimental
