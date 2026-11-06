// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Experimental and subject to change: this header carries no API-stability guarantee.
namespace tt::tt_metal::experimental {

// The numbers split_work_to_cores returns, without building its CoreRangeSets: the first num_cores_group_1 cores of
// its walk get units_per_core_group_1 units each, and the remaining num_cores - num_cores_group_1 cores get
// units_per_core_group_2, which is 0 when the units divide evenly.
struct WorkSplitCounts {
    uint32_t num_cores = 0;
    uint32_t num_cores_group_1 = 0;
    uint32_t units_per_core_group_1 = 0;
    uint32_t units_per_core_group_2 = 0;
};

// max_num_cores is the number of cores to split over: grid_size.x * grid_size.y, or core_grid.num_cores().
WorkSplitCounts split_work_to_cores_counts(uint32_t max_num_cores, uint32_t units_to_divide);

}  // namespace tt::tt_metal::experimental
