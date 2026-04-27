// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/work_split.hpp>

namespace tt::tt_metal {

const CachedWorkSplit& cached_split_work_to_cores(CoreCoord grid_size, uint32_t units_to_divide, bool row_wise) {
    // Single-entry thread-local cache keyed on (grid, units, row_wise).
    // Sub-device callers must pass their sub-device's grid here — this helper
    // does not introspect the device.
    struct CacheEntry {
        uint32_t grid_x = 0, grid_y = 0, units = 0;
        bool row_wise = false;
        bool populated = false;
        CachedWorkSplit result;

        bool matches(uint32_t gx, uint32_t gy, uint32_t u, bool rw) const {
            return populated && grid_x == gx && grid_y == gy && units == u && row_wise == rw;
        }
    };
    static thread_local CacheEntry cache;

    if (!cache.matches(grid_size.x, grid_size.y, units_to_divide, row_wise)) {
        auto [nc, ac, cg1, cg2, upcg1, upcg2] = split_work_to_cores(grid_size, units_to_divide, row_wise);
        cache.grid_x = grid_size.x;
        cache.grid_y = grid_size.y;
        cache.units = units_to_divide;
        cache.row_wise = row_wise;
        cache.populated = true;
        cache.result.num_cores = nc;
        cache.result.all_cores = std::move(ac);
        cache.result.core_group_1 = std::move(cg1);
        cache.result.core_group_2 = std::move(cg2);
        cache.result.units_per_core_group_1 = upcg1;
        cache.result.units_per_core_group_2 = upcg2;
        cache.result.cores = grid_to_cores(nc, grid_size.x, grid_size.y, row_wise);
    }

    return cache.result;
}

}  // namespace tt::tt_metal
