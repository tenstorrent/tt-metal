// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/device.hpp>

#include <unordered_map>

namespace tt::tt_metal {

const CachedWorkSplit& cached_split_work_to_cores(IDevice* device, uint32_t units_to_divide, bool row_wise) {
    // Cache device → grid (never changes for a given device)
    static thread_local std::unordered_map<IDevice*, CoreCoord> grid_cache;

    // Single-entry cache for (grid, units, row_wise) → work split result
    struct CacheEntry {
        uint32_t grid_x = 0, grid_y = 0, units = 0;
        bool row_wise = false;
        CachedWorkSplit result;

        bool matches(uint32_t gx, uint32_t gy, uint32_t u, bool rw) const {
            return grid_x == gx && grid_y == gy && units == u && row_wise == rw;
        }
    };
    static thread_local CacheEntry work_cache;

    // Resolve grid from device (cached — grid never changes for a given device)
    auto [it, inserted] = grid_cache.try_emplace(device, CoreCoord{0, 0});
    if (inserted) {
        it->second = device->compute_with_storage_grid_size();
    }
    const auto& grid = it->second;

    // Check work split cache
    if (!work_cache.matches(grid.x, grid.y, units_to_divide, row_wise)) {
        auto [nc, ac, cg1, cg2, upcg1, upcg2] = split_work_to_cores(grid, units_to_divide, row_wise);
        work_cache.grid_x = grid.x;
        work_cache.grid_y = grid.y;
        work_cache.units = units_to_divide;
        work_cache.row_wise = row_wise;
        work_cache.result.num_cores = nc;
        work_cache.result.all_cores = std::move(ac);
        work_cache.result.core_group_1 = std::move(cg1);
        work_cache.result.core_group_2 = std::move(cg2);
        work_cache.result.units_per_core_group_1 = upcg1;
        work_cache.result.units_per_core_group_2 = upcg2;
        work_cache.result.cores = grid_to_cores(nc, grid.x, grid.y, row_wise);
    }

    return work_cache.result;
}

}  // namespace tt::tt_metal
