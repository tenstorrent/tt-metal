// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// Contains utility functions for partitioning work between multiple cores.
//

#include <cstdint>
#include <tuple>
#include <vector>

#include "tt_metal/common/assert.hpp"
#include "tt_metal/common/core_coord.hpp"
#include "tt_metal/common/math.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"

namespace tt {
namespace tt_metal {

uint32_t merge_num_sticks_to_read(uint32_t num_sticks_to_read, uint32_t stick_size_bytes, uint32_t max_read_size) {
    uint32_t total_bytes = num_sticks_to_read * stick_size_bytes;
    uint32_t new_num_sticks_to_read = num_sticks_to_read;
    uint32_t new_stick_size_bytes = stick_size_bytes;

    for (uint32_t current_size = stick_size_bytes; current_size <= max_read_size; current_size += stick_size_bytes) {
        if (total_bytes % current_size == 0) {
            new_stick_size_bytes = current_size;
            new_num_sticks_to_read = total_bytes / current_size;
        }
    }
    return new_num_sticks_to_read;
}

std::tuple<uint32_t, uint32_t> get_max_cores_divisible_by_tiles_per_core_tiles(
    const uint32_t& num_tiles, const uint32_t& num_cores_max, bool request_even) {
    uint32_t num_cores = 1;
    for (int i = 2; i <= num_cores_max; i++) {
        if ((num_tiles % i) == 0) {
            num_cores = i;
        }
    }
    if (request_even) {
        num_cores = num_cores - num_cores % 2;
    }
    uint32_t per_core_tiles_dim = num_tiles / num_cores;
    if (num_tiles % num_cores != 0) {
        per_core_tiles_dim++;
    }
    return {num_cores, per_core_tiles_dim};
}

int find_max_divisor(uint32_t val, uint32_t start_max_div) {
    int result = 1;
    for (int find_divisor = start_max_div; find_divisor >= 1; find_divisor--) {
        if (find_divisor == 7 || find_divisor == 5) {
            continue;
        }
        if (val % find_divisor == 0) {
            result = find_divisor;
            break;
        }
    }
    return result;
}

int find_max_block_size(uint32_t val, uint32_t max_block_size) {
    int result = 1;
    for (int find_divisor = max_block_size; find_divisor >= 1; find_divisor--) {
        if (val % find_divisor == 0) {
            result = find_divisor;
            break;
        }
    }
    return result;
}

CoreRangeSet num_cores_to_corerangeset(
    const CoreCoord start_core, const uint32_t target_num_cores, const CoreCoord grid_size, const bool row_wise) {
    uint32_t num_cores_x = grid_size.x;
    uint32_t num_cores_y = grid_size.y;
    uint32_t total_available_cores = 0;
    TT_FATAL(start_core.x < num_cores_x && start_core.y < num_cores_y, "Start core must be within grid size");
    if (row_wise) {
        // Full Rows
        total_available_cores += (num_cores_y - 1 - start_core.y) * num_cores_x;
        // Partial Rows
        total_available_cores += num_cores_x - start_core.x;
    } else {
        // Full Cols
        total_available_cores += (num_cores_x - 1 - start_core.x) * num_cores_y;
        // Partial Cols
        total_available_cores += num_cores_y - start_core.y;
    }
    TT_FATAL(
        target_num_cores <= total_available_cores,
        "Target number of cores {} is greater than total number of available cores {}",
        target_num_cores,
        total_available_cores);

    // 3 is the max number of ranges that will be generated when splitting a grid
    std::vector<CoreRange> all_cores;
    all_cores.reserve(3);
    uint32_t leftover_size = target_num_cores;
    CoreCoord s_core = start_core;
    if (row_wise) {
        // Partial row at start
        if (s_core.x != 0 && leftover_size > num_cores_x - start_core.x) {
            all_cores.emplace_back(s_core, CoreCoord(num_cores_x - 1, s_core.y));
            s_core = {0, s_core.y + 1};
            leftover_size -= all_cores.back().size();
        }
        // Full rows
        if (leftover_size > num_cores_x) {
            uint32_t num_full_rows = leftover_size / num_cores_x;
            all_cores.emplace_back(s_core, CoreCoord(num_cores_x - 1, s_core.y + num_full_rows - 1));
            leftover_size -= all_cores.back().size();
            s_core = {0, s_core.y + num_full_rows};
        }
        // Partial row at end
        if (leftover_size > 0) {
            all_cores.emplace_back(s_core, CoreCoord(s_core.x + leftover_size - 1, s_core.y));
        }
    } else {
        // Partial col at start
        if (s_core.y != 0 && leftover_size > num_cores_y - start_core.y) {
            all_cores.emplace_back(s_core, CoreCoord(s_core.x, num_cores_y - 1));
            s_core = {s_core.x + 1, 0};
            leftover_size -= all_cores.back().size();
        }
        // Full cols
        if (leftover_size > num_cores_y) {
            uint32_t num_full_cols = leftover_size / num_cores_y;
            all_cores.emplace_back(s_core, CoreCoord(s_core.x + num_full_cols - 1, num_cores_y - 1));
            leftover_size -= all_cores.back().size();
            s_core = {s_core.x + num_full_cols, 0};
        }
        // Partial row at end
        if (leftover_size > 0) {
            all_cores.emplace_back(s_core, CoreCoord(s_core.x, s_core.y + leftover_size - 1));
        }
    }
    return CoreRangeSet(std::move(all_cores));
}

CoreRangeSet num_cores_to_corerangeset(
    const uint32_t target_num_cores, const CoreCoord grid_size, const bool row_wise) {
    return num_cores_to_corerangeset({0, 0}, target_num_cores, grid_size, row_wise);
}

std::tuple<uint32_t, CoreRangeSet, CoreRangeSet, CoreRangeSet, uint32_t, uint32_t> split_work_to_cores(
    const CoreCoord grid_size, const uint32_t units_to_divide, const bool row_wise) {
    ZoneScoped;
    uint32_t num_cores_x = grid_size.x, num_cores_y = grid_size.y;
    auto target_num_cores = std::min(units_to_divide, num_cores_x * num_cores_y);
    CoreRangeSet all_cores = num_cores_to_corerangeset(target_num_cores, grid_size, row_wise);

    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t units_per_core_group_1 = target_num_cores == 0 ? 0 : units_to_divide / target_num_cores;
    uint32_t units_per_core_group_2 = 0;
    // Evenly divided units to all target cores
    if (target_num_cores == 0 || units_to_divide % target_num_cores == 0) {
        core_group_1 = all_cores;
        // Uneven division of units across cores
        // This case should only be hit when there are more units of work than a full grid of cores
        // which is implicitly assumed in the following logic
    } else {
        // Group of cores that do more work
        core_group_1 = num_cores_to_corerangeset(units_to_divide % target_num_cores, grid_size, row_wise);
        const auto& last_block_group_1 = (*core_group_1.ranges().rbegin());
        const auto& last_block_all_cores = (*all_cores.ranges().rbegin());
        if (row_wise) {
            // Case where only the last row is divided between core group 1 and 2
            if (last_block_group_1.end_coord.y == last_block_all_cores.end_coord.y &&
                last_block_group_1.end_coord.x != last_block_all_cores.end_coord.x) {
                CoreRange leftover_block(
                    CoreCoord(last_block_group_1.end_coord.x + 1, last_block_group_1.end_coord.y),
                    last_block_all_cores.end_coord);
                core_group_2 = CoreRangeSet(leftover_block);
            } else {
                std::vector<CoreRange> core_group_2_set;
                // Case where a middle row is divided between core group 1 and 2
                if (last_block_group_1.end_coord.x != num_cores_x - 1) {
                    core_group_2_set.emplace_back(
                        CoreCoord(last_block_group_1.end_coord.x + 1, last_block_group_1.end_coord.y),
                        CoreCoord(num_cores_x - 1, last_block_group_1.end_coord.y));
                }
                // Remaining rows of cores that does less work
                core_group_2_set.emplace_back(
                    CoreCoord(0, last_block_group_1.end_coord.y + 1), last_block_all_cores.end_coord);
                core_group_2 = CoreRangeSet(std::move(core_group_2_set));
            }
        } else {
            // Case where only the last column is divided between core group 1 and 2
            if (last_block_group_1.end_coord.x == last_block_all_cores.end_coord.x &&
                last_block_group_1.end_coord.y != last_block_all_cores.end_coord.y) {
                CoreRange leftover_block(
                    CoreCoord(last_block_group_1.end_coord.x, last_block_group_1.end_coord.y + 1),
                    last_block_all_cores.end_coord);
                core_group_2 = CoreRangeSet(leftover_block);
            } else {
                std::vector<CoreRange> core_group_2_set;
                // Case where a middle column is divided between core group 1 and 2
                if (last_block_group_1.end_coord.y != num_cores_y - 1) {
                    core_group_2_set.emplace_back(
                        CoreCoord(last_block_group_1.end_coord.x, last_block_group_1.end_coord.y + 1),
                        CoreCoord(last_block_group_1.end_coord.x, num_cores_y - 1));
                }
                // Remaining columns of cores that does less work
                core_group_2_set.emplace_back(
                    CoreCoord(last_block_group_1.end_coord.x + 1, 0), last_block_all_cores.end_coord);
                core_group_2 = CoreRangeSet(std::move(core_group_2_set));
            }
        }
        units_per_core_group_2 = units_per_core_group_1;
        units_per_core_group_1++;
    }

    return std::make_tuple(
        target_num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2);
}

}  // namespace tt_metal
}  // namespace tt
