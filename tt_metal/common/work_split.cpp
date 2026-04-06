// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// Contains utility functions for partitioning work between multiple cores.
//

#include <tt_stl/assert.hpp>
#include <core_coord.hpp>
#include <algorithm>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include "tracy/Tracy.hpp"
#include <umd/device/types/xy_pair.hpp>

namespace tt::tt_metal {

uint32_t merge_num_sticks_to_read(uint32_t num_sticks_to_read, uint32_t stick_size_bytes, uint32_t max_read_size) {
    uint32_t total_bytes = num_sticks_to_read * stick_size_bytes;
    uint32_t new_num_sticks_to_read = num_sticks_to_read;

    for (uint32_t current_size = stick_size_bytes; current_size <= max_read_size; current_size += stick_size_bytes) {
        if (total_bytes % current_size == 0) {
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
    if (request_even && num_cores > 1) {
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

CoreRangeSet num_cores_to_corerangeset_in_subcoregrids(
    const CoreCoord start_core,
    const uint32_t target_num_cores,
    const CoreRangeSet& sub_core_grids,
    const bool row_wise = false) {
    TT_FATAL(target_num_cores > 0, "Target number of cores must be greater than 0");
    TT_FATAL(
        target_num_cores <= sub_core_grids.num_cores(),
        "Target number of cores {} is greater than total available cores {}",
        target_num_cores,
        sub_core_grids.num_cores());

    TT_FATAL(sub_core_grids.contains(start_core), "Start core must be inside sub_core_grids");

    std::vector<CoreRange> result;
    uint32_t remaining = target_num_cores;

    bool in_active_grid = false;
    CoreCoord cur_start = start_core;
    CoreCoord cur_end = start_core;

    // ------------ Helpers -------------

    auto emit_current_range = [&](bool force = false) {
        if (force || (cur_start.x <= cur_end.x && cur_start.y <= cur_end.y)) {
            result.emplace_back(cur_start, cur_end);
        }
    };

    // Row-wise consumption (left to right, then next row)
    auto process_row = [&](const CoreRange& grid) {
        uint32_t y = cur_start.y;

        for (; y <= grid.end_coord.y && remaining > 0; ++y) {
            uint32_t row_start_x = (y == cur_start.y) ? cur_start.x : grid.start_coord.x;
            uint32_t row_end_x = grid.end_coord.x;

            uint32_t row_width = row_end_x - row_start_x + 1;
            uint32_t take = std::min(row_width, remaining);

            cur_start = {row_start_x, y};
            cur_end = {row_start_x + take - 1, y};

            emit_current_range(true);
            remaining -= take;
        }

        // Prepare start for next grid
        cur_start = {grid.start_coord.x, grid.start_coord.y};
        cur_end = cur_start;
    };

    // Column-wise consumption (top to bottom, then next column)
    auto process_col = [&](const CoreRange& grid) {
        uint32_t x = cur_start.x;

        for (; x <= grid.end_coord.x && remaining > 0; ++x) {
            uint32_t col_start_y = (x == cur_start.x) ? cur_start.y : grid.start_coord.y;
            uint32_t col_end_y = grid.end_coord.y;

            uint32_t col_height = col_end_y - col_start_y + 1;
            uint32_t take = std::min(col_height, remaining);

            cur_start = {x, col_start_y};
            cur_end = {x, col_start_y + take - 1};

            emit_current_range(true);
            remaining -= take;
        }

        // Prepare start for next grid
        cur_start = {grid.start_coord.x, grid.start_coord.y};
        cur_end = cur_start;
    };

    // ------------ Main Loop -------------

    for (const auto& grid : sub_core_grids.ranges()) {
        if (!in_active_grid) {
            if (!grid.contains(start_core)) {
                continue;
            }

            // First active grid
            in_active_grid = true;
            cur_start = start_core;
            cur_end = start_core;
        } else {
            // Move to the next grid cleanly
            cur_start = grid.start_coord;
            cur_end = cur_start;
        }

        if (row_wise) {
            process_row(grid);
        } else {
            process_col(grid);
        }

        if (remaining == 0) {
            break;
        }
    }

    auto merge_rectangles = [&](std::vector<CoreRange>& ranges, bool row_wise) {
        if (ranges.empty()) {
            return;
        }

        std::vector<CoreRange> merged;
        merged.reserve(ranges.size());

        // Start with first range
        CoreRange cur = ranges[0];

        for (size_t i = 1; i < ranges.size(); i++) {
            const CoreRange& nxt = ranges[i];

            if (row_wise) {
                // Merge consecutive rows with identical x-span
                bool same_x_span = (cur.start_coord.x == nxt.start_coord.x) && (cur.end_coord.x == nxt.end_coord.x);

                bool consecutive_row =
                    nxt.start_coord.y == cur.end_coord.y + 1 && nxt.end_coord.y == cur.end_coord.y + 1;

                if (same_x_span && consecutive_row) {
                    // Expand rectangle downward
                    cur.end_coord.y = nxt.end_coord.y;
                    continue;
                }
            } else {
                // Merge consecutive columns with identical y-span
                bool same_y_span = (cur.start_coord.y == nxt.start_coord.y) && (cur.end_coord.y == nxt.end_coord.y);

                bool consecutive_col =
                    nxt.start_coord.x == cur.end_coord.x + 1 && nxt.end_coord.x == cur.end_coord.x + 1;

                if (same_y_span && consecutive_col) {
                    // Expand rectangle sideways
                    cur.end_coord.x = nxt.end_coord.x;
                    continue;
                }
            }

            // If cannot merge, push current and start new
            merged.push_back(cur);
            cur = nxt;
        }

        // Push last accumulated range
        merged.push_back(cur);

        ranges = std::move(merged);
    };

    TT_FATAL(remaining == 0, "Failed to assign all {} requested cores", target_num_cores);

    // Merge consecutive rectangular ranges
    merge_rectangles(result, row_wise);

    // Return merged CoreRangeSet
    return CoreRangeSet(std::move(result));
}

std::tuple<std::vector<uint32_t>, CoreRangeSet> split_work_to_cores_even_multiples(
    const CoreCoord& core_grid, const uint32_t units_to_divide, const uint32_t multiple, const bool row_wise) {
    const uint32_t batches_to_divide = std::ceil(units_to_divide / multiple), max_num_cores = core_grid.x * core_grid.y;
    const uint32_t target_num_cores = (batches_to_divide >= max_num_cores) ? max_num_cores : batches_to_divide;

    std::vector<uint32_t> increments(target_num_cores, 0ul);
    auto it = increments.begin();
    for (uint32_t units = 0; units < units_to_divide; units += multiple) {
        *(it++) += multiple;
        if (it == increments.end()) {
            it = increments.begin();
        }
    }

    auto rem = units_to_divide % multiple;
    if (rem != 0) {
        *it += rem;
    }
    const auto utilized_cores = num_cores_to_corerangeset({0, 0}, target_num_cores, core_grid, row_wise);

    return std::make_tuple(increments, utilized_cores);
}

std::tuple<uint32_t, CoreRangeSet, CoreRangeSet, CoreRangeSet, uint32_t, uint32_t> split_work_to_cores(
    const CoreCoord grid_size, const uint32_t units_to_divide, const bool row_wise) {
    if (units_to_divide == 0) {
        return std::make_tuple(0, CoreRangeSet(), CoreRangeSet(), CoreRangeSet(), 0, 0);
    }
    uint32_t num_cores_x = grid_size.x, num_cores_y = grid_size.y, max_num_cores = num_cores_x * num_cores_y,
             target_num_cores;
    CoreRangeSet all_cores;
    if (units_to_divide >= max_num_cores) {
        target_num_cores = max_num_cores;
        all_cores = CoreRangeSet(CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1}));
    } else {
        target_num_cores = units_to_divide;
        all_cores = num_cores_to_corerangeset(target_num_cores, grid_size, row_wise);
    }

    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t units_per_core_group_1 = units_to_divide / target_num_cores;
    uint32_t units_per_core_group_2 = 0;
    uint32_t num_cores_with_more_work = units_to_divide % target_num_cores;
    // Evenly divided units to all target cores
    if (units_to_divide % target_num_cores == 0) {
        core_group_1 = all_cores;
    }
    // Uneven division of units across cores
    // This case should only be hit when there are more units of work than a full grid of cores
    // which is implicitly assumed in the following logic
    else {
        // Group of cores that do more work
        uint32_t num_core_group_1_cores = num_cores_with_more_work;
        uint32_t num_core_group_2_cores = target_num_cores - num_core_group_1_cores;
        core_group_1 = num_cores_to_corerangeset(num_core_group_1_cores, grid_size, row_wise);
        const auto& last_core_group_1 = (*core_group_1.ranges().rbegin()).end_coord;
        if (row_wise) {
            // Start in the same row
            if (last_core_group_1.x != num_cores_x - 1) {
                core_group_2 = num_cores_to_corerangeset(
                    {last_core_group_1.x + 1, last_core_group_1.y}, num_core_group_2_cores, grid_size, row_wise);
            }
            // Start in the next row
            else {
                core_group_2 = num_cores_to_corerangeset(
                    {0, last_core_group_1.y + 1}, num_core_group_2_cores, grid_size, row_wise);
            }
        } else {
            // Start in the same column
            if (last_core_group_1.y != num_cores_y - 1) {
                core_group_2 = num_cores_to_corerangeset(
                    {last_core_group_1.x, last_core_group_1.y + 1}, num_core_group_2_cores, grid_size, row_wise);
            }
            // Start in the next column
            else {
                core_group_2 = num_cores_to_corerangeset(
                    {last_core_group_1.x + 1, 0}, num_core_group_2_cores, grid_size, row_wise);
            }
        }
        units_per_core_group_2 = units_per_core_group_1;
        units_per_core_group_1++;
    }

    return std::make_tuple(
        target_num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2);
}

std::tuple<uint32_t, CoreRangeSet, CoreRangeSet, CoreRangeSet, uint32_t, uint32_t> split_work_to_cores(
    const CoreRangeSet& core_grid, const uint32_t units_to_divide, const bool row_wise) {
    if (units_to_divide == 0) {
        return std::make_tuple(0, CoreRangeSet(), CoreRangeSet(), CoreRangeSet(), 0, 0);
    }
    uint32_t max_num_cores = core_grid.num_cores(), target_num_cores;
    TT_FATAL(max_num_cores > 0, "Core grid must contain at least one core");
    auto start_core = core_grid.ranges().begin()->start_coord;
    CoreRangeSet all_cores;
    if (units_to_divide >= max_num_cores) {
        target_num_cores = max_num_cores;
        all_cores = core_grid;
    } else {
        target_num_cores = units_to_divide;
        all_cores = num_cores_to_corerangeset_in_subcoregrids(start_core, target_num_cores, core_grid, row_wise);
    }

    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t units_per_core_group_1 = units_to_divide / target_num_cores;
    uint32_t units_per_core_group_2 = 0;
    uint32_t num_cores_with_more_work = units_to_divide % target_num_cores;
    // Evenly divided units to all target cores
    if (target_num_cores == 0 || num_cores_with_more_work == 0) {
        core_group_1 = all_cores;
    }
    // Uneven division of units across cores
    // This case should only be hit when there are more units of work than a full grid of cores
    // which is implicitly assumed in the following logic
    else {
        // Group of cores that do more work
        uint32_t num_core_group_1_cores = num_cores_with_more_work;
        uint32_t num_core_group_2_cores = target_num_cores - num_core_group_1_cores;
        core_group_1 =
            num_cores_to_corerangeset_in_subcoregrids(start_core, num_core_group_1_cores, core_grid, row_wise);
        const auto& last_core_group_1 = (*core_group_1.ranges().rbegin()).end_coord;
        const auto& core_grid_ranges = core_grid.ranges();
        uint32_t num_cores_counted = 0, i;
        for (i = 0; i < core_grid_ranges.size(); i++) {
            num_cores_counted += core_grid_ranges[i].size();
            if (num_cores_counted >= num_core_group_1_cores) {
                break;
            }
        }
        const auto& range_containing_last_core_group_1 = core_grid_ranges[i];
        // Start in next core range
        if (last_core_group_1 == range_containing_last_core_group_1.end_coord) {
            core_group_2 = num_cores_to_corerangeset_in_subcoregrids(
                core_grid_ranges[i + 1].start_coord, num_core_group_2_cores, core_grid, row_wise);
        } else if (row_wise) {
            // Start in the same row
            if (last_core_group_1.x != range_containing_last_core_group_1.end_coord.x) {
                core_group_2 = num_cores_to_corerangeset_in_subcoregrids(
                    {last_core_group_1.x + 1, last_core_group_1.y}, num_core_group_2_cores, core_grid, row_wise);
            }
            // Start in the next row
            else {
                core_group_2 = num_cores_to_corerangeset_in_subcoregrids(
                    {range_containing_last_core_group_1.start_coord.x, last_core_group_1.y + 1},
                    num_core_group_2_cores,
                    core_grid,
                    row_wise);
            }
        } else {
            // Start in the same column
            if (last_core_group_1.y != range_containing_last_core_group_1.end_coord.y) {
                core_group_2 = num_cores_to_corerangeset_in_subcoregrids(
                    {last_core_group_1.x, last_core_group_1.y + 1}, num_core_group_2_cores, core_grid, row_wise);
            }
            // Start in the next column
            else {
                core_group_2 = num_cores_to_corerangeset_in_subcoregrids(
                    {last_core_group_1.x + 1, range_containing_last_core_group_1.end_coord.y},
                    num_core_group_2_cores,
                    core_grid,
                    row_wise);
            }
        }
        units_per_core_group_2 = units_per_core_group_1;
        units_per_core_group_1++;
    }

    return std::make_tuple(
        target_num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2);
}

}  // namespace tt::tt_metal
