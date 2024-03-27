// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// Contains utility functions for partitioning work between multiple cores.
//

#pragma once

#include "tt_metal/common/assert.hpp"
#include "tt_metal/common/core_coord.h"
#include "tt_metal/common/math.hpp"

#include "tt_metal/host_api.hpp"


namespace tt {
namespace tt_metal {

// splits the tiles evenly between num_cores,
// with option of padding where necessary
struct TilesSplit {
    int num_cores_;
    int total_tiles_;
    int tpc_; // unclipped tiles per core

    inline TilesSplit(int num_cores, int total_tiles) : num_cores_(num_cores), total_tiles_(total_tiles) {
        tpc_ = div_up(total_tiles_, num_cores_);
    }

    // number of tiles per core for div_up split
    inline uint32_t get_tpc() const { return tpc_; }

    // number of tiles per core for close to even split with multiples of 8 going to each core
    inline uint32_t get_clipped_tpc(int icore) const {
        auto result = ( tpc_*(icore+1) > total_tiles_ ) ? ( total_tiles_ - tpc_*(icore+1) ) : tpc_;
        return result;
    }
};

struct CoreGridDesc {
    uint32_t x_, y_;
    CoreGridDesc(Device* dev) { auto gs = dev->compute_with_storage_grid_size(); x_ = gs.x; y_ = gs.y; TT_ASSERT(x_ > 0 && y_ > 0); }
    uint32_t total_cores() const { return x_*y_; }
    CoreCoord wrap_core(int icore) const {
        TT_ASSERT(icore < total_cores());
        CoreCoord core = {(std::size_t) icore % x_, (std::size_t) icore / x_};
        return core;
    }

    int numcores_dividing_numtiles(int num_tiles, int block_size = 1) const {
        // since we will be splitting num_tiles into num_cores we need to find num_cores such that
        // num_tiles % num_cores = 0, so that it's evenly divided since we don't support leftovers at the moment
        // TODO(AP): optimize if needed, O(max_cores) atm
        uint32_t max_cores = total_cores();
        TT_ASSERT(max_cores % block_size == 0 || max_cores == 1);
        if (max_cores > num_tiles)
            max_cores = num_tiles;
        for (int j = max_cores; j >= 1; j--)
            if (num_tiles % j == 0)
                return j;
        return 1;
    }
};

// Given a number of tiles and number of cores available
// Set the largest number of cores less than the number of tiles
// Returns the number of cores as well as the number of tiles per core

inline std::tuple<uint32_t, uint32_t> get_max_cores_divisible_by_tiles_per_core_tiles(
    const uint32_t &num_tiles, const uint32_t &num_cores_max, bool request_even = false) {
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
    if (num_tiles % num_cores != 0)
        per_core_tiles_dim++;
    return {num_cores, per_core_tiles_dim};
}

// Finds the maximum even divisor of val starting at start_max_div and below
inline int find_max_divisor(uint32_t val, uint32_t start_max_div) {
    int result = 1;
    for (int find_divisor = start_max_div; find_divisor >= 1; find_divisor--) {
        if (find_divisor == 7 || find_divisor == 5)
            continue;
        if (val % find_divisor == 0) {
            result = find_divisor;
            break;
        }
    }
    return result;
}

inline int find_max_block_size(uint32_t val, uint32_t max_block_size=8) {
    int result = 1;
    for (int find_divisor = max_block_size; find_divisor >= 1; find_divisor--) {
        if (val % find_divisor == 0) {
            result = find_divisor;
            break;
        }
    }
    return result;
}

inline std::set<CoreRange> num_cores_to_corerange_set(const CoreCoord start_core, const uint32_t target_num_cores, const CoreCoord grid_size, const bool row_wise = false) {
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
	TT_FATAL(target_num_cores <= total_available_cores, "Target number of cores {} is greater than total number of available cores {}", target_num_cores, total_available_cores);
	std::set<CoreRange> all_cores_set;
    uint32_t leftover_size = target_num_cores;
    CoreCoord s_core = start_core;
    if (row_wise) {
        // Partial row at start
        if (s_core.x != 0 && leftover_size > num_cores_x - start_core.x) {
            CoreRange start_block(s_core, {num_cores_x - 1, s_core.y});
            all_cores_set.insert(start_block);
            s_core = {0, s_core.y + 1};
            leftover_size -= start_block.size();
        }
        // Full rows
        if (leftover_size > num_cores_x) {
            uint32_t num_full_rows = leftover_size / num_cores_x;
            CoreRange full_block(s_core, {num_cores_x - 1, s_core.y + num_full_rows - 1});
            all_cores_set.insert(full_block);
            leftover_size -= full_block.size();
            s_core = {0, s_core.y + num_full_rows};
        }
        // Partial row at end
        if (leftover_size > 0) {
            CoreRange leftover_block(s_core, {s_core.x + leftover_size - 1, s_core.y});
            all_cores_set.insert(leftover_block);
        }
    } else {
        // Partial col at start
        if (s_core.y != 0 && leftover_size > num_cores_y - start_core.y) {
            CoreRange start_block(s_core, {s_core.x, num_cores_y - 1});
            all_cores_set.insert(start_block);
            s_core = {s_core.x + 1, 0};
            leftover_size -= start_block.size();
        }
        // Full cols
        if (leftover_size > num_cores_y) {
            uint32_t num_full_cols = leftover_size / num_cores_y;
            CoreRange full_block(s_core, {s_core.x + num_full_cols - 1, num_cores_y - 1});
            all_cores_set.insert(full_block);
            leftover_size -= full_block.size();
            s_core = {s_core.x + num_full_cols, 0};
        }
        // Partial row at end
        if (leftover_size > 0) {
            CoreRange leftover_block(s_core, {s_core.x, s_core.y + leftover_size - 1});
            all_cores_set.insert(leftover_block);
        }
    }
	return all_cores_set;
}

// TODO: Get rid of old function
inline std::set<CoreRange> num_cores_to_corerange_set(const uint32_t target_num_cores, const CoreCoord grid_size, const bool row_wise = false) {
	return num_cores_to_corerange_set({0, 0}, target_num_cores, grid_size, row_wise);
}

// This function takes in the core grid size, as well as the number of units of work to divide between the cores
// This function returns the number of cores, the CoreRangeSet of all cores, and then the CoreRangeSet that does
// the greater amount of work, and the CoreRangeSet that does less work if work cannot be evenly divided
// If it can be evenly divided, the second CoreRangeSet is the same as the first, and the last is empty
// The last 2 args are the units of work for the two core grids
inline std::tuple<uint32_t, CoreRangeSet, CoreRangeSet, CoreRangeSet, uint32_t, uint32_t> split_work_to_cores(const CoreCoord grid_size, const uint32_t units_to_divide, const bool row_wise = false) {
    ZoneScoped;
	uint32_t num_cores_x = grid_size.x, num_cores_y = grid_size.y;
	auto target_num_cores = std::min(units_to_divide, num_cores_x * num_cores_y);
	CoreRangeSet all_cores(num_cores_to_corerange_set(target_num_cores, grid_size, row_wise));

	std::set<CoreRange> core_group_1_set;
	std::set<CoreRange> core_group_2_set;
	uint32_t units_per_core_group_1 = units_to_divide / target_num_cores;
	uint32_t units_per_core_group_2 = 0;
    // Evenly divided units to all target cores
	if (units_to_divide % target_num_cores == 0) {
		core_group_1_set = all_cores.ranges();
    // Uneven division of units across cores
    // This case should only be hit when there are more units of work than a full grid of cores
    // which is implicitly assumed in the following logic
	} else {
        // Group of cores that do more work
        core_group_1_set = num_cores_to_corerange_set(units_to_divide % target_num_cores, grid_size, row_wise);
        auto last_block_group_1 = (*core_group_1_set.rbegin());
        auto last_block_all_cores = (*all_cores.ranges().rbegin());
        if (row_wise) {
            // Case where only the last row is divided between core group 1 and 2
            if (last_block_group_1.end.y == last_block_all_cores.end.y && last_block_group_1.end.x != last_block_all_cores.end.x) {
                CoreRange leftover_block(
                    {last_block_group_1.end.x + 1, last_block_group_1.end.y},
                    last_block_all_cores.end
                );
                core_group_2_set.insert(leftover_block);
            } else {
                // Case where a middle row is divided between core group 1 and 2
                if (last_block_group_1.end.x != num_cores_x - 1) {
                    CoreRange leftover_stick(
                        {last_block_group_1.end.x + 1, last_block_group_1.end.y},
                        {num_cores_x - 1, last_block_group_1.end.y}
                    );
                    core_group_2_set.insert(leftover_stick);
                }
                // Remaining rows of cores that does less work
                CoreRange leftover_block(
                    {0, last_block_group_1.end.y + 1},
                    last_block_all_cores.end
                );
                core_group_2_set.insert(leftover_block);
            }
        } else {
            // Case where only the last column is divided between core group 1 and 2
            if (last_block_group_1.end.x == last_block_all_cores.end.x && last_block_group_1.end.y != last_block_all_cores.end.y) {
                CoreRange leftover_block(
                    {last_block_group_1.end.x, last_block_group_1.end.y + 1},
                    last_block_all_cores.end
                );
                core_group_2_set.insert(leftover_block);
            } else {
                // Case where a middle column is divided between core group 1 and 2
                if (last_block_group_1.end.y != num_cores_y - 1) {
                    CoreRange leftover_stick(
                        {last_block_group_1.end.x, last_block_group_1.end.y + 1},
                        {last_block_group_1.end.x, num_cores_y - 1}
                    );
                    core_group_2_set.insert(leftover_stick);
                }
                // Remaining columns of cores that does less work
                CoreRange leftover_block(
                    {last_block_group_1.end.x + 1, 0},
                    last_block_all_cores.end
                );
                core_group_2_set.insert(leftover_block);
            }
        }
		units_per_core_group_2 = units_per_core_group_1;
        units_per_core_group_1++;
	}
	CoreRangeSet core_group_1(core_group_1_set);
	CoreRangeSet core_group_2(core_group_2_set);

	return std::make_tuple(target_num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2);
}

} // namespace tt_metal
} // namespace tt
