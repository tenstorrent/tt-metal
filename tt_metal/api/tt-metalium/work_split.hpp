// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// Contains utility functions for partitioning work between multiple cores.
//

#pragma once

#include <cstdint>
#include <tuple>
#include <vector>

#include <tt-metalium/core_coord.hpp>

namespace tt::tt_metal {

uint32_t merge_num_sticks_to_read(uint32_t num_sticks_to_read, uint32_t stick_size_bytes, uint32_t max_read_size);

// Given a number of tiles and number of cores available
// Set the largest number of cores less than the number of tiles
// Returns the number of cores as well as the number of tiles per core
std::tuple<uint32_t, uint32_t> get_max_cores_divisible_by_tiles_per_core_tiles(
    const uint32_t& num_tiles, const uint32_t& num_cores_max, bool request_even = false);

// Finds the maximum divisor (excluding 5 or 7) of val starting at start_max_div and below
int find_max_divisor(uint32_t val, uint32_t start_max_div);

CoreRangeSet num_cores_to_corerangeset(
    CoreCoord start_core, uint32_t target_num_cores, CoreCoord grid_size, bool row_wise = false);

// TODO: Get rid of old function
CoreRangeSet num_cores_to_corerangeset(uint32_t target_num_cores, CoreCoord grid_size, bool row_wise = false);

CoreRangeSet num_cores_to_corerangeset_in_subcoregrids(
    CoreCoord start_core, uint32_t target_num_cores, const CoreRangeSet& sub_core_grids, bool row_wise = false);

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

// This function takes in the core grid size, as well as the number of units of work to divide between the cores.
// This function returns the number of cores, the CoreRangeSet of all cores, and then the CoreRangeSet that does
// the greater amount of work, and the CoreRangeSet that does less work if work cannot be evenly divided
// If it can be evenly divided, the second CoreRangeSet is the same as the first, and the last is empty
// The last 2 elements of the tuple are the units of work for the two core grids.
// This overload assumes that the core grid starts at (0, 0).
std::tuple<uint32_t, CoreRangeSet, CoreRangeSet, CoreRangeSet, uint32_t, uint32_t> split_work_to_cores(
    CoreCoord grid_size, uint32_t units_to_divide, bool row_wise = false);

// This overload takes in an arbitrary CoreRangeSet of cores, and the number of units of work to divide
// between the cores.
std::tuple<uint32_t, CoreRangeSet, CoreRangeSet, CoreRangeSet, uint32_t, uint32_t> split_work_to_cores(
    const CoreRangeSet& core_grid, uint32_t units_to_divide, bool row_wise = false);

// Splits up work in chunks of `multiple` . Not the most efficient distribution but useful for an even distribution
// of the batch dimension
std::tuple<std::vector<uint32_t>, CoreRangeSet> split_work_to_cores_even_multiples(
    const CoreCoord& core_grid, uint32_t units_to_divide, uint32_t multiple, bool row_wise = false);

}  // namespace tt::tt_metal
