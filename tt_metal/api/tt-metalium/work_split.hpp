// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// Contains utility functions for partitioning work between multiple cores.
//

#pragma once

#include <cstdint>
#include <tuple>
#include <vector>

#include "core_coord.hpp"

namespace tt {
namespace tt_metal {

uint32_t merge_num_sticks_to_read(uint32_t num_sticks_to_read, uint32_t stick_size_bytes, uint32_t max_read_size);

// Given a number of tiles and number of cores available
// Set the largest number of cores less than the number of tiles
// Returns the number of cores as well as the number of tiles per core
std::tuple<uint32_t, uint32_t> get_max_cores_divisible_by_tiles_per_core_tiles(
    const uint32_t& num_tiles, const uint32_t& num_cores_max, bool request_even = false);

// Finds the maximum divisor (excluding 5 or 7) of val starting at start_max_div and below
int find_max_divisor(uint32_t val, uint32_t start_max_div);

int find_max_block_size(uint32_t val, uint32_t max_block_size = 8);

CoreRangeSet num_cores_to_corerangeset(
    const CoreCoord start_core,
    const uint32_t target_num_cores,
    const CoreCoord grid_size,
    const bool row_wise = false);

// TODO: Get rid of old function
CoreRangeSet num_cores_to_corerangeset(
    const uint32_t target_num_cores, const CoreCoord grid_size, const bool row_wise = false);

CoreRangeSet num_cores_to_corerangeset_in_subcoregrids(
    const CoreCoord start_core,
    const uint32_t target_num_cores,
    const CoreRangeSet& sub_core_grids,
    const bool row_wise = false);
// This function takes in the core grid size, as well as the number of units of work to divide between the cores
// This function returns the number of cores, the CoreRangeSet of all cores, and then the CoreRangeSet that does
// the greater amount of work, and the CoreRangeSet that does less work if work cannot be evenly divided
// If it can be evenly divided, the second CoreRangeSet is the same as the first, and the last is empty
// The last 2 args are the units of work for the two core grids
std::tuple<uint32_t, CoreRangeSet, CoreRangeSet, CoreRangeSet, uint32_t, uint32_t> split_work_to_cores(
    const CoreCoord grid_size, const uint32_t units_to_divide, const bool row_wise = false);

std::tuple<uint32_t, CoreRangeSet, CoreRangeSet, CoreRangeSet, uint32_t, uint32_t> split_work_to_cores(
    const CoreRangeSet& core_grid, const uint32_t units_to_divide, const bool row_wise = false);

}  // namespace tt_metal
}  // namespace tt
