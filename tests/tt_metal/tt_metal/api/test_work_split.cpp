// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <tuple>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/work_split.hpp>
#include <tt-metalium/work_split.hpp>

namespace tt::tt_metal {
namespace {

using experimental::split_work_to_cores_counts;
using experimental::WorkSplitCounts;

using SplitWorkToCoresResult = std::tuple<uint32_t, CoreRangeSet, CoreRangeSet, CoreRangeSet, uint32_t, uint32_t>;

// split_work_to_cores_counts gives the numbers of split_work_to_cores, and they account for every unit.
void expect_counts_match(const WorkSplitCounts& counts, const SplitWorkToCoresResult& split, uint32_t units_to_divide) {
    const auto& [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split;
    EXPECT_EQ(counts.num_cores, num_cores);
    EXPECT_EQ(counts.num_cores, all_cores.num_cores());
    EXPECT_EQ(counts.num_cores_group_1, core_group_1.num_cores());
    EXPECT_EQ(counts.num_cores - counts.num_cores_group_1, core_group_2.num_cores());
    EXPECT_EQ(counts.units_per_core_group_1, units_per_core_group_1);
    EXPECT_EQ(counts.units_per_core_group_2, units_per_core_group_2);
    EXPECT_EQ(
        (counts.num_cores_group_1 * counts.units_per_core_group_1) +
            ((counts.num_cores - counts.num_cores_group_1) * counts.units_per_core_group_2),
        units_to_divide);
}

TEST(WorkSplit, CountsMatchSplitWorkToCoresOnGrid) {
    for (const CoreCoord& grid_size :
         {CoreCoord(1, 1), CoreCoord(1, 8), CoreCoord(8, 1), CoreCoord(3, 5), CoreCoord(8, 8), CoreCoord(13, 10)}) {
        const auto max_num_cores = static_cast<uint32_t>(grid_size.x * grid_size.y);
        for (uint32_t units = 0; units <= (3 * max_num_cores) + 1; ++units) {
            for (const bool row_wise : {false, true}) {
                SCOPED_TRACE(
                    testing::Message() << "grid " << grid_size.str() << ", units " << units << ", row_wise "
                                       << row_wise);
                expect_counts_match(
                    split_work_to_cores_counts(max_num_cores, units),
                    split_work_to_cores(grid_size, units, row_wise),
                    units);
            }
        }
    }
}

TEST(WorkSplit, CountsMatchSplitWorkToCoresOnCoreRangeSet) {
    const CoreRangeSet core_grid(std::vector<CoreRange>{CoreRange({0, 0}, {3, 1}), CoreRange({5, 0}, {6, 2})});
    const auto max_num_cores = static_cast<uint32_t>(core_grid.num_cores());
    for (uint32_t units = 0; units <= (3 * max_num_cores) + 1; ++units) {
        for (const bool row_wise : {false, true}) {
            SCOPED_TRACE(testing::Message() << "units " << units << ", row_wise " << row_wise);
            expect_counts_match(
                split_work_to_cores_counts(max_num_cores, units),
                split_work_to_cores(core_grid, units, row_wise),
                units);
        }
    }
}

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> counts_tuple(const WorkSplitCounts& counts) {
    return {counts.num_cores, counts.num_cores_group_1, counts.units_per_core_group_1, counts.units_per_core_group_2};
}

TEST(WorkSplit, KnownCounts) {
    // 100 units over 64 cores: the first 36 cores take 2 units, the other 28 take 1.
    EXPECT_EQ(counts_tuple(split_work_to_cores_counts(64, 100)), std::make_tuple(64U, 36U, 2U, 1U));
    // An even split puts every core in group 1.
    EXPECT_EQ(counts_tuple(split_work_to_cores_counts(64, 128)), std::make_tuple(64U, 64U, 2U, 0U));
    // Fewer units than cores: one unit on each of the first 10 cores.
    EXPECT_EQ(counts_tuple(split_work_to_cores_counts(64, 10)), std::make_tuple(10U, 10U, 1U, 0U));
    EXPECT_EQ(counts_tuple(split_work_to_cores_counts(64, 0)), std::make_tuple(0U, 0U, 0U, 0U));
    EXPECT_ANY_THROW(split_work_to_cores_counts(0, 1));
}

}  // namespace
}  // namespace tt::tt_metal
