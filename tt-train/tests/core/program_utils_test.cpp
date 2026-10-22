// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host-only tests for the core walk every tt-train program factory uses for its runtime arguments. No device:
// split_work_to_cores and CoreRangeSet are plain host code, so this pins the traversal contract directly rather
// than through an op whose shape happens to produce one core or one work group.

#include "metal/common/program_utils.hpp"

#include <gtest/gtest.h>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/work_split.hpp>
#include <vector>

namespace {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;
using ttml::metal::CoreWork;
using ttml::metal::for_each_core;
using ttml::metal::for_each_core_with_work;

// The walk order every tt-train reader/writer assumes: core i -> {i / num_cores_y, i % num_cores_y}.
CoreCoord expected_core(uint32_t i, uint32_t num_cores_y) {
    return CoreCoord{i / num_cores_y, i % num_cores_y};
}

std::vector<CoreWork> collect(
    uint32_t num_cores,
    uint32_t num_cores_y,
    const CoreRangeSet& group_1,
    const CoreRangeSet& group_2,
    uint32_t units_1,
    uint32_t units_2) {
    std::vector<CoreWork> seen;
    for_each_core_with_work(num_cores, num_cores_y, group_1, group_2, units_1, units_2, [&](const CoreWork& work) {
        seen.push_back(work);
    });
    return seen;
}

}  // namespace

TEST(ProgramUtilsCoreWalk, TwoGroupsPartialLastColumnAndCumulativeStart) {
    // 3x4 grid, 14 units: split_work_to_cores gives group 1 two cores with 2 units and group 2 ten cores with 1,
    // i.e. all 12 cores work, the groups meet mid-walk and the last column is full. The walk must visit column by
    // column, hand each core its group's unit count and a start offset equal to everything handed out before it.
    const CoreCoord grid{3, 4};
    const uint32_t num_cores_y = grid.y;
    auto [num_cores, all_cores, group_1, group_2, units_1, units_2] = tt::tt_metal::split_work_to_cores(grid, 14U);
    ASSERT_EQ(num_cores, 12U);
    ASSERT_EQ(units_1, 2U);
    ASSERT_EQ(units_2, 1U);

    const auto seen = collect(num_cores, num_cores_y, group_1, group_2, units_1, units_2);
    ASSERT_EQ(seen.size(), num_cores);

    uint32_t start = 0U;
    bool saw_group_2 = false;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& work = seen[i];
        EXPECT_EQ(work.core, expected_core(i, num_cores_y)) << "core " << i;
        EXPECT_EQ(work.index, i);
        EXPECT_EQ(work.in_group_1, group_1.contains(work.core)) << "core " << i;
        EXPECT_EQ(work.num_units, work.in_group_1 ? units_1 : units_2) << "core " << i;
        EXPECT_EQ(work.start, start) << "core " << i;
        // Once the walk enters group 2 it never returns to group 1.
        if (!work.in_group_1) {
            saw_group_2 = true;
        }
        EXPECT_FALSE(work.in_group_1 && saw_group_2) << "group 1 core after a group 2 core at " << i;
        start += work.num_units;
    }
    EXPECT_EQ(start, 14U);
    EXPECT_TRUE(saw_group_2);
}

TEST(ProgramUtilsCoreWalk, PartialLastColumnWhenFewerUnitsThanCores) {
    // 3x4 grid, 10 units: 10 cores of one unit each, so the last column {2, y} only has y = 0 and y = 1, and the
    // walk stops there instead of visiting the idle cores {2, 2} and {2, 3}.
    const CoreCoord grid{3, 4};
    auto [num_cores, all_cores, group_1, group_2, units_1, units_2] = tt::tt_metal::split_work_to_cores(grid, 10U);
    ASSERT_EQ(num_cores, 10U);
    ASSERT_TRUE(group_2.ranges().empty());

    const auto seen = collect(num_cores, grid.y, group_1, group_2, units_1, units_2);
    ASSERT_EQ(seen.size(), 10U);
    EXPECT_EQ(seen.back().core, (CoreCoord{2, 1}));
    EXPECT_EQ(seen.back().start, 9U);
    for (const auto& work : seen) {
        EXPECT_TRUE(work.in_group_1);
        EXPECT_EQ(work.num_units, 1U);
        EXPECT_TRUE(all_cores.contains(work.core));
    }
}

TEST(ProgramUtilsCoreWalk, SingleCore) {
    const CoreCoord grid{3, 4};
    auto [num_cores, all_cores, group_1, group_2, units_1, units_2] = tt::tt_metal::split_work_to_cores(grid, 1U);
    ASSERT_EQ(num_cores, 1U);

    const auto seen = collect(num_cores, grid.y, group_1, group_2, units_1, units_2);
    ASSERT_EQ(seen.size(), 1U);
    EXPECT_EQ(seen[0].core, (CoreCoord{0, 0}));
    EXPECT_EQ(seen[0].index, 0U);
    EXPECT_EQ(seen[0].num_units, 1U);
    EXPECT_EQ(seen[0].start, 0U);
    EXPECT_TRUE(seen[0].in_group_1);
}

TEST(ProgramUtilsCoreWalk, CoreInNeitherGroupIsAnError) {
    // Groups that do not cover the walk: {0,0} and {0,1} are handed out, {0,2} is in neither set.
    const CoreRangeSet group_1(CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}});
    const CoreRangeSet group_2(CoreRange{CoreCoord{0, 1}, CoreCoord{0, 1}});
    uint32_t visited = 0U;
    EXPECT_THROW(
        for_each_core_with_work(3U, 4U, group_1, group_2, 1U, 1U, [&](const CoreWork&) { ++visited; }),
        std::runtime_error);
    EXPECT_EQ(visited, 2U);
}

TEST(ProgramUtilsCoreWalk, ForEachCoreVisitsTheSameOrderWithEitherSignature) {
    const uint32_t num_cores = 7U;
    const uint32_t num_cores_y = 3U;

    std::vector<CoreCoord> cores_only;
    for_each_core(num_cores, num_cores_y, [&](const CoreCoord& core) { cores_only.push_back(core); });

    std::vector<CoreCoord> cores_indexed;
    std::vector<uint32_t> indices;
    for_each_core(num_cores, num_cores_y, [&](const CoreCoord& core, uint32_t index) {
        cores_indexed.push_back(core);
        indices.push_back(index);
    });

    ASSERT_EQ(cores_only.size(), num_cores);
    ASSERT_EQ(cores_indexed, cores_only);
    for (uint32_t i = 0; i < num_cores; ++i) {
        EXPECT_EQ(cores_only[i], expected_core(i, num_cores_y)) << "core " << i;
        EXPECT_EQ(indices[i], i);
    }
    // 7 cores over 3 rows: columns 0 and 1 full, column 2 holds only row 0.
    EXPECT_EQ(cores_only.back(), (CoreCoord{2, 0}));
}
