// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-side tests for plan_row_slices, the split of RMSNorm tile-rows over the cores. It is plain arithmetic, so these
// run without a device and pin its answers on several grids, whatever grid the test machine has. Each pinned answer
// follows from the cost the planner minimizes:
//   cost(S) = ceil(rows / min(rows, cores / S)) * (St + (S > 1 ? S + kRowSliceExchangeCostTiles : 0)) + St,
// with St = ceil(Wt / S).

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "metal/ops/common/rmsnorm_common.hpp"

namespace {

using tt::tt_metal::CoreCoord;
using ttml::metal::ops::rmsnorm::for_each_row_slice_core;
using ttml::metal::ops::rmsnorm::kMaxRowSlices;
using ttml::metal::ops::rmsnorm::plan_row_slices;
using ttml::metal::ops::rmsnorm::row_slice_cores;
using ttml::metal::ops::rmsnorm::RowSlicePlan;
using ttml::metal::ops::rmsnorm::row_group_in_one_column;
using ttml::metal::ops::rmsnorm::RowSliceWork;

const CoreCoord kGrid8x8{8U, 8U};
const CoreCoord kGrid11x10{11U, 10U};
const CoreCoord kGrid13x10{13U, 10U};
const std::vector<CoreCoord> kGrids = {kGrid8x8, kGrid11x10, kGrid13x10};

RowSlicePlan plan(const CoreCoord grid, const uint32_t total_rows, const uint32_t Wt) {
    return plan_row_slices(grid, total_rows, Wt, kMaxRowSlices, [](uint32_t, uint32_t) { return true; });
}

std::string describe(const CoreCoord grid, const uint32_t total_rows, const uint32_t Wt) {
    return std::to_string(grid.x) + "x" + std::to_string(grid.y) + " grid, " + std::to_string(total_rows) +
           " rows, Wt = " + std::to_string(Wt);
}

void expect_plan_eq(const RowSlicePlan& actual, const RowSlicePlan& expected) {
    EXPECT_EQ(actual.num_slices, expected.num_slices);
    EXPECT_EQ(actual.slice_tiles, expected.slice_tiles);
    EXPECT_EQ(actual.num_groups_1, expected.num_groups_1);
    EXPECT_EQ(actual.rows_per_group_1, expected.rows_per_group_1);
    EXPECT_EQ(actual.rows_per_group_2, expected.rows_per_group_2);
    EXPECT_EQ(actual.num_cores, expected.num_cores);
    EXPECT_EQ(actual.num_cores_y, expected.num_cores_y);
}

struct PlanCase {
    CoreCoord grid;
    uint32_t total_rows;
    uint32_t Wt;
    RowSlicePlan expected;
};

}  // namespace

TEST(RMSNormPlannerTest, ZeroRowsGetNoCores) {
    for (const auto& grid : kGrids) {
        SCOPED_TRACE(describe(grid, 0U, 160U));
        expect_plan_eq(
            plan(grid, 0U, 160U),
            RowSlicePlan{
                .num_slices = 1U,
                .slice_tiles = 160U,
                .num_groups_1 = 0U,
                .rows_per_group_1 = 0U,
                .rows_per_group_2 = 0U,
                .num_cores = 0U,
                .num_cores_y = static_cast<uint32_t>(grid.y)});
    }
}

// No slice count divides 37, so the last slice is shorter: 8 slices of St = ceil(37 / 8) = 5 cost 5 + 12 + 5 = 22,
// tied with 10 slices of 4 (4 + 14 + 4), against 74 for the whole row.
TEST(RMSNormPlannerTest, SliceCountsNeedNotDivideWt) {
    for (const auto& grid : kGrids) {
        SCOPED_TRACE(describe(grid, 1U, 37U));
        const auto p = plan(grid, 1U, 37U);
        EXPECT_EQ(p.num_slices, 8U);
        EXPECT_EQ(p.slice_tiles, 5U);
        EXPECT_EQ(p.num_cores, 8U);
    }
}

// A count whose ceil(Wt / S)-tile slices cover Wt before the last one is skipped: with Wt = 10, S = 6 to 9 give slices
// of 2, and 5 of them already cover the row.
TEST(RMSNormPlannerTest, SliceCountsWithAnEmptyLastSliceAreSkipped) {
    for (const auto& grid : kGrids) {
        SCOPED_TRACE(describe(grid, 1U, 10U));
        std::vector<std::pair<uint32_t, uint32_t>> fit_checks;
        plan_row_slices(grid, 1U, 10U, kMaxRowSlices, [&](const uint32_t slice_tiles, const uint32_t num_slices) {
            fit_checks.emplace_back(slice_tiles, num_slices);
            return true;
        });
        const std::vector<std::pair<uint32_t, uint32_t>> expected_checks = {
            {5U, 2U}, {4U, 3U}, {3U, 4U}, {2U, 5U}, {1U, 10U}};
        EXPECT_EQ(fit_checks, expected_checks);
    }
}

// Every plan covers its row, and its last slice, which the kernels move back to end at Wt, keeps a tile of its own.
TEST(RMSNormPlannerTest, SlicesCoverTheRowAndTheLastKeepsATile) {
    for (const auto& grid : kGrids) {
        for (const uint32_t total_rows : {1U, 32U, 128U}) {
            for (uint32_t Wt = 1U; Wt <= 300U; ++Wt) {
                SCOPED_TRACE(describe(grid, total_rows, Wt));
                const auto p = plan(grid, total_rows, Wt);
                EXPECT_GE(p.num_slices * p.slice_tiles, Wt);
                EXPECT_LT((p.num_slices - 1U) * p.slice_tiles, Wt);
            }
        }
    }
}

// One row of 512 tiles takes 16 slices when they fit (cost 32 + 16 + 4 + 32 = 84 against 1024 for whole rows). The
// predicate sees every count with its slice width, and rejecting them all falls back to whole rows.
TEST(RMSNormPlannerTest, SlicesThatDontFitFallBackToWholeRows) {
    for (const auto& grid : kGrids) {
        SCOPED_TRACE(describe(grid, 1U, 512U));
        EXPECT_EQ(plan(grid, 1U, 512U).num_slices, 16U);

        std::vector<std::pair<uint32_t, uint32_t>> fit_checks;
        const auto p =
            plan_row_slices(grid, 1U, 512U, kMaxRowSlices, [&](const uint32_t slice_tiles, const uint32_t num_slices) {
                fit_checks.emplace_back(slice_tiles, num_slices);
                return false;
            });
        expect_plan_eq(
            p,
            RowSlicePlan{
                .num_slices = 1U,
                .slice_tiles = 512U,
                .num_groups_1 = 1U,
                .rows_per_group_1 = 1U,
                .rows_per_group_2 = 1U,
                .num_cores = 1U,
                .num_cores_y = static_cast<uint32_t>(grid.y)});
        const std::vector<std::pair<uint32_t, uint32_t>> expected_checks = {
            {256U, 2U},
            {171U, 3U},
            {128U, 4U},
            {103U, 5U},
            {86U, 6U},
            {74U, 7U},
            {64U, 8U},
            {57U, 9U},
            {52U, 10U},
            {47U, 11U},
            {43U, 12U},
            {40U, 13U},
            {37U, 14U},
            {35U, 15U},
            {32U, 16U}};
        EXPECT_EQ(fit_checks, expected_checks);
    }
}

// max_slices caps the row above, which takes 16 slices uncapped: 1 keeps whole rows, and 4 gives 4 slices of 128 (cost
// 128 + 8 + 128 = 264, against 349 for 3). Counts above the cap never reach the fit check.
TEST(RMSNormPlannerTest, MaxSlicesCapsTheSliceCount) {
    for (const auto& grid : kGrids) {
        for (const auto& [max_slices, num_slices, num_checks] :
             {std::tuple{1U, 1U, size_t{0U}}, std::tuple{4U, 4U, size_t{3U}}}) {
            SCOPED_TRACE(describe(grid, 1U, 512U) + ", max_slices = " + std::to_string(max_slices));
            size_t fit_checks = 0U;
            const auto p = plan_row_slices(grid, 1U, 512U, max_slices, [&](uint32_t, uint32_t) {
                ++fit_checks;
                return true;
            });
            EXPECT_EQ(p.num_slices, num_slices);
            EXPECT_EQ(p.slice_tiles, 512U / num_slices);
            EXPECT_EQ(p.num_cores, num_slices);
            EXPECT_EQ(fit_checks, num_checks);
        }
    }
}

// Few rows leave cores idle, so slicing wins; with many rows every core already has whole rows to work on, and the
// exchange makes slicing a loss. Each cost ends with the core's one read of its slice of gamma.
TEST(RMSNormPlannerTest, FewRowsSliceAndManyRowsKeepWholeRows) {
    struct Case {
        CoreCoord grid;
        uint32_t total_rows;
        uint32_t Wt;
        uint32_t num_slices;
        uint32_t slice_tiles;
    };
    const std::vector<Case> cases = {
        // 110 cores, Wt = 160 (C = 5120). 32 rows: S = 6 slices of 27 make 18 groups, 2 * (27 + 10) + 27 = 101;
        // S = 10 costs 3 * (16 + 14) + 16 = 106, S = 5 114 and whole rows 160 + 160 = 320.
        {kGrid11x10, 32U, 160U, 6U, 27U},
        // 128 rows: S = 5 makes 22 groups, 6 * 41 + 32 = 278; S = 4 costs 5 * (40 + 8) + 40 = 280 and whole rows
        // 2 * 160 + 160 = 480.
        {kGrid11x10, 128U, 160U, 5U, 32U},
        // 2048 rows: whole rows take 19 rounds, 19 * 160 + 160 = 3200; S = 2 costs 38 * 86 + 80 = 3348.
        {kGrid11x10, 2048U, 160U, 1U, 160U},
        // Wt = 224 (C = 7168), 32 rows: S = 6 slices of 38 make 18 groups, 2 * (38 + 10) + 38 = 134, tied with S = 10;
        // S = 8 costs 3 * (28 + 12) + 28 = 148 and S = 2 112 + 6 + 112 = 230.
        {kGrid11x10, 32U, 224U, 6U, 38U},
        // 130 cores, Wt = 16, 128 rows: whole rows take one round, 16 + 16 = 32; S = 2 costs 2 * (8 + 6) + 8 = 36.
        {kGrid13x10, 128U, 16U, 1U, 16U},
        // 64 cores, 37 rows of 77 tiles: S = 3 slices of 26 make 21 groups, 2 * (26 + 7) + 26 = 92; S = 4 costs
        // 3 * (20 + 8) + 20 = 104 and whole rows 77 + 77 = 154.
        {kGrid8x8, 37U, 77U, 3U, 26U},
    };
    for (const auto& c : cases) {
        SCOPED_TRACE(describe(c.grid, c.total_rows, c.Wt));
        const auto p = plan(c.grid, c.total_rows, c.Wt);
        EXPECT_EQ(p.num_slices, c.num_slices);
        EXPECT_EQ(p.slice_tiles, c.slice_tiles);
    }
}

// Ties keep the smaller slice count: it uses fewer cores and exchanges less.
TEST(RMSNormPlannerTest, TiesKeepTheSmallerSliceCount) {
    for (const auto& grid : kGrids) {
        // One row of 12 tiles: S = 4 and S = 6 both cost 14 (3 + 8 + 3 and 2 + 10 + 2); S = 3 costs 15.
        {
            SCOPED_TRACE(describe(grid, 1U, 12U));
            EXPECT_EQ(plan(grid, 1U, 12U).num_slices, 4U);
        }
        // One row of 24 tiles: S = 6 and S = 8 both cost 18 (4 + 10 + 4 and 3 + 12 + 3); S = 5 costs 19.
        {
            SCOPED_TRACE(describe(grid, 1U, 24U));
            EXPECT_EQ(plan(grid, 1U, 24U).num_slices, 6U);
        }
    }
}

// Rows that don't divide over the row-groups: the first num_groups_1 groups take one row more.
TEST(RMSNormPlannerTest, UnevenSplitsGiveTwoGroupSizes) {
    const std::vector<PlanCase> cases = {
        // The Plain_1x1184x1x2463 device test: 37 rows of 77 tiles in 5 slices of 16, the last 13 tiles of its own.
        // 110 / 5 = 22 groups, 37 = 15 * 2 + 7 * 1.
        {kGrid11x10,
         37U,
         77U,
         {.num_slices = 5U,
          .slice_tiles = 16U,
          .num_groups_1 = 15U,
          .rows_per_group_1 = 2U,
          .rows_per_group_2 = 1U,
          .num_cores = 110U,
          .num_cores_y = 10U}},
        // The Plain_1x4192x1x2368 device test: 131 rows of 74 tiles in 3 slices of 25, the last 24 of its own.
        // 43 groups, 131 = 2 * 4 + 41 * 3.
        {kGrid13x10,
         131U,
         74U,
         {.num_slices = 3U,
          .slice_tiles = 25U,
          .num_groups_1 = 2U,
          .rows_per_group_1 = 4U,
          .rows_per_group_2 = 3U,
          .num_cores = 129U,
          .num_cores_y = 10U}},
        // 36 groups, 131 = 23 * 4 + 13 * 3.
        {kGrid11x10,
         131U,
         74U,
         {.num_slices = 3U,
          .slice_tiles = 25U,
          .num_groups_1 = 23U,
          .rows_per_group_1 = 4U,
          .rows_per_group_2 = 3U,
          .num_cores = 108U,
          .num_cores_y = 10U}},
        // 21 groups, 131 = 5 * 7 + 16 * 6.
        {kGrid8x8,
         131U,
         74U,
         {.num_slices = 3U,
          .slice_tiles = 25U,
          .num_groups_1 = 5U,
          .rows_per_group_1 = 7U,
          .rows_per_group_2 = 6U,
          .num_cores = 63U,
          .num_cores_y = 8U}},
        // Wt = 128: 110 / 4 = 27 groups, 128 = 20 * 5 + 7 * 4, so two cores of the last column stay idle.
        {kGrid11x10,
         128U,
         128U,
         {.num_slices = 4U,
          .slice_tiles = 32U,
          .num_groups_1 = 20U,
          .rows_per_group_1 = 5U,
          .rows_per_group_2 = 4U,
          .num_cores = 108U,
          .num_cores_y = 10U}},
        // Whole rows: 2048 = 68 * 19 + 42 * 18.
        {kGrid11x10,
         2048U,
         160U,
         {.num_slices = 1U,
          .slice_tiles = 160U,
          .num_groups_1 = 68U,
          .rows_per_group_1 = 19U,
          .rows_per_group_2 = 18U,
          .num_cores = 110U,
          .num_cores_y = 10U}},
        // Even: 32 rows in 16 groups of 8 slices, one core group.
        {kGrid13x10,
         32U,
         160U,
         {.num_slices = 8U,
          .slice_tiles = 20U,
          .num_groups_1 = 16U,
          .rows_per_group_1 = 2U,
          .rows_per_group_2 = 2U,
          .num_cores = 128U,
          .num_cores_y = 10U}},
    };
    for (const auto& c : cases) {
        SCOPED_TRACE(describe(c.grid, c.total_rows, c.Wt));
        const auto p = plan(c.grid, c.total_rows, c.Wt);
        expect_plan_eq(p, c.expected);
        EXPECT_EQ(p.has_group_2(), c.expected.rows_per_group_1 != c.expected.rows_per_group_2);
        const uint32_t num_groups = p.num_cores / p.num_slices;
        EXPECT_EQ(
            p.num_groups_1 * p.rows_per_group_1 + (num_groups - p.num_groups_1) * p.rows_per_group_2, c.total_rows);
    }
}

// row_slice_cores() must give the cores for_each_row_slice_core() walks, each in the core group of its row-group, and
// the walk must hand out every row exactly once.
TEST(RMSNormPlannerTest, CoreRangesMatchTheWalk) {
    const std::vector<std::pair<uint32_t, uint32_t>> shapes = {
        {37U, 77U}, {131U, 74U}, {128U, 160U}, {512U, 160U}, {32U, 160U}, {1U, 512U}, {1U, 27U}, {100U, 1U}};
    for (const auto& grid : kGrids) {
        for (const auto& [total_rows, Wt] : shapes) {
            SCOPED_TRACE(describe(grid, total_rows, Wt));
            const auto p = plan(grid, total_rows, Wt);
            const auto cores = row_slice_cores(p, grid);

            uint32_t num_cores_1 = 0U;
            uint32_t num_cores_2 = 0U;
            uint32_t next_row = 0U;
            for_each_row_slice_core(p, [&](const RowSliceWork& work) {
                const bool in_group_1 = work.group < p.num_groups_1;
                EXPECT_TRUE(cores.all_cores.contains(work.core)) << work.core.str();
                EXPECT_EQ(cores.core_group_1.contains(work.core), in_group_1) << work.core.str();
                EXPECT_EQ(cores.core_group_2.contains(work.core), !in_group_1) << work.core.str();
                EXPECT_EQ(work.num_rows, in_group_1 ? p.rows_per_group_1 : p.rows_per_group_2);
                ++(in_group_1 ? num_cores_1 : num_cores_2);
                if (work.slice == 0U) {
                    EXPECT_EQ(work.start_row, next_row) << "group " << work.group;
                    next_row += work.num_rows;
                }
            });

            EXPECT_EQ(num_cores_1 + num_cores_2, p.num_cores);
            EXPECT_EQ(cores.all_cores.num_cores(), p.num_cores);
            EXPECT_EQ(cores.core_group_1.num_cores(), num_cores_1);
            EXPECT_EQ(cores.core_group_2.num_cores(), num_cores_2);
            EXPECT_EQ(next_row, total_rows);
        }
    }
}

// A row-group stays in one column of 10 cores, so its exchange can multicast, unless it wraps into the next column.
TEST(RMSNormPlannerTest, RowGroupsInOneColumn) {
    const auto groups_in_one_column = [](const uint32_t num_slices, const uint32_t num_groups) {
        const RowSlicePlan p{.num_slices = num_slices, .num_cores_y = 10U};
        std::vector<bool> in_one_column;
        in_one_column.reserve(num_groups);
        for (uint32_t group = 0; group < num_groups; ++group) {
            in_one_column.push_back(row_group_in_one_column(p, group));
        }
        return in_one_column;
    };
    // Slice counts that divide the column height never wrap.
    for (const uint32_t num_slices : {1U, 2U, 5U, 10U}) {
        EXPECT_EQ(groups_in_one_column(num_slices, 130U / num_slices), std::vector<bool>(130U / num_slices, true))
            << num_slices << " slices";
    }
    // 4 slices: groups 2 (cores 8-11) and 7 (cores 28-31) wrap.
    const std::vector<bool> four_slices = {true, true, false, true, true, true, true, false, true, true};
    EXPECT_EQ(groups_in_one_column(4U, 10U), four_slices);
    // 16 slices are taller than the column.
    EXPECT_EQ(groups_in_one_column(16U, 8U), std::vector<bool>(8U, false));
}

// row_group_in_one_column() agrees with the cores for_each_row_slice_core() gives each row-group.
TEST(RMSNormPlannerTest, RowGroupInOneColumnMatchesTheWalk) {
    const std::vector<std::pair<uint32_t, uint32_t>> shapes = {{32U, 160U}, {128U, 160U}, {37U, 77U}, {1U, 512U}};
    for (const auto& grid : kGrids) {
        for (const auto& [total_rows, Wt] : shapes) {
            SCOPED_TRACE(describe(grid, total_rows, Wt));
            const auto p = plan(grid, total_rows, Wt);
            std::map<uint32_t, std::vector<CoreCoord>> group_cores;
            for_each_row_slice_core(
                p, [&](const RowSliceWork& work) { group_cores[work.group].push_back(work.core); });
            for (const auto& [group, cores] : group_cores) {
                const bool same_x = std::all_of(
                    cores.begin(), cores.end(), [&](const CoreCoord& core) { return core.x == cores.front().x; });
                EXPECT_EQ(row_group_in_one_column(p, group), same_x) << "group " << group;
            }
        }
    }
}
