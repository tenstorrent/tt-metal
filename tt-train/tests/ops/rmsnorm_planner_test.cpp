// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-side tests for plan_row_slices, the split of RMSNorm tile-rows over the cores. It is plain arithmetic, so these
// run without a device and pin its answers on several grids, whatever grid the test machine has. Each pinned answer
// follows from the cost the planner minimizes:
//   cost(S) = ceil(rows / min(rows, cores / S)) * (Wt / S + (S > 1 ? S + kRowSliceExchangeCostTiles : 0)).

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
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

// 37 is prime and above kMaxRowSlices, so no slice count divides it. A single row would otherwise be sliced.
TEST(RMSNormPlannerTest, PrimeWtKeepsWholeRows) {
    for (const auto& grid : kGrids) {
        SCOPED_TRACE(describe(grid, 1U, 37U));
        uint32_t fit_checks = 0U;
        const auto p = plan_row_slices(grid, 1U, 37U, kMaxRowSlices, [&](uint32_t, uint32_t) {
            ++fit_checks;
            return true;
        });
        EXPECT_EQ(p.num_slices, 1U);
        EXPECT_EQ(p.slice_tiles, 37U);
        EXPECT_EQ(p.num_cores, 1U);
        EXPECT_EQ(fit_checks, 0U);
    }
}

// One row of 512 tiles takes 16 slices when they fit (cost 32 + 16 + 4 = 52 against 512 for whole rows). The predicate
// sees every dividing count with its slice width, and rejecting them all falls back to whole rows.
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
            {256U, 2U}, {128U, 4U}, {64U, 8U}, {32U, 16U}};
        EXPECT_EQ(fit_checks, expected_checks);
    }
}

// Few rows leave cores idle, so slicing wins; with many rows every core already has whole rows to work on, and the
// exchange makes slicing a loss.
TEST(RMSNormPlannerTest, FewRowsSliceAndManyRowsKeepWholeRows) {
    struct Case {
        CoreCoord grid;
        uint32_t total_rows;
        uint32_t Wt;
        uint32_t num_slices;
    };
    const std::vector<Case> cases = {
        // 110 cores, Wt = 160 (C = 5120). 32 rows: S = 5 makes 22 groups, 2 rounds of 32 + 5 + 4 = 82; S = 2 costs
        // 80 + 6 = 86 and whole rows 160.
        {kGrid11x10, 32U, 160U, 5U},
        // 128 rows: S = 4 makes 27 groups, 5 rounds of 40 + 8 = 240; S = 5 costs 6 * 41 = 246, S = 2 3 * 86 = 258 and
        // whole rows 2 * 160 = 320.
        {kGrid11x10, 128U, 160U, 4U},
        // 512 rows: whole rows take 5 rounds of 160 = 800; S = 2 costs 10 * 86 = 860.
        {kGrid11x10, 512U, 160U, 1U},
        // 130 cores, Wt = 128, 128 rows: whole rows take one round of 128; S = 2 costs 2 * (64 + 6) = 140.
        {kGrid13x10, 128U, 128U, 1U},
        // 64 cores, 37 rows of 77 tiles: whole rows cost 77; S = 7 makes 9 groups, 5 rounds of 11 + 11 = 110, and
        // S = 11 makes 5 groups, 8 rounds of 7 + 15 = 176.
        {kGrid8x8, 37U, 77U, 1U},
    };
    for (const auto& c : cases) {
        SCOPED_TRACE(describe(c.grid, c.total_rows, c.Wt));
        const auto p = plan(c.grid, c.total_rows, c.Wt);
        EXPECT_EQ(p.num_slices, c.num_slices);
        EXPECT_EQ(p.slice_tiles, c.Wt / c.num_slices);
    }
}

// Ties keep the smaller slice count: it uses fewer cores and exchanges less.
TEST(RMSNormPlannerTest, TiesKeepTheSmallerSliceCount) {
    for (const auto& grid : kGrids) {
        // One row of 32 tiles: S = 4 and S = 8 both cost 16 (8 + 8 and 4 + 12); S = 2 and S = 16 cost 22.
        {
            SCOPED_TRACE(describe(grid, 1U, 32U));
            EXPECT_EQ(plan(grid, 1U, 32U).num_slices, 4U);
        }
        // One row of 27 tiles: S = 3 and S = 9 both cost 16 (9 + 7 and 3 + 13).
        {
            SCOPED_TRACE(describe(grid, 1U, 27U));
            EXPECT_EQ(plan(grid, 1U, 27U).num_slices, 3U);
        }
    }
}

// Rows that don't divide over the row-groups: the first num_groups_1 groups take one row more.
TEST(RMSNormPlannerTest, UnevenSplitsGiveTwoGroupSizes) {
    const std::vector<PlanCase> cases = {
        // The Plain_1x1184x1x2463 device test: 37 rows of 77 tiles, 7 slices. 130 / 7 = 18 groups, 37 = 1 * 3 + 17 * 2.
        {kGrid13x10,
         37U,
         77U,
         {.num_slices = 7U,
          .slice_tiles = 11U,
          .num_groups_1 = 1U,
          .rows_per_group_1 = 3U,
          .rows_per_group_2 = 2U,
          .num_cores = 126U,
          .num_cores_y = 10U}},
        // 110 / 7 = 15 groups, 37 = 7 * 3 + 8 * 2.
        {kGrid11x10,
         37U,
         77U,
         {.num_slices = 7U,
          .slice_tiles = 11U,
          .num_groups_1 = 7U,
          .rows_per_group_1 = 3U,
          .rows_per_group_2 = 2U,
          .num_cores = 105U,
          .num_cores_y = 10U}},
        // The Plain_1x4192x1x2368 device test, the 2-slice split model shapes get when they slice: 131 rows of 74 =
        // 2 * 37 tiles. Slicing costs 3 * (37 + 6) = 129 against 2 * 74 = 148. 65 groups, 131 = 1 * 3 + 64 * 2.
        {kGrid13x10,
         131U,
         74U,
         {.num_slices = 2U,
          .slice_tiles = 37U,
          .num_groups_1 = 1U,
          .rows_per_group_1 = 3U,
          .rows_per_group_2 = 2U,
          .num_cores = 130U,
          .num_cores_y = 10U}},
        // 55 groups, 131 = 21 * 3 + 34 * 2.
        {kGrid11x10,
         131U,
         74U,
         {.num_slices = 2U,
          .slice_tiles = 37U,
          .num_groups_1 = 21U,
          .rows_per_group_1 = 3U,
          .rows_per_group_2 = 2U,
          .num_cores = 110U,
          .num_cores_y = 10U}},
        // 5 * (37 + 6) = 215 against 3 * 74 = 222. 32 groups, 131 = 3 * 5 + 29 * 4.
        {kGrid8x8,
         131U,
         74U,
         {.num_slices = 2U,
          .slice_tiles = 37U,
          .num_groups_1 = 3U,
          .rows_per_group_1 = 5U,
          .rows_per_group_2 = 4U,
          .num_cores = 64U,
          .num_cores_y = 8U}},
        // 110 / 4 = 27 groups, 128 = 20 * 5 + 7 * 4, so two cores of the last column stay idle.
        {kGrid11x10,
         128U,
         160U,
         {.num_slices = 4U,
          .slice_tiles = 40U,
          .num_groups_1 = 20U,
          .rows_per_group_1 = 5U,
          .rows_per_group_2 = 4U,
          .num_cores = 108U,
          .num_cores_y = 10U}},
        // Whole rows: 512 = 72 * 5 + 38 * 4.
        {kGrid11x10,
         512U,
         160U,
         {.num_slices = 1U,
          .slice_tiles = 160U,
          .num_groups_1 = 72U,
          .rows_per_group_1 = 5U,
          .rows_per_group_2 = 4U,
          .num_cores = 110U,
          .num_cores_y = 10U}},
        // Even: 32 rows in 32 groups of 4 slices, one core group.
        {kGrid13x10,
         32U,
         160U,
         {.num_slices = 4U,
          .slice_tiles = 40U,
          .num_groups_1 = 32U,
          .rows_per_group_1 = 1U,
          .rows_per_group_2 = 1U,
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
