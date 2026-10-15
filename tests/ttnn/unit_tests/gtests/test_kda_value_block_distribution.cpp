// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <set>
#include <utility>

#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

namespace {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using ttnn::experimental::prim::kda_factory_detail::distribute_value_blocks;
using ttnn::experimental::prim::kda_factory_detail::ValueBlockDistribution;

// Every (head, block) pair gets one distinct core inside the grid, in head-major order, and the core set covers
// exactly those cores.
void expect_consistent(const ValueBlockDistribution& distribution, CoreCoord grid, uint32_t batch_heads) {
    const uint32_t count = batch_heads * distribution.value_blocks;
    ASSERT_EQ(distribution.cores.size(), count);
    ASSERT_EQ(distribution.head.size(), count);
    ASSERT_EQ(distribution.value_block.size(), count);
    std::set<std::pair<std::size_t, std::size_t>> seen;
    for (uint32_t index = 0; index < count; ++index) {
        const CoreCoord core = distribution.cores[index];
        EXPECT_LT(core.x, grid.x);
        EXPECT_LT(core.y, grid.y);
        EXPECT_TRUE(seen.insert({core.x, core.y}).second) << "core " << core.str() << " assigned twice";
        EXPECT_EQ(distribution.head[index], index / distribution.value_blocks);
        EXPECT_EQ(distribution.value_block[index], index % distribution.value_blocks);
        EXPECT_TRUE(distribution.core_set.contains(core));
    }
    EXPECT_EQ(distribution.core_set.num_cores(), count);
}

// Kimi-K3 on a Blackhole Galaxy chip: 24 TP-local heads of 4 value tiles on the 12x10 grid. Both directions allow
// four blocks, so each head's blocks share one column and the heads fill 12 columns of 8 rows.
TEST(KdaValueBlockDistribution, KimiK3StacksEachHeadInOneColumn) {
    const CoreCoord grid{12, 10};
    const auto distribution = distribute_value_blocks(grid, 24, 4);
    expect_consistent(distribution, grid, 24);
    EXPECT_EQ(distribution.value_blocks, 4u);
    EXPECT_EQ(distribution.value_tiles_per_core, 1u);
    for (uint32_t head = 0; head < 24; ++head) {
        for (uint32_t block = 0; block < 4; ++block) {
            const CoreCoord core = distribution.cores[head * 4 + block];
            EXPECT_EQ(core.x, head / 2);
            EXPECT_EQ(core.y, (head % 2) * 4 + block);
        }
    }
    ASSERT_EQ(distribution.core_set.ranges().size(), 1u);
    EXPECT_EQ(distribution.core_set.ranges()[0], CoreRange(CoreCoord{0, 0}, CoreCoord{11, 7}));
}

// When rows allow a larger split than columns, rows win: fewer value tiles per core keep the per-core L1 down.
// 28 heads of 8 value tiles: rows of 12 split each head four ways, columns of 10 only two ways.
TEST(KdaValueBlockDistribution, KeepsTheLargerRowSplit) {
    const CoreCoord grid{12, 10};
    const auto distribution = distribute_value_blocks(grid, 28, 8);
    expect_consistent(distribution, grid, 28);
    EXPECT_EQ(distribution.value_blocks, 4u);
    EXPECT_EQ(distribution.value_tiles_per_core, 2u);
    for (uint32_t head = 0; head < 28; ++head) {
        for (uint32_t block = 0; block < 4; ++block) {
            const CoreCoord core = distribution.cores[head * 4 + block];
            EXPECT_EQ(core.x, (head % 3) * 4 + block);
            EXPECT_EQ(core.y, head / 3);
        }
    }
}

// A head with one value tile cannot split: one core per head, filled row by row.
TEST(KdaValueBlockDistribution, UnsplitHeadsFillRows) {
    const CoreCoord grid{4, 4};
    const auto distribution = distribute_value_blocks(grid, 5, 1);
    expect_consistent(distribution, grid, 5);
    EXPECT_EQ(distribution.value_blocks, 1u);
    EXPECT_EQ(distribution.value_tiles_per_core, 1u);
    EXPECT_EQ(distribution.cores.back(), (CoreCoord{0, 1}));
}

// Value tiles that no line length divides beyond one also stay unsplit, whatever the head count.
TEST(KdaValueBlockDistribution, PrimeValueTilesLongerThanTheGridStayUnsplit) {
    const CoreCoord grid{8, 8};
    const auto distribution = distribute_value_blocks(grid, 10, 11);
    expect_consistent(distribution, grid, 10);
    EXPECT_EQ(distribution.value_blocks, 1u);
    EXPECT_EQ(distribution.value_tiles_per_core, 11u);
}

// A short partial last column: 5 heads of 4 tiles on a 4x8 grid split four ways in either direction, so the heads
// stack two per column over three columns, the last holding one.
TEST(KdaValueBlockDistribution, PartialLastColumn) {
    const CoreCoord grid{4, 8};
    const auto distribution = distribute_value_blocks(grid, 5, 4);
    expect_consistent(distribution, grid, 5);
    EXPECT_EQ(distribution.value_blocks, 4u);
    EXPECT_EQ(distribution.cores[4 * 4], (CoreCoord{2, 0}));
    EXPECT_EQ(distribution.cores.back(), (CoreCoord{2, 3}));
}

}  // namespace
