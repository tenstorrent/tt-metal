// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <set>
#include <utility>

#include "gtest/gtest.h"
#include <tt-metalium/core_coord.hpp>
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

namespace ttnn::experimental::prim::kda_factory_detail::test {

using tt::tt_metal::CoreCoord;

struct DistributionCase {
    CoreCoord grid;
    uint32_t batch_heads;
    uint32_t value_tiles;
    uint32_t expected_blocks;
};

class ValueBlockDistributionTest : public testing::TestWithParam<DistributionCase> {};

TEST_P(ValueBlockDistributionTest, PlacesEveryHeadsBlocksInOneRow) {
    const auto& param = GetParam();
    const auto distribution = distribute_value_blocks(param.grid, param.batch_heads, param.value_tiles);

    EXPECT_EQ(distribution.value_blocks, param.expected_blocks);
    EXPECT_EQ(distribution.value_tiles_per_core * distribution.value_blocks, param.value_tiles);
    const size_t cores = static_cast<size_t>(param.batch_heads) * distribution.value_blocks;
    ASSERT_EQ(distribution.cores.size(), cores);
    ASSERT_EQ(distribution.head.size(), cores);
    ASSERT_EQ(distribution.value_block.size(), cores);
    EXPECT_EQ(distribution.core_set.num_cores(), cores);

    std::set<std::pair<size_t, size_t>> seen;
    for (size_t index = 0; index < cores; ++index) {
        const auto& core = distribution.cores[index];
        EXPECT_LT(core.x, param.grid.x);
        EXPECT_LT(core.y, param.grid.y);
        EXPECT_TRUE(distribution.core_set.contains(core));
        EXPECT_TRUE(seen.insert({core.x, core.y}).second) << "core " << core.str() << " assigned twice";
        // Blocks of a head are consecutive, start at block 0 and sit side by side in one row, so value block 0 can
        // multicast to the rest over a single row segment.
        EXPECT_EQ(distribution.head[index], index / distribution.value_blocks);
        EXPECT_EQ(distribution.value_block[index], index % distribution.value_blocks);
        if (distribution.value_block[index] != 0) {
            const auto& previous = distribution.cores[index - 1];
            EXPECT_EQ(core.y, previous.y);
            EXPECT_EQ(core.x, previous.x + 1);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    KdaFactory,
    ValueBlockDistributionTest,
    testing::Values(
        // Galaxy SP8xTP4: 24 heads with 4 value tiles fill 8 rows of a 12-wide grid.
        DistributionCase{CoreCoord{12, 10}, 24, 4, 4},
        // A width that is not a multiple of the block count: 3 heads per 13-wide row.
        DistributionCase{CoreCoord{13, 10}, 24, 4, 4},
        // Too few cores for 4 blocks per head: falls back to 2.
        DistributionCase{CoreCoord{8, 8}, 24, 4, 2},
        // A prime value-tile count has no divisor that fits.
        DistributionCase{CoreCoord{12, 10}, 24, 5, 1},
        // Heads already fill the grid.
        DistributionCase{CoreCoord{12, 10}, 120, 4, 1},
        // A partial last row: 25 heads at 3 per row leave one head in the ninth row.
        DistributionCase{CoreCoord{12, 10}, 25, 4, 4}));

}  // namespace ttnn::experimental::prim::kda_factory_detail::test
