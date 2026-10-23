// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

#include "ttnn/operations/core/work_split/work_split_tilize.hpp"

namespace ttnn {
namespace {

namespace work_split = operations::core::work_split;

TEST(WorkSplitTilizeTest, TryComputeRejectsInvalidBlockSizeLimit) {
    EXPECT_FALSE(
        work_split::try_compute_ncores_wh_sb(/*grid_area=*/1, /*nblocks=*/1, /*width_tiles=*/1, /*height_tiles=*/1, 0));
}

TEST(WorkSplitTilizeTest, TryComputeReturnsNulloptWhenNoSplitExists) {
    EXPECT_FALSE(
        work_split::try_compute_ncores_wh_sb(/*grid_area=*/1, /*nblocks=*/6, /*width_tiles=*/2, /*height_tiles=*/3, 2));
}

TEST(WorkSplitTilizeTest, ThrowingWrapperMatchesValidTryComputeResult) {
    constexpr size_t grid_area = 64;
    constexpr uint32_t nblocks = 64;
    constexpr uint32_t width_tiles = 8;
    constexpr uint32_t height_tiles = 8;
    constexpr uint32_t single_block_size_limit = 2;

    const auto maybe_result =
        work_split::try_compute_ncores_wh_sb(grid_area, nblocks, width_tiles, height_tiles, single_block_size_limit);
    ASSERT_TRUE(maybe_result.has_value());

    const auto result =
        work_split::compute_ncores_wh_sb(grid_area, nblocks, width_tiles, height_tiles, single_block_size_limit);
    EXPECT_EQ(result.ncores, maybe_result->ncores);
    EXPECT_EQ(result.nblocks_per_core, maybe_result->nblocks_per_core);
    EXPECT_EQ(result.total_blocks_width, maybe_result->total_blocks_width);
    EXPECT_EQ(result.total_blocks_height, maybe_result->total_blocks_height);
    EXPECT_EQ(result.single_block_size, maybe_result->single_block_size);
    EXPECT_EQ(result.single_sub_block_size, maybe_result->single_sub_block_size);
}

TEST(WorkSplitTilizeTest, ThrowingWrapperPreservesInvalidLimitDiagnostic) {
    if (std::getenv("TT_ASSERT_ABORT") != nullptr) {
        GTEST_SKIP() << "TT_FATAL aborts instead of throwing when TT_ASSERT_ABORT is set";
    }
    try {
        (void)work_split::compute_ncores_wh_sb(
            /*grid_area=*/1, /*nblocks=*/1, /*width_tiles=*/1, /*height_tiles=*/1, 0);
        FAIL() << "Expected compute_ncores_wh_sb to reject a zero block size limit";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(
            std::string_view(error.what()).find("single_block_size_limit must be at least 1"), std::string_view::npos);
    }
}

}  // namespace
}  // namespace ttnn
