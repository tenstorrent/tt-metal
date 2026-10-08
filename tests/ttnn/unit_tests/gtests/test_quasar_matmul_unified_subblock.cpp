// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-side checks of the Quasar-native matmul's auto subblock choice (no device needed).

#include <cstdint>
#include <utility>

#include "gtest/gtest.h"
#include "ttnn/operations/experimental/quasar/matmul/device/factory/matmul_unified_program_factory.hpp"

namespace {

using ttnn::prim::qsr::detail::maximize_subblock_size;
using Subblock = std::pair<uint32_t, uint32_t>;

constexpr auto any_subblock = [](uint32_t, uint32_t) { return true; };

Subblock choose(uint32_t C_slice_M_tiles, uint32_t C_slice_N_tiles, uint32_t dst_tiles, uint32_t threads) {
    return maximize_subblock_size(C_slice_M_tiles, C_slice_N_tiles, dst_tiles, threads, any_subblock);
}

// One thread: the largest subblock, least padding on ties.
TEST(QuasarMatmulUnifiedSubblock, OneThreadTakesTheLargestThenTheLeastPadded) {
    EXPECT_EQ(choose(1, 1, 8, 1), Subblock(1, 8));  // pads the single tile to a full DST
    EXPECT_EQ(choose(3, 3, 8, 1), Subblock(2, 4));  // 4x4 padded beats 1x8's 3x8
    EXPECT_EQ(choose(4, 4, 8, 1), Subblock(2, 4));
    EXPECT_EQ(choose(4, 4, 4, 1), Subblock(1, 4));  // fp32 accumulation halves DST
}

// Several threads: the least work on the busiest thread first, then the largest subblock.
TEST(QuasarMatmulUnifiedSubblock, SeveralThreadsBalanceBeforeVolume) {
    EXPECT_EQ(choose(1, 1, 8, 4), Subblock(1, 1));  // a padded 1x8 would be all one thread's work
    EXPECT_EQ(choose(4, 4, 8, 4), Subblock(1, 4));  // 2x4 leaves two of four threads idle
    EXPECT_EQ(choose(4, 4, 8, 2), Subblock(2, 4));  // two 2x4 subblocks already balance two threads
}

TEST(QuasarMatmulUnifiedSubblock, OnlyAcceptedShapesAreChosen) {
    const auto single_column = [](uint32_t, uint32_t subblock_N_tiles) { return subblock_N_tiles == 1; };
    EXPECT_EQ(maximize_subblock_size(4, 4, 8, 1, single_column), Subblock(8, 1));
    const auto nothing = [](uint32_t, uint32_t) { return false; };
    EXPECT_EQ(maximize_subblock_size(4, 4, 8, 4, nothing), Subblock(1, 1));
}

}  // namespace
