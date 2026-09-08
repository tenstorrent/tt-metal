// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "ttnn/operations/core/program_cache_l1.hpp"

namespace {

using ttnn::operations::core::available_program_l1_capacity;

TEST(ProgramCacheL1, OccupancyBelowBaseDoesNotWrap) {
    EXPECT_EQ(available_program_l1_capacity(0, 4096), 0);
    EXPECT_EQ(available_program_l1_capacity(4095, 4096), 0);
}

TEST(ProgramCacheL1, OccupancyAtBaseLeavesNoCapacity) {
    EXPECT_EQ(available_program_l1_capacity(4096, 4096), 0);
    EXPECT_EQ(available_program_l1_capacity(0, 0), 0);
}

TEST(ProgramCacheL1, OccupancyAboveBaseReturnsExactSpan) {
    EXPECT_EQ(available_program_l1_capacity(4097, 4096), 1);
    EXPECT_EQ(available_program_l1_capacity(1'500'000, 4096), 1'495'904);
}

}  // namespace
