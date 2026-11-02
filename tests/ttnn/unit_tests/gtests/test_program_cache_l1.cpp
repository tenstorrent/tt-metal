// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "ttnn/operations/core/program_cache_l1.hpp"

namespace {

using ttnn::operations::core::available_program_l1_capacity;

TEST(ProgramCacheL1, OccupancyBelowBaseDoesNotWrap) {
    EXPECT_EQ(available_program_l1_capacity(0, 4096, 1'500'000), 0);
    EXPECT_EQ(available_program_l1_capacity(4095, 4096, 1'500'000), 0);
}

TEST(ProgramCacheL1, OccupancyAtBaseLeavesNoCapacity) {
    EXPECT_EQ(available_program_l1_capacity(4096, 4096, 1'500'000), 0);
    EXPECT_EQ(available_program_l1_capacity(0, 0, 1'500'000), 0);
}

TEST(ProgramCacheL1, OccupancyAboveBaseReturnsExactSpan) {
    EXPECT_EQ(available_program_l1_capacity(4097, 4096, 1'500'000), 1);
    EXPECT_EQ(available_program_l1_capacity(1'500'000, 4096, 1'500'000), 1'495'904);
}

TEST(ProgramCacheL1, ReservedRegionCapsCapacityWithoutOrdinaryL1Occupancy) {
    constexpr std::uint64_t physical_l1_size = 1536 * 1024;
    constexpr std::uint64_t cb_l1_base = 128 * 1024;
    constexpr std::uint64_t reserved_l1_small = 1024 * 1024;
    constexpr std::uint64_t bank_capacity = physical_l1_size - cb_l1_base - reserved_l1_small;
    EXPECT_EQ(available_program_l1_capacity(physical_l1_size, cb_l1_base, bank_capacity), bank_capacity);
    EXPECT_EQ(available_program_l1_capacity(cb_l1_base + 1, cb_l1_base, bank_capacity), 1);
    EXPECT_EQ(available_program_l1_capacity(physical_l1_size, cb_l1_base, 0), 0);
}

}  // namespace
