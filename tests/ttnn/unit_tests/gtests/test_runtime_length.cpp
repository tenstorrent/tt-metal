// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Boundaries of the shared host/device work-split helpers (ttnn/operations/common/shared_with_host/
// runtime_length.hpp). Host and kernels derive every length-dependent count from them, so they must cover
// the input exactly, without overlap, for zero work, fewer tiles than cores, uneven splits and partial
// blocks. Pure constexpr arithmetic: no device.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>

#include "ttnn/operations/common/shared_with_host/runtime_length.hpp"

namespace ttnn::runtime_length {
namespace {

// Every core's [start, end) in order; they must tile [0, total) exactly.
void expect_exact_cover(uint32_t total, uint32_t num_cores) {
    const uint32_t per_core = tiles_per_core(total, num_cores);
    uint32_t next = 0;
    for (uint32_t core = 0; core < num_cores; ++core) {
        EXPECT_EQ(core_start(core, per_core, total), next) << "total " << total << " cores " << num_cores;
        const uint32_t end = core_end(core, per_core, total);
        EXPECT_LE(end, total);
        EXPECT_GE(end, next);
        next = end;
    }
    EXPECT_EQ(next, total);
}

void expect_balanced_cover(uint32_t total, uint32_t num_cores) {
    uint32_t next = 0;
    uint32_t smallest = UINT32_MAX;
    uint32_t largest = 0;
    for (uint32_t core = 0; core < num_cores; ++core) {
        EXPECT_EQ(balanced_start(core, total, num_cores), next);
        const uint32_t end = balanced_end(core, total, num_cores);
        EXPECT_GE(end, next);
        smallest = std::min(smallest, end - next);
        largest = std::max(largest, end - next);
        next = end;
    }
    EXPECT_EQ(next, total);
    EXPECT_LE(largest - smallest, 1u) << "total " << total << " cores " << num_cores;
}

TEST(RuntimeLengthTest, DivUpAndMin) {
    static_assert(div_up(0, 4) == 0);
    static_assert(div_up(1, 4) == 1);
    static_assert(div_up(4, 4) == 1);
    static_assert(div_up(5, 4) == 2);
    static_assert(min_u32(3, 5) == 3);
    static_assert(min_u32(5, 3) == 3);
}

TEST(RuntimeLengthTest, ZeroWorkGivesEmptyRanges) {
    const uint32_t per_core = tiles_per_core(0, 8);
    EXPECT_EQ(per_core, 0u);
    for (uint32_t core = 0; core < 8; ++core) {
        EXPECT_EQ(core_start(core, per_core, 0), 0u);
        EXPECT_EQ(core_end(core, per_core, 0), 0u);
        EXPECT_EQ(balanced_start(core, 0, 8), 0u);
        EXPECT_EQ(balanced_end(core, 0, 8), 0u);
    }
    EXPECT_EQ(blocks_per_core(0, 4), 0u);
    EXPECT_EQ(padded_tiles(0, 8, 4), 0u);
    EXPECT_EQ(rounds(0, 8), 0u);
}

TEST(RuntimeLengthTest, FewerTilesThanCoresLeavesTrailingCoresEmpty) {
    const uint32_t total = 3;
    const uint32_t per_core = tiles_per_core(total, 8);
    EXPECT_EQ(per_core, 1u);
    expect_exact_cover(total, 8);
    // Cores past the end get the empty range [total, total).
    EXPECT_EQ(core_start(5, per_core, total), total);
    EXPECT_EQ(core_end(5, per_core, total), total);
    expect_balanced_cover(total, 8);
    EXPECT_EQ(balanced_end(7, total, 8) - balanced_start(7, total, 8), 0u);
}

TEST(RuntimeLengthTest, UnevenSplitsCoverExactly) {
    for (uint32_t total : {1u, 7u, 8u, 9u, 17u, 148u, 287u, 470u, 1000u}) {
        for (uint32_t cores : {1u, 3u, 8u, 64u, 120u}) {
            expect_exact_cover(total, cores);
            expect_balanced_cover(total, cores);
        }
    }
}

TEST(RuntimeLengthTest, LastCoreIsClampedNotOverrun) {
    // 17 tiles on 8 cores: 3 per core, the sixth core ends at 17, the seventh and eighth are empty.
    const uint32_t per_core = tiles_per_core(17, 8);
    EXPECT_EQ(per_core, 3u);
    EXPECT_EQ(core_end(5, per_core, 17), 17u);
    EXPECT_EQ(core_start(6, per_core, 17), 17u);
    EXPECT_EQ(core_end(7, per_core, 17), 17u);
}

TEST(RuntimeLengthTest, PartialBlocksRoundUpPerCore) {
    EXPECT_EQ(blocks_per_core(7, 4), 2u);
    EXPECT_EQ(blocks_per_core(8, 4), 2u);
    EXPECT_EQ(blocks_per_core(9, 4), 3u);
    // 17 tiles on 8 cores in blocks of 4: 3 tiles per core round to one block, so 4 * 8 = 32 padded tiles.
    EXPECT_EQ(padded_tiles(17, 8, 4), 32u);
    for (uint32_t total : {1u, 17u, 148u, 470u, 1000u}) {
        for (uint32_t cores : {1u, 8u, 64u}) {
            for (uint32_t block : {1u, 2u, 4u, 8u}) {
                const uint32_t padded = padded_tiles(total, cores, block);
                EXPECT_GE(padded, total);
                EXPECT_EQ(padded % (cores * block), 0u);
                // Never more than one block per core of slack.
                EXPECT_LE(padded, tiles_per_core(total, cores) * cores + (block - 1) * cores);
            }
        }
    }
}

TEST(RuntimeLengthTest, RoundsCountWorkerPasses) {
    EXPECT_EQ(rounds(1, 5), 1u);
    EXPECT_EQ(rounds(5, 5), 1u);
    EXPECT_EQ(rounds(6, 5), 2u);
    EXPECT_EQ(rounds(4736, 5), 948u);
}

}  // namespace
}  // namespace ttnn::runtime_length
