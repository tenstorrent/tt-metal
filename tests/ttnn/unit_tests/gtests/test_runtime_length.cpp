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
#include "tt_metal/hw/inc/api/alignment.h"

namespace ttnn::runtime_length {
namespace {

// Every core's clamped [start, end) in order; they must tile [0, total) exactly.
void expect_exact_cover(uint32_t total, uint32_t num_cores) {
    const uint32_t per_core = div_up(total, num_cores);
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

TEST(RuntimeLengthTest, DivUpRoundsUpWithoutOverflow) {
    static_assert(div_up(0, 4) == 0);
    static_assert(div_up(1, 4) == 1);
    static_assert(div_up(4, 4) == 1);
    static_assert(div_up(5, 4) == 2);
    // a + b - 1 would wrap here.
    static_assert(div_up(UINT32_MAX, 1) == UINT32_MAX);
    static_assert(div_up(UINT32_MAX, 2) == 2147483648u);
    static_assert(div_up(UINT32_MAX - 1, UINT32_MAX) == 1);
}

TEST(RuntimeLengthTest, KernelDivUpMatchesTheHost) {
    // Kernels resolve div_up to the kernel API's (tt_metal/hw/inc/api/alignment.h), the host to tt::div_up:
    // the same results, including where a + b - 1 would wrap, and align() built on it matches the old formula
    // wherever that did not wrap.
    static_assert(::div_up(0u, 4u) == 0u);
    static_assert(::div_up(5u, 4u) == 2u);
    static_assert(::div_up(UINT32_MAX, 2u) == 2147483648u);
    static_assert(::div_up(UINT32_MAX - 1, UINT32_MAX) == 1u);
    static_assert(::div_up(-5, 4) == -1);
    static_assert(::align(0u, 64u) == 0u);
    static_assert(::align(1u, 64u) == 64u);
    static_assert(::align(64u, 64u) == 64u);
    static_assert(::align(UINT32_MAX - 1, 3u) == UINT32_MAX);  // addr + 2 would wrap to 0
    for (uint32_t a : {0u, 1u, 7u, 31u, 32u, 33u, 1000u, 65535u, 1u << 20}) {
        for (uint32_t b : {1u, 2u, 3u, 16u, 32u, 64u, 1024u}) {
            EXPECT_EQ(::div_up(a, b), tt::div_up(a, b));
            EXPECT_EQ(::align(a, b), ((a + b - 1) / b) * b);
        }
    }
}

TEST(RuntimeLengthTest, ZeroWorkGivesEmptyRanges) {
    const uint32_t per_core = div_up(0, 8);
    EXPECT_EQ(per_core, 0u);
    for (uint32_t core = 0; core < 8; ++core) {
        EXPECT_EQ(core_start(core, per_core, 0), 0u);
        EXPECT_EQ(core_end(core, per_core, 0), 0u);
        EXPECT_EQ(balanced_start(core, 0, 8), 0u);
        EXPECT_EQ(balanced_end(core, 0, 8), 0u);
        const auto split = uniform_split(0, 8, core, 4);
        EXPECT_EQ(split.per_core, 0u);
        EXPECT_EQ(split.blocks_per_core, 0u);
        EXPECT_EQ(split.padded_total, 0u);
        EXPECT_EQ(split.start, 0u);
        EXPECT_EQ(split.end, 0u);
    }
}

TEST(RuntimeLengthTest, FewerTilesThanCoresLeavesTrailingCoresEmpty) {
    const uint32_t total = 3;
    const uint32_t per_core = div_up(total, 8);
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

TEST(RuntimeLengthTest, RangePairsMatchTheirStartAndEnd) {
    // chunked_range wraps core_start / core_end at per_core = div_up(total, cores); balanced_range wraps
    // balanced_start / balanced_end. The kernels take the pair form.
    for (uint32_t total : {0u, 1u, 7u, 17u, 148u, 470u}) {
        for (uint32_t cores : {1u, 3u, 8u, 64u}) {
            const uint32_t per_core = div_up(total, cores);
            for (uint32_t core = 0; core < cores; ++core) {
                const auto [c_start, c_end] = chunked_range(core, total, cores);
                EXPECT_EQ(c_start, core_start(core, per_core, total)) << "total " << total << " cores " << cores;
                EXPECT_EQ(c_end, core_end(core, per_core, total));
                const auto [b_start, b_end] = balanced_range(core, total, cores);
                EXPECT_EQ(b_start, balanced_start(core, total, cores));
                EXPECT_EQ(b_end, balanced_end(core, total, cores));
            }
        }
    }
}

TEST(RuntimeLengthTest, LastCoreIsClampedNotOverrun) {
    // 17 tiles on 8 cores: 3 per core, the sixth core ends at 17, the seventh and eighth are empty.
    const uint32_t per_core = div_up(17, 8);
    EXPECT_EQ(per_core, 3u);
    EXPECT_EQ(core_end(5, per_core, 17), 17u);
    EXPECT_EQ(core_start(6, per_core, 17), 17u);
    EXPECT_EQ(core_end(7, per_core, 17), 17u);
}

TEST(RuntimeLengthTest, UniformSplitKeepsEveryCoreOnTheSameBlockCount) {
    // 17 tiles on 8 cores in blocks of 4: 3 tiles per core round to one block; the split spans 24 tiles, and the
    // sixth core's range [15, 18) runs past the total, which the kernels clip themselves.
    constexpr auto split = uniform_split(17, 8, 5, 4);
    static_assert(split.per_core == 3);
    static_assert(split.blocks_per_core == 1);
    static_assert(split.padded_total == 24);
    static_assert(split.start == 15);
    static_assert(split.end == 18);
    for (uint32_t total : {1u, 17u, 148u, 470u, 1000u}) {
        for (uint32_t cores : {1u, 8u, 64u}) {
            for (uint32_t block : {1u, 2u, 4u, 8u}) {
                uint32_t next = 0;
                for (uint32_t core = 0; core < cores; ++core) {
                    const auto s = uniform_split(total, cores, core, block);
                    EXPECT_EQ(s.per_core, div_up(total, cores));
                    EXPECT_EQ(s.blocks_per_core, div_up(s.per_core, block));
                    EXPECT_EQ(s.padded_total, s.per_core * cores);
                    EXPECT_GE(s.padded_total, total);
                    EXPECT_EQ(s.start, next);
                    EXPECT_EQ(s.end - s.start, s.per_core);
                    next = s.end;
                }
                EXPECT_EQ(next, div_up(total, cores) * cores);
            }
        }
    }
}

}  // namespace
}  // namespace ttnn::runtime_length
