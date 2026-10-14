// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Host tests for the args-block and result I/O boundary of harness/regions.

#include "harness/regions/regions.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

namespace hreg = x280::harness::regions;

// Records writes and returns `read_bytes` bytes of 0xA5 for any read.
struct FakeTile {
    struct Write {
        uint64_t addr;
        std::vector<uint8_t> bytes;
    };
    std::vector<Write> writes;
    size_t reads = 0;
    size_t read_bytes = 0;

    void write_available_lim(uint64_t addr, std::span<const uint8_t> data) {
        writes.push_back({addr, {data.begin(), data.end()}});
    }
    std::vector<uint8_t> read_available_lim(uint64_t, size_t) {
        ++reads;
        return std::vector<uint8_t>(read_bytes, 0xA5);
    }
};

struct Block65 {
    uint8_t bytes[65];
};

constexpr hreg::Region kLine{.addr = 0x1000, .bytes = hreg::kCacheLineBytes};

}  // namespace

TEST(HarnessRegions, WriteArgsPadsToTheRegionAndRefusesAnOversizeBlock) {
    FakeTile tile;
    hreg::write_args(tile, kLine, uint64_t{0x1122334455667788u});
    ASSERT_EQ(tile.writes.size(), 1u);
    EXPECT_EQ(tile.writes[0].addr, kLine.addr);
    ASSERT_EQ(tile.writes[0].bytes.size(), kLine.bytes);
    EXPECT_EQ(tile.writes[0].bytes[0], 0x88u);
    EXPECT_EQ(tile.writes[0].bytes[8], 0u);

    EXPECT_THROW(hreg::write_args(tile, kLine, Block65{}), std::invalid_argument);
    EXPECT_EQ(tile.writes.size(), 1u) << "an oversize block must be refused before any write";
}

TEST(HarnessRegions, ReadStructRefusesAShortReadAndAnOversizeStruct) {
    FakeTile tile;
    tile.read_bytes = sizeof(uint64_t);
    EXPECT_EQ(hreg::read_struct<uint64_t>(tile, kLine), 0xA5A5A5A5A5A5A5A5u);

    tile.read_bytes = sizeof(uint64_t) - 1u;
    EXPECT_THROW(hreg::read_struct<uint64_t>(tile, kLine), std::runtime_error);

    const size_t reads = tile.reads;
    EXPECT_THROW(hreg::read_struct<Block65>(tile, kLine), std::invalid_argument);
    EXPECT_EQ(tile.reads, reads) << "an oversize struct must be refused before any read";
}

TEST(HarnessRegions, ArenaRefusesAnAllocationPastItsEndAndLeavesTheCursor) {
    hreg::Arena arena(hreg::Region{.addr = 0x1000, .bytes = 2u * hreg::kCacheLineBytes});
    EXPECT_EQ(arena.allocate(1u).bytes, hreg::kCacheLineBytes);
    EXPECT_THROW((void)arena.allocate(hreg::kCacheLineBytes + 1u), std::invalid_argument);
    EXPECT_THROW((void)arena.allocate(UINT64_MAX), std::invalid_argument)
        << "a size whose rounding wraps to zero must not vend an empty region";
    EXPECT_EQ(arena.cursor(), 0x1000u + hreg::kCacheLineBytes);
    EXPECT_EQ(arena.allocate(hreg::kCacheLineBytes).addr, 0x1000u + hreg::kCacheLineBytes);

    EXPECT_THROW(hreg::Arena(hreg::Region{.addr = UINT64_MAX - 63u, .bytes = 64u}), std::invalid_argument);
}
