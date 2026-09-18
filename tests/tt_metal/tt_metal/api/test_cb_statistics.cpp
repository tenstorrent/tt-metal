// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <map>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include "impl/program/program_impl.hpp"

namespace tt::tt_metal {

TEST(CircularBufferStatistics, CPU_OverlappingCoreRangesPreservePhysicalAddressUnion) {
    // A byte-set oracle is deliberately independent of interval merging.
    // Duplicate, nested and adjacent allocations must count physical bytes once.
    using Regions = std::vector<std::pair<uint64_t, uint64_t>>;
    std::map<CoreRange, Regions> ranges;
    std::map<CoreCoord, std::set<uint64_t>> expected;
    std::mt19937 random(35);
    for (unsigned allocation = 0; allocation < 200; ++allocation) {
        const CoreCoord first(random() % 4, random() % 4);
        const CoreCoord last(first.x + random() % 3, first.y + random() % 3);
        const uint64_t begin = random() % 128;
        const uint64_t end = begin + 1 + random() % 32;
        ranges[CoreRange(first, last)].emplace_back(begin, end);
        for (uint32_t x = first.x; x <= last.x; ++x) {
            for (uint32_t y = first.y; y <= last.y; ++y) {
                for (uint64_t address = begin; address < end; ++address) {
                    expected[CoreCoord(x, y)].insert(address);
                }
            }
        }
    }
    const auto actual = detail::ProgramImpl::expand_cb_l1_regions_per_core(ranges);
    ASSERT_EQ(actual.size(), expected.size());
    for (const auto& [core, regions] : actual) {
        std::set<uint64_t> addresses;
        uint64_t previous_end = 0;
        for (const auto& [begin, end] : regions) {
            EXPECT_GT(end, begin);
            EXPECT_GE(begin, previous_end);
            for (uint64_t address = begin; address < end; ++address) {
                EXPECT_TRUE(addresses.insert(address).second);
            }
            previous_end = end;
        }
        EXPECT_EQ(addresses, expected.at(core));
    }
    EXPECT_TRUE(detail::ProgramImpl::expand_cb_l1_regions_per_core({}).empty());
}

}  // namespace tt::tt_metal
