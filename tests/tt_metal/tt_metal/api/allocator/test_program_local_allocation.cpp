// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <random>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "impl/allocator/persistent_l1_arena.hpp"
#include "impl/program/program_allocation_support.hpp"

namespace tt::tt_metal::program_local_allocation_tests {
namespace {

using detail::EphemeralAllocationMetadata;
using detail::prepare_ephemeral_allocation;
using detail::register_ephemeral_allocation;

constexpr uint64_t base = 4096;
const CoreRangeSet left(CoreRange({0, 0}));
const CoreRangeSet right(CoreRange({1, 0}));
const CoreRangeSet both(CoreRange({0, 0}, {1, 0}));

// This oracle records actual byte intervals on individual cores. It never reads
// allocator regions or waterlines and does not use the allocator's intersection scan.
using Occupancy = std::map<CoreCoord, std::vector<std::pair<uint64_t, uint64_t>>>;

void check_and_record(Occupancy& occupied, const CoreRangeSet& cores, uint64_t address, uint64_t size) {
    for (const auto& range : cores.ranges()) {
        for (uint32_t y = range.start_coord.y; y <= range.end_coord.y; ++y) {
            for (uint32_t x = range.start_coord.x; x <= range.end_coord.x; ++x) {
                auto& intervals = occupied[CoreCoord{x, y}];
                for (const auto& [begin, end] : intervals) {
                    ASSERT_TRUE(address + size <= begin || end <= address)
                        << "Overlap on core (" << x << ", " << y << "): [" << address << ", " << address + size
                        << ") and [" << begin << ", " << end << ")";
                }
                if (size != 0) {
                    intervals.emplace_back(address, address + size);
                }
            }
        }
    }
}

class ProgramLocalAllocationTest : public ::testing::TestWithParam<bool> {
protected:
    uint64_t allocate(EphemeralAllocationMetadata& metadata, const CoreRangeSet& cores, uint64_t size) const {
        return GetParam() ? detail::allocate_scratchpad(metadata, cores, size, 32)
                          : detail::allocate_dfb(metadata, cores, size, 32);
    }
};

TEST_P(ProgramLocalAllocationTest, CPU_UnequalDisjointPrefixesThenSharedRange) {
    // The two-core, three-DFB reproduction from #62451.
    PersistentL1Arena arena(base, 1 << 20, both);
    EphemeralAllocationMetadata metadata;
    for (const auto& cores : {left, right, both}) {
        register_ephemeral_allocation(metadata, cores);
    }
    prepare_ephemeral_allocation(metadata, arena);
    Occupancy occupied;
    const auto a = allocate(metadata, left, 1024);
    const auto b = allocate(metadata, right, 2048);
    const auto ab = allocate(metadata, both, 1024);
    EXPECT_EQ(a, base);
    EXPECT_EQ(b, base) << "Disjoint allocations may reuse addresses despite a registered spanning range";
    EXPECT_EQ(ab, base + 2048);
    ASSERT_NO_FATAL_FAILURE(check_and_record(occupied, left, a, 1024));
    ASSERT_NO_FATAL_FAILURE(check_and_record(occupied, right, b, 2048));
    ASSERT_NO_FATAL_FAILURE(check_and_record(occupied, both, ab, 1024));
}

TEST_P(ProgramLocalAllocationTest, CPU_UnequalPersistentBasesAndReallocation) {
    PersistentL1Arena arena(base, 1 << 20, both);
    arena.allocate(right, 1024, 32);
    EphemeralAllocationMetadata metadata;
    register_ephemeral_allocation(metadata, left);
    register_ephemeral_allocation(metadata, both);
    for (int round = 0; round < 2; ++round) {
        SCOPED_TRACE(round);
        prepare_ephemeral_allocation(metadata, arena);
        Occupancy occupied{{CoreCoord{0, 0}, {{0, base}}}, {CoreCoord{1, 0}, {{0, base + 1024}}}};
        const auto a = allocate(metadata, left, 512);
        EXPECT_EQ(a, base);
        ASSERT_NO_FATAL_FAILURE(check_and_record(occupied, left, a, 512));
        const auto ab = allocate(metadata, both, 1024);
        ASSERT_NO_FATAL_FAILURE(check_and_record(occupied, both, ab, 1024));
        // Preparing again must retain the persistent base and the live allocations.
        ASSERT_NO_THROW(prepare_ephemeral_allocation(metadata, arena));
        const auto next = allocate(metadata, left, 256);
        ASSERT_NO_FATAL_FAILURE(check_and_record(occupied, left, next, 256));
        detail::invalidate_ephemeral_allocation_metadata(metadata);
    }
}

TEST_P(ProgramLocalAllocationTest, CPU_MultipleRangesTouchOneTrackedRange) {
    const CoreRangeSet whole(CoreRange({0, 0}, {2, 0}));
    const CoreRangeSet ends(std::vector<CoreRange>{CoreRange({0, 0}), CoreRange({2, 0})});
    PersistentL1Arena arena(base, 1 << 20, whole);
    EphemeralAllocationMetadata metadata;
    register_ephemeral_allocation(metadata, whole);
    register_ephemeral_allocation(metadata, ends);
    prepare_ephemeral_allocation(metadata, arena);
    Occupancy occupied;
    const auto a = allocate(metadata, ends, 96);
    const auto b = allocate(metadata, whole, 32);
    EXPECT_EQ(a, base);
    EXPECT_EQ(b, base + 96);
    ASSERT_NO_FATAL_FAILURE(check_and_record(occupied, ends, a, 96));
    ASSERT_NO_FATAL_FAILURE(check_and_record(occupied, whole, b, 32));
}

TEST_P(ProgramLocalAllocationTest, CPU_ReprepareWithHigherPersistentBaseBelowLiveAllocations) {
    PersistentL1Arena first_arena(base, 1 << 20, both);
    first_arena.allocate(right, 1024, 32);
    PersistentL1Arena second_arena(base, 1 << 20, both);
    second_arena.allocate(left, 512, 32);
    second_arena.allocate(right, 1024, 32);
    EphemeralAllocationMetadata metadata;
    register_ephemeral_allocation(metadata, left);
    register_ephemeral_allocation(metadata, both);
    prepare_ephemeral_allocation(metadata, first_arena);
    const auto ab = allocate(metadata, both, 1024);
    const auto a = allocate(metadata, left, 512);
    ASSERT_EQ(ab, base + 1024);
    ASSERT_EQ(a, base + 2048);
    Occupancy occupied{{CoreCoord{0, 0}, {{0, base + 512}}}, {CoreCoord{1, 0}, {{0, base + 1024}}}};
    ASSERT_NO_FATAL_FAILURE(check_and_record(occupied, both, ab, 1024));
    ASSERT_NO_FATAL_FAILURE(check_and_record(occupied, left, a, 512));
    ASSERT_NO_THROW(prepare_ephemeral_allocation(metadata, second_arena));
    const auto next = allocate(metadata, left, 256);
    ASSERT_NO_FATAL_FAILURE(check_and_record(occupied, left, next, 256));
}

TEST_P(ProgramLocalAllocationTest, CPU_RegisteringNewRangesRequiresInvalidatingLiveAllocations) {
    PersistentL1Arena arena(base, 1 << 20, both);
    EphemeralAllocationMetadata metadata;
    register_ephemeral_allocation(metadata, left);
    prepare_ephemeral_allocation(metadata, arena);
    allocate(metadata, left, 1024);
    EXPECT_NO_THROW(register_ephemeral_allocation(metadata, left));
    EXPECT_THROW(register_ephemeral_allocation(metadata, both), std::exception);
    detail::invalidate_ephemeral_allocation_metadata(metadata);
    register_ephemeral_allocation(metadata, both);
    prepare_ephemeral_allocation(metadata, arena);
    Occupancy occupied;
    const auto a = allocate(metadata, left, 1024);
    const auto ab = allocate(metadata, both, 1024);
    ASSERT_NO_FATAL_FAILURE(check_and_record(occupied, left, a, 1024));
    ASSERT_NO_FATAL_FAILURE(check_and_record(occupied, both, ab, 1024));
}

TEST_P(ProgramLocalAllocationTest, CPU_EmptyPlacementAndZeroSizeDoNotReserveSpace) {
    PersistentL1Arena arena(base, 1 << 20, both);
    EphemeralAllocationMetadata metadata;
    register_ephemeral_allocation(metadata, left);
    prepare_ephemeral_allocation(metadata, arena);
    EXPECT_EQ(allocate(metadata, CoreRangeSet{}, 1024), base);
    EXPECT_EQ(allocate(metadata, left, 0), base);
    EXPECT_EQ(allocate(metadata, left, 32), base);
    EXPECT_EQ(allocate(metadata, left, 32), base + 32);
}

TEST(ProgramLocalAllocationGuardTest, CPU_RejectsUnpreparedIntersectingRange) {
    PersistentL1Arena arena(base, 1 << 20, both);
    EphemeralAllocationMetadata metadata;
    register_ephemeral_allocation(metadata, left);
    prepare_ephemeral_allocation(metadata, arena);
    register_ephemeral_allocation(metadata, both);
    EXPECT_THROW(detail::allocate_dfb(metadata, left, 32, 32), std::exception);
}

TEST(ProgramLocalAllocationGuardTest, CPU_RejectsUnpreparedAndBackwardMarks) {
    EphemeralAllocationMetadata::LockstepAllocationRegion region(CoreRange({0, 0}));
    EXPECT_THROW(region.mark_address(base, 32), std::exception);
    region.l1_regions.emplace_back(base, base);
    region.mark_address(base, 32);
    EXPECT_THROW(region.mark_address(base, 16), std::exception);
    region.mark_address(base + 64, 32);
    EXPECT_EQ(region.l1_regions.back(), std::make_pair(base + 64, base + 96));
}

TEST_P(ProgramLocalAllocationTest, CPU_GeneratedPlacementsUseIndependentPerCoreOracle) {
    struct Request {
        CoreRangeSet cores;
        uint64_t size;
        uint32_t alignment;
    };
    constexpr std::array<uint64_t, 6> sizes{16, 32, 96, 128, 1024, 2048};
    constexpr std::array<uint32_t, 4> alignments{16, 32, 64, 256};
    constexpr uint32_t grid_width = 4, grid_height = 3;
    constexpr uint32_t num_cores = grid_width * grid_height;
    const CoreRangeSet grid(CoreRange({0, 0}, {grid_width - 1, grid_height - 1}));
    std::mt19937 random(62451);
    for (uint32_t sample = 0; sample < 128; ++sample) {
        SCOPED_TRACE(sample);
        std::vector<Request> requests;
        for (uint32_t i = 0; i < 24; ++i) {
            std::vector<CoreRange> ranges;
            if (i % 3 == 0) {
                // A set of distinct singleton ranges, including nonrectangular placements.
                const uint32_t mask = 1 + random() % ((1 << num_cores) - 1);
                for (uint32_t bit = 0; bit < num_cores; ++bit) {
                    if (mask & (1 << bit)) {
                        ranges.emplace_back(CoreCoord{bit % grid_width, bit / grid_width});
                    }
                }
            } else {
                const uint32_t x0 = random() % grid_width, x1 = random() % grid_width;
                const uint32_t y0 = random() % grid_height, y1 = random() % grid_height;
                ranges.emplace_back(
                    CoreCoord{std::min(x0, x1), std::min(y0, y1)}, CoreCoord{std::max(x0, x1), std::max(y0, y1)});
            }
            requests.push_back(
                {CoreRangeSet(std::move(ranges)),
                 sizes[random() % sizes.size()],
                 alignments[random() % alignments.size()]});
        }
        // Registration order and invalidation must not change placement or safety.
        std::vector<uint64_t> expected_addresses;
        for (bool reverse_registration : {false, true}) {
            PersistentL1Arena arena(base, 1 << 20, grid);
            Occupancy persistent;
            for (uint32_t y = 0; y < grid_height; ++y) {
                for (uint32_t x = 0; x < grid_width; ++x) {
                    const CoreCoord core{x, y};
                    const CoreRangeSet cores{CoreRange(core)};
                    const uint64_t size = ((sample + x * 17 + y * 13) % 5) * 32;
                    if (size != 0) {
                        arena.allocate(cores, size, 32);
                    }
                    persistent[core].emplace_back(0, arena.high_water_mark(cores));
                }
            }
            EphemeralAllocationMetadata metadata;
            for (size_t i = 0; i < requests.size(); ++i) {
                const auto& request = requests[reverse_registration ? requests.size() - 1 - i : i];
                register_ephemeral_allocation(metadata, request.cores);
            }
            for (int round = 0; round < 2; ++round) {
                prepare_ephemeral_allocation(metadata, arena);
                Occupancy occupied = persistent;
                std::vector<uint64_t> addresses;
                for (size_t i = 0; i < requests.size(); ++i) {
                    SCOPED_TRACE(i);
                    const auto& r = requests[i];
                    // Exercise DFB-only sequences and mixed DFB/scratchpad sequences.
                    const auto allocator =
                        GetParam() && i % 2 == 0 ? detail::allocate_scratchpad : detail::allocate_dfb;
                    const auto address = allocator(metadata, r.cores, r.size, r.alignment);
                    ASSERT_EQ(address % r.alignment, 0u);
                    ASSERT_NO_FATAL_FAILURE(check_and_record(occupied, r.cores, address, r.size));
                    addresses.push_back(address);
                }
                if (expected_addresses.empty()) {
                    expected_addresses = addresses;
                } else {
                    EXPECT_EQ(addresses, expected_addresses);
                }
                ASSERT_NO_THROW(prepare_ephemeral_allocation(metadata, arena));
                detail::invalidate_ephemeral_allocation_metadata(metadata);
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    DFBAndScratchpad, ProgramLocalAllocationTest, ::testing::Bool(), [](const ::testing::TestParamInfo<bool>& info) {
        return info.param ? "Scratchpad" : "DFB";
    });

}  // namespace
}  // namespace tt::tt_metal::program_local_allocation_tests
