// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "tt_metal/impl/dispatch/simple_trace_allocator.hpp"

namespace tt::tt_metal {

// NOLINTBEGIN(cppcoreguidelines-virtual-class-destructor)
class SimpleTraceAllocatorFixture : public ::testing::Test {
public:
    using ExtraData = SimpleTraceAllocator::ExtraData;
    using RegionAllocator = SimpleTraceAllocator::RegionAllocator;

    SimpleTraceAllocatorFixture(const SimpleTraceAllocatorFixture&) = delete;
    SimpleTraceAllocatorFixture& operator=(const SimpleTraceAllocatorFixture&) = delete;
    SimpleTraceAllocatorFixture(SimpleTraceAllocatorFixture&&) = delete;
    SimpleTraceAllocatorFixture& operator=(SimpleTraceAllocatorFixture&&) = delete;

protected:
    SimpleTraceAllocatorFixture() = default;
    ~SimpleTraceAllocatorFixture() override = default;

    static bool intersects(uint32_t begin_1, uint32_t size_1, uint32_t begin_2, uint32_t size_2) {
        return SimpleTraceAllocator::intersects(begin_1, size_1, begin_2, size_2);
    }

    static std::optional<uint32_t> merge_syncs(std::optional<uint32_t> a, std::optional<uint32_t> b) {
        return SimpleTraceAllocator::merge_syncs(a, b);
    }

    RegionAllocator make_allocator(uint32_t ringbuffer_size) { return RegionAllocator(ringbuffer_size, extra_data_); }

    std::vector<ExtraData> extra_data_;
};
// NOLINTEND(cppcoreguidelines-virtual-class-destructor)

using ExtraData = SimpleTraceAllocatorFixture::ExtraData;

// --- intersects ---

TEST_F(SimpleTraceAllocatorFixture, IntersectsNonOverlapping) {
    EXPECT_FALSE(intersects(0, 10, 10, 10));
    EXPECT_FALSE(intersects(10, 10, 0, 10));
    EXPECT_FALSE(intersects(0, 5, 100, 5));
}

TEST_F(SimpleTraceAllocatorFixture, IntersectsOverlapping) {
    EXPECT_TRUE(intersects(0, 10, 5, 10));
    EXPECT_TRUE(intersects(5, 10, 0, 10));
}

TEST_F(SimpleTraceAllocatorFixture, IntersectsContainment) {
    EXPECT_TRUE(intersects(0, 100, 10, 5));
    EXPECT_TRUE(intersects(10, 5, 0, 100));
}

TEST_F(SimpleTraceAllocatorFixture, IntersectsSameRegion) { EXPECT_TRUE(intersects(5, 10, 5, 10)); }

// --- merge_syncs ---

TEST_F(SimpleTraceAllocatorFixture, MergeSyncsBothNullopt) {
    EXPECT_EQ(merge_syncs(std::nullopt, std::nullopt), std::nullopt);
}

TEST_F(SimpleTraceAllocatorFixture, MergeSyncsFirstOnly) { EXPECT_EQ(merge_syncs(5, std::nullopt), 5); }

TEST_F(SimpleTraceAllocatorFixture, MergeSyncsSecondOnly) { EXPECT_EQ(merge_syncs(std::nullopt, 7), 7); }

TEST_F(SimpleTraceAllocatorFixture, MergeSyncsBothPicksMax) {
    EXPECT_EQ(merge_syncs(3, 9), 9);
    EXPECT_EQ(merge_syncs(9, 3), 9);
    EXPECT_EQ(merge_syncs(4, 4), 4);
}

// --- allocate_region ---

TEST_F(SimpleTraceAllocatorFixture, ZeroSizeAllocation) {
    extra_data_.resize(1);
    auto alloc = make_allocator(1024);
    auto [sync, addr] = alloc.allocate_region(0, 0, ExtraData::kNonBinary, 100);
    EXPECT_FALSE(sync.has_value());
    EXPECT_EQ(addr, 0u);
}

TEST_F(SimpleTraceAllocatorFixture, BasicAllocationEmptyBuffer) {
    extra_data_.resize(1);
    auto alloc = make_allocator(1024);
    auto [sync, addr] = alloc.allocate_region(100, 0, ExtraData::kNonBinary, 100);
    EXPECT_FALSE(sync.has_value());
    ASSERT_TRUE(addr.has_value());
    EXPECT_EQ(*addr, 0u);
}

TEST_F(SimpleTraceAllocatorFixture, SequentialAllocationsNoOverlap) {
    extra_data_.resize(3);
    extra_data_[0].next_use_idx[ExtraData::kNonBinary] = 2;
    extra_data_[1].next_use_idx[ExtraData::kNonBinary] = 2;
    auto alloc = make_allocator(1024);

    auto [sync0, addr0] = alloc.allocate_region(100, 0, ExtraData::kNonBinary, 10);
    ASSERT_TRUE(addr0.has_value());
    EXPECT_EQ(*addr0, 0u);

    auto [sync1, addr1] = alloc.allocate_region(100, 1, ExtraData::kNonBinary, 20);
    ASSERT_TRUE(addr1.has_value());
    // Should be placed after the first region since evicting has a cost.
    EXPECT_EQ(*addr1, 100u);
}

TEST_F(SimpleTraceAllocatorFixture, AllocationTooLargeForBuffer) {
    extra_data_.resize(1);
    auto alloc = make_allocator(50);
    auto [sync, addr] = alloc.allocate_region(100, 0, ExtraData::kNonBinary, 100);
    EXPECT_FALSE(addr.has_value());
}

TEST_F(SimpleTraceAllocatorFixture, AllocationExactFit) {
    extra_data_.resize(1);
    auto alloc = make_allocator(100);
    auto [sync, addr] = alloc.allocate_region(100, 0, ExtraData::kNonBinary, 100);
    ASSERT_TRUE(addr.has_value());
    EXPECT_EQ(*addr, 0u);
}

TEST_F(SimpleTraceAllocatorFixture, EvictionWhenBufferFull) {
    // Buffer of size 100, allocate two 60-byte regions. The second must evict the first.
    extra_data_.resize(2);
    auto alloc = make_allocator(100);

    auto [sync0, addr0] = alloc.allocate_region(60, 0, ExtraData::kNonBinary, 10);
    ASSERT_TRUE(addr0.has_value());
    EXPECT_EQ(*addr0, 0u);

    auto [sync1, addr1] = alloc.allocate_region(60, 1, ExtraData::kNonBinary, 20);
    ASSERT_TRUE(addr1.has_value());
    // Only placement that fits is [0,60) which overlaps the existing region.
    EXPECT_EQ(*addr1, 0u);
    // Sync should point to the evicted region's trace_idx.
    EXPECT_TRUE(sync1.has_value());
    EXPECT_EQ(*sync1, 0u);
}

TEST_F(SimpleTraceAllocatorFixture, CurrentNodeEvictionPenalty) {
    // Two regions. One has next_use == current trace_idx (penalized), the other has next_use far away.
    // Use wide spacing to isolate the penalty from stall-avoidance costs.
    constexpr uint32_t spacing = 100;
    extra_data_.resize(3 * spacing);
    extra_data_[0].next_use_idx[ExtraData::kNonBinary] = 2 * spacing;            // Next use IS the current allocation.
    extra_data_[spacing].next_use_idx[ExtraData::kNonBinary] = 3 * spacing - 1;  // Far away (low Belady cost).

    auto alloc = make_allocator(200);

    alloc.allocate_region(100, 0, ExtraData::kNonBinary, 10);
    alloc.allocate_region(100, spacing, ExtraData::kNonBinary, 20);

    // Allocating at trace_idx=2*spacing. Region 0 has next_use_idx==2*spacing (massive penalty).
    // Region spacing has next_use_idx far away (low cost). Should evict region spacing.
    auto [sync, addr] = alloc.allocate_region(100, 2 * spacing, ExtraData::kNonBinary, 30);
    ASSERT_TRUE(addr.has_value());
    EXPECT_EQ(*addr, 100u);
}

TEST_F(SimpleTraceAllocatorFixture, NowInUseSkipsPlacement) {
    // A region with the same trace_idx as the current allocation cannot be evicted.
    extra_data_.resize(2);

    auto alloc = make_allocator(100);

    // Allocate a region at trace_idx=0.
    alloc.allocate_region(60, 0, ExtraData::kNonBinary, 10);

    // Try to allocate at trace_idx=0 again. Placement at addr 0 is skipped (now_in_use).
    // The only other placement starts at 60, and 60+60=120 > 100, so it doesn't fit.
    auto [sync, addr] = alloc.allocate_region(60, 0, ExtraData::kBinary, 10);
    EXPECT_FALSE(addr.has_value());
}

TEST_F(SimpleTraceAllocatorFixture, NowInUseWithRoomAfter) {
    extra_data_.resize(2);

    auto alloc = make_allocator(200);

    alloc.allocate_region(60, 0, ExtraData::kNonBinary, 10);

    // Same trace_idx but enough room after the existing region.
    auto [sync, addr] = alloc.allocate_region(60, 0, ExtraData::kBinary, 10);
    ASSERT_TRUE(addr.has_value());
    EXPECT_EQ(*addr, 60u);
}

TEST_F(SimpleTraceAllocatorFixture, EvictionReturnsSyncIdx) {
    extra_data_.resize(3);
    auto alloc = make_allocator(100);

    alloc.allocate_region(50, 0, ExtraData::kNonBinary, 10);
    alloc.allocate_region(50, 1, ExtraData::kNonBinary, 20);

    // Evicting both: sync_idx should be max(0, 1) == 1.
    auto [sync, addr] = alloc.allocate_region(100, 2, ExtraData::kNonBinary, 30);
    ASSERT_TRUE(addr.has_value());
    EXPECT_EQ(*addr, 0u);
    ASSERT_TRUE(sync.has_value());
    EXPECT_EQ(*sync, 1u);
}

TEST_F(SimpleTraceAllocatorFixture, ResetAllocatorClearsState) {
    extra_data_.resize(2);
    auto alloc = make_allocator(100);

    alloc.allocate_region(50, 0, ExtraData::kNonBinary, 10);
    alloc.add_region(ExtraData::kBinary, 10, 0);

    alloc.reset_allocator();

    // After reset, the buffer should be empty.
    EXPECT_FALSE(alloc.get_region(ExtraData::kBinary, 10).has_value());

    // Should allocate at 0 again with no sync.
    auto [sync, addr] = alloc.allocate_region(50, 1, ExtraData::kNonBinary, 20);
    ASSERT_TRUE(addr.has_value());
    EXPECT_EQ(*addr, 0u);
    EXPECT_FALSE(sync.has_value());
}

TEST_F(SimpleTraceAllocatorFixture, AddAndGetRegion) {
    extra_data_.resize(1);
    auto alloc = make_allocator(100);

    EXPECT_FALSE(alloc.get_region(ExtraData::kBinary, 42).has_value());
    alloc.add_region(ExtraData::kBinary, 42, 200);
    EXPECT_EQ(alloc.get_region(ExtraData::kBinary, 42), 200u);
    EXPECT_FALSE(alloc.get_region(ExtraData::kNonBinary, 42).has_value());
}

TEST_F(SimpleTraceAllocatorFixture, UpdateRegionTraceIdx) {
    extra_data_.resize(3);
    extra_data_[0].next_use_idx[ExtraData::kNonBinary] = 2;

    auto alloc = make_allocator(200);

    alloc.allocate_region(50, 0, ExtraData::kNonBinary, 10);

    // Update trace_idx of the region at addr 0 from 0 to 1.
    alloc.update_region_trace_idx(0, 1);

    // Now allocating at trace_idx=1 with the same data_type: addr 0 is "now_in_use" for trace_idx 1.
    auto [sync, addr] = alloc.allocate_region(50, 1, ExtraData::kNonBinary, 20);
    ASSERT_TRUE(addr.has_value());
    EXPECT_EQ(*addr, 50u);
}

TEST_F(SimpleTraceAllocatorFixture, UpdateRegionTraceIdxNonexistent) {
    extra_data_.resize(1);
    auto alloc = make_allocator(100);

    // Should not crash when the address doesn't exist.
    alloc.update_region_trace_idx(999, 0);
}

TEST_F(SimpleTraceAllocatorFixture, MultipleDataTypes) {
    // Two allocations with different data_types at the same trace_idx are independent.
    // The second allocation can't overlap the first (same trace_idx → now_in_use), so it goes after.
    extra_data_.resize(1);
    auto alloc = make_allocator(200);

    auto [s0, a0] = alloc.allocate_region(100, 0, ExtraData::kNonBinary, 10);
    auto [s1, a1] = alloc.allocate_region(100, 0, ExtraData::kBinary, 10);

    ASSERT_TRUE(a0.has_value());
    ASSERT_TRUE(a1.has_value());
    EXPECT_EQ(*a0, 0u);
    EXPECT_EQ(*a1, 100u);
}

TEST_F(SimpleTraceAllocatorFixture, OldRegionsWithNoFutureUseDeleted) {
    // Regions with no future uses that are old enough get cleaned up from the internal map.
    // Use a large gap to ensure we exceed max_stall_history_size regardless of its compile-time value.
    constexpr uint32_t gap = 100;
    extra_data_.resize(gap + 1);

    auto alloc = make_allocator(10000);

    // Allocate a region at trace_idx=0 with no future use.
    alloc.allocate_region(100, 0, ExtraData::kNonBinary, 10);
    alloc.add_region(ExtraData::kNonBinary, 10, 0);

    // Allocate something overlapping region 0, from a trace_idx far enough away that
    // the old region gets cleaned up via marked_for_deletion.
    auto [sync, addr] = alloc.allocate_region(50, gap, ExtraData::kNonBinary, 99);
    ASSERT_TRUE(addr.has_value());
    EXPECT_EQ(*addr, 0u);
    // The program_ids_memory_map entry for program 10 should also be removed.
    EXPECT_FALSE(alloc.get_region(ExtraData::kNonBinary, 10).has_value());
}

TEST_F(SimpleTraceAllocatorFixture, BeladyEvictsLowestCostRegion) {
    // Fill a buffer with three regions, then request one that requires evicting.
    // Use trace indices spaced far apart to avoid the stall-avoidance cost dominating.
    constexpr uint32_t spacing = 100;
    extra_data_.resize(4 * spacing);

    // Regions at indices 0, spacing, 2*spacing. Future uses at different distances from 3*spacing.
    extra_data_[0].next_use_idx[ExtraData::kNonBinary] = 3 * spacing + 2;            // distance 2 (high cost)
    extra_data_[spacing].next_use_idx[ExtraData::kNonBinary] = 3 * spacing + 1;      // distance 1 (highest cost)
    extra_data_[2 * spacing].next_use_idx[ExtraData::kNonBinary] = 3 * spacing + 3;  // distance 3 (lowest cost)

    auto alloc = make_allocator(300);

    alloc.allocate_region(100, 0, ExtraData::kNonBinary, 10);            // [0, 100)
    alloc.allocate_region(100, spacing, ExtraData::kNonBinary, 20);      // [100, 200)
    alloc.allocate_region(100, 2 * spacing, ExtraData::kNonBinary, 30);  // [200, 300)

    // Allocate 100 bytes at trace_idx=3*spacing. Each placement overlaps exactly one region.
    // The Belady cost for each: size / (next_use - trace_idx).
    //   Region 0: 100 / 2 = 50
    //   Region spacing: 100 / 1 = 100
    //   Region 2*spacing: 100 / 3 = 33.3
    // All are far enough from the current idx that the stall penalty doesn't apply.
    // Region 2*spacing has the lowest cost, so it should be evicted.
    auto [sync, addr] = alloc.allocate_region(100, 3 * spacing, ExtraData::kNonBinary, 40);
    ASSERT_TRUE(addr.has_value());
    EXPECT_EQ(*addr, 200u);
}

TEST_F(SimpleTraceAllocatorFixture, ProgramIdsMemoryMapCleanedOnEviction) {
    extra_data_.resize(2);
    auto alloc = make_allocator(100);

    alloc.allocate_region(100, 0, ExtraData::kNonBinary, 42);
    alloc.add_region(ExtraData::kNonBinary, 42, 0);
    EXPECT_TRUE(alloc.get_region(ExtraData::kNonBinary, 42).has_value());

    // Evict the region.
    alloc.allocate_region(100, 1, ExtraData::kNonBinary, 99);

    // The old program_id entry should have been removed.
    EXPECT_FALSE(alloc.get_region(ExtraData::kNonBinary, 42).has_value());
}

TEST_F(SimpleTraceAllocatorFixture, ZeroCostEarlyExit) {
    // When a placement has zero cost (no overlapping regions), the search stops immediately.
    extra_data_.resize(2);
    auto alloc = make_allocator(1000);

    alloc.allocate_region(100, 0, ExtraData::kNonBinary, 10);  // [0, 100)

    // Second allocation should find zero-cost placement at [100, 200) and stop.
    auto [sync, addr] = alloc.allocate_region(100, 1, ExtraData::kNonBinary, 20);
    ASSERT_TRUE(addr.has_value());
    EXPECT_EQ(*addr, 100u);
    EXPECT_FALSE(sync.has_value());
}

TEST_F(SimpleTraceAllocatorFixture, StallAvoidanceIncreasesCost) {
    // Evicting a region with a very recent trace_idx should have a much higher cost than evicting
    // one with an older trace_idx (beyond desired_write_ahead).
    // desired_write_ahead = min(launch_msg_buffer_num_entries, 7) = min(8, 7) = 7.
    constexpr uint32_t desired_write_ahead = 7;

    uint32_t num_entries = desired_write_ahead + 2;
    extra_data_.resize(num_entries);

    auto alloc = make_allocator(200);

    // Allocate two regions that will be candidates for eviction.
    // Region at trace_idx = num_entries-3 (recent, within desired_write_ahead of trace_idx num_entries-1).
    // Region at trace_idx = 0 (old, outside desired_write_ahead).
    uint32_t old_idx = 0;
    uint32_t recent_idx = num_entries - 3;

    alloc.allocate_region(100, old_idx, ExtraData::kNonBinary, 10);     // [0, 100)
    alloc.allocate_region(100, recent_idx, ExtraData::kNonBinary, 20);  // [100, 200)

    // Allocate at trace_idx = num_entries-1. Both placements overlap one region each.
    // Region at old_idx: region_idx_diff = (num_entries-1) - 0 = num_entries-1 >= desired_write_ahead, no stall
    // penalty. Region at recent_idx: region_idx_diff = (num_entries-1) - recent_idx = 2 < desired_write_ahead, stall
    // penalty. Should prefer evicting the old region.
    uint32_t alloc_idx = num_entries - 1;
    auto [sync, addr] = alloc.allocate_region(100, alloc_idx, ExtraData::kNonBinary, 30);
    ASSERT_TRUE(addr.has_value());
    EXPECT_EQ(*addr, 0u);
}

}  // namespace tt::tt_metal
