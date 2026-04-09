// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "tt_metal/impl/dispatch/simple_trace_allocator.hpp"
#include "tt_metal/impl/trace/trace_node.hpp"
#include "tests/tt_metal/tt_metal/common/mesh_dispatch_fixture.hpp"

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

// NOLINTBEGIN(cppcoreguidelines-virtual-class-destructor)
class SimpleTraceAllocatorDeviceFixture : public MeshDispatchFixture {
protected:
    struct ProgramSpec {
        uint32_t nonbinary_size = 0;
        uint32_t binary_size = 0;
    };

    enum class BinaryPlacement { InConfig, FixedAddress };

    const Hal& hal_ = MetalContext::instance().hal();

    std::optional<uint32_t> first_core_type_index(BinaryPlacement placement) const {
        for (uint32_t index = 0; index < hal_.get_programmable_core_type_count(); ++index) {
            bool binary_in_config =
                hal_.get_core_kernel_stored_in_config_buffer(hal_.get_programmable_core_type(index));
            if ((placement == BinaryPlacement::InConfig && binary_in_config) ||
                (placement == BinaryPlacement::FixedAddress && !binary_in_config)) {
                return index;
            }
        }
        return std::nullopt;
    }

    std::vector<ProgramSpec> make_program_specs(
        uint32_t nonbinary_size, uint32_t binary_size, BinaryPlacement placement) const {
        std::vector<ProgramSpec> specs(hal_.get_programmable_core_type_count());
        for (uint32_t index = 0; index < specs.size(); ++index) {
            bool binary_in_config =
                hal_.get_core_kernel_stored_in_config_buffer(hal_.get_programmable_core_type(index));
            if ((placement == BinaryPlacement::InConfig && binary_in_config) ||
                (placement == BinaryPlacement::FixedAddress && !binary_in_config)) {
                specs[index] = {.nonbinary_size = nonbinary_size, .binary_size = binary_size};
            }
        }
        return specs;
    }

    std::shared_ptr<detail::ProgramImpl> make_program(const std::vector<ProgramSpec>& specs) const {
        TT_FATAL(
            specs.size() == hal_.get_programmable_core_type_count(),
            "Expected {} program specs, got {}",
            hal_.get_programmable_core_type_count(),
            specs.size());

        auto program = std::make_shared<detail::ProgramImpl>();
        auto& program_config_sizes = program->get_program_config_sizes();
        for (uint32_t index = 0; index < specs.size(); ++index) {
            auto& program_config = program->get_program_config(index);
            program_config = {};
            program_config.kernel_text_offset = specs[index].nonbinary_size;
            program_config.kernel_text_size = specs[index].binary_size;

            bool binary_in_config =
                hal_.get_core_kernel_stored_in_config_buffer(hal_.get_programmable_core_type(index));
            program_config_sizes[index] =
                specs[index].nonbinary_size + (binary_in_config ? specs[index].binary_size : 0);
        }
        return program;
    }

    TraceNode make_trace_node(
        const std::vector<ProgramSpec>& specs,
        SubDeviceId sub_device_id,
        uint32_t num_workers,
        std::shared_ptr<detail::ProgramImpl> program = nullptr) const {
        program = program ? std::move(program) : make_program(specs);

        TraceNode node{};
        node.program = std::move(program);
        node.program_runtime_id = static_cast<uint32_t>(node.program->get_id());
        node.sub_device_id = sub_device_id;
        node.num_workers = num_workers;
        return node;
    }

    SimpleTraceAllocator make_allocator(uint32_t ringbuffer_size, uint32_t ringbuffer_start = 0) const {
        std::vector<SimpleTraceAllocator::RingbufferConfig> configs;
        configs.reserve(hal_.get_programmable_core_type_count());
        for (uint32_t index = 0; index < hal_.get_programmable_core_type_count(); ++index) {
            configs.push_back({.start = ringbuffer_start, .size = ringbuffer_size});
        }
        return SimpleTraceAllocator(configs);
    }

    std::vector<TraceNode*> make_trace_node_ptrs(std::vector<TraceNode>& trace_nodes) const {
        std::vector<TraceNode*> ptrs;
        ptrs.reserve(trace_nodes.size());
        for (auto& trace_node : trace_nodes) {
            ptrs.push_back(&trace_node);
        }
        return ptrs;
    }

    void populate_kernel_groups(detail::ProgramImpl& program, BinaryPlacement placement) const {
        for (uint32_t index = 0; index < hal_.get_programmable_core_type_count(); ++index) {
            bool binary_in_config =
                hal_.get_core_kernel_stored_in_config_buffer(hal_.get_programmable_core_type(index));
            if ((placement == BinaryPlacement::FixedAddress && !binary_in_config) ||
                (placement == BinaryPlacement::InConfig && binary_in_config)) {
                program.get_kernel_groups(index).push_back(nullptr);
            }
        }
    }

    void allocate_on_subdevice(
        SimpleTraceAllocator& allocator, std::vector<TraceNode>& trace_nodes, SubDeviceId sub_device_id) const {
        auto trace_node_ptrs = make_trace_node_ptrs(trace_nodes);
        allocator.extra_data_.clear();
        allocator.extra_data_.resize(trace_nodes.size());
        allocator.allocate_trace_programs_on_subdevice(hal_, trace_node_ptrs, sub_device_id);
    }
};
// NOLINTEND(cppcoreguidelines-virtual-class-destructor)

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
    extra_data_[0].next_use_idx[ExtraData::kNonBinary] = 2 * spacing;  // Next use IS the current allocation.
    extra_data_[spacing].next_use_idx[ExtraData::kNonBinary] = (3 * spacing) - 1;  // Far away (low Belady cost).

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
    extra_data_[0].next_use_idx[ExtraData::kNonBinary] = (3 * spacing) + 2;            // distance 2 (high cost)
    extra_data_[spacing].next_use_idx[ExtraData::kNonBinary] = (3 * spacing) + 1;      // distance 1 (highest cost)
    extra_data_[2 * spacing].next_use_idx[ExtraData::kNonBinary] = (3 * spacing) + 3;  // distance 3 (lowest cost)

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

TEST_F(SimpleTraceAllocatorDeviceFixture, SingleProgram) {
    auto core_index = first_core_type_index(BinaryPlacement::InConfig);
    if (!core_index.has_value()) {
        GTEST_SKIP() << "No core type stores kernels in the config buffer on this architecture.";
    }

    auto specs = make_program_specs(32, 16, BinaryPlacement::InConfig);
    std::vector<TraceNode> trace_nodes = {make_trace_node(specs, SubDeviceId{0}, 3)};
    auto trace_node_ptrs = make_trace_node_ptrs(trace_nodes);

    auto allocator = make_allocator(256);
    allocator.allocate_trace_programs(hal_, trace_node_ptrs);

    ASSERT_EQ(
        trace_nodes[0].dispatch_metadata.nonbinary_kernel_config_addrs.size(), hal_.get_programmable_core_type_count());
    ASSERT_EQ(
        trace_nodes[0].dispatch_metadata.binary_kernel_config_addrs.size(), hal_.get_programmable_core_type_count());
    EXPECT_EQ(trace_nodes[0].dispatch_metadata.nonbinary_kernel_config_addrs[*core_index].addr, 0u);
    EXPECT_EQ(trace_nodes[0].dispatch_metadata.binary_kernel_config_addrs[*core_index].addr, 32u);
    EXPECT_TRUE(trace_nodes[0].dispatch_metadata.send_binary);
    EXPECT_EQ(trace_nodes[0].dispatch_metadata.sync_count, 0u);
    EXPECT_TRUE(trace_nodes[0].dispatch_metadata.stall_first);
    EXPECT_FALSE(trace_nodes[0].dispatch_metadata.stall_before_program);
}

TEST_F(SimpleTraceAllocatorDeviceFixture, TwoDistinctPrograms) {
    auto core_index = first_core_type_index(BinaryPlacement::InConfig);
    if (!core_index.has_value()) {
        GTEST_SKIP() << "No core type stores kernels in the config buffer on this architecture.";
    }

    auto specs = make_program_specs(32, 16, BinaryPlacement::InConfig);
    std::vector<TraceNode> trace_nodes = {
        make_trace_node(specs, SubDeviceId{0}, 1), make_trace_node(specs, SubDeviceId{0}, 1)};
    auto trace_node_ptrs = make_trace_node_ptrs(trace_nodes);

    auto allocator = make_allocator(256);
    allocator.allocate_trace_programs(hal_, trace_node_ptrs);

    EXPECT_EQ(trace_nodes[0].dispatch_metadata.nonbinary_kernel_config_addrs[*core_index].addr, 0u);
    EXPECT_EQ(trace_nodes[0].dispatch_metadata.binary_kernel_config_addrs[*core_index].addr, 32u);
    EXPECT_EQ(trace_nodes[1].dispatch_metadata.nonbinary_kernel_config_addrs[*core_index].addr, 48u);
    EXPECT_EQ(trace_nodes[1].dispatch_metadata.binary_kernel_config_addrs[*core_index].addr, 80u);
    EXPECT_TRUE(trace_nodes[1].dispatch_metadata.send_binary);
}

TEST_F(SimpleTraceAllocatorDeviceFixture, SameProgramBinaryCached) {
    auto core_index = first_core_type_index(BinaryPlacement::InConfig);
    if (!core_index.has_value()) {
        GTEST_SKIP() << "No core type stores kernels in the config buffer on this architecture.";
    }

    auto specs = make_program_specs(32, 16, BinaryPlacement::InConfig);
    auto shared_program = make_program(specs);
    std::vector<TraceNode> trace_nodes = {
        make_trace_node(specs, SubDeviceId{0}, 1, shared_program),
        make_trace_node(specs, SubDeviceId{0}, 1, shared_program)};
    auto trace_node_ptrs = make_trace_node_ptrs(trace_nodes);

    auto allocator = make_allocator(256);
    allocator.allocate_trace_programs(hal_, trace_node_ptrs);

    EXPECT_TRUE(trace_nodes[0].dispatch_metadata.send_binary);
    EXPECT_FALSE(trace_nodes[1].dispatch_metadata.send_binary);
    EXPECT_EQ(
        trace_nodes[0].dispatch_metadata.binary_kernel_config_addrs[*core_index].addr,
        trace_nodes[1].dispatch_metadata.binary_kernel_config_addrs[*core_index].addr);
    EXPECT_EQ(trace_nodes[1].dispatch_metadata.nonbinary_kernel_config_addrs[*core_index].addr, 48u);
}

TEST_F(SimpleTraceAllocatorDeviceFixture, BinaryEvictionRetriesAfterReset) {
    auto core_index = first_core_type_index(BinaryPlacement::InConfig);
    if (!core_index.has_value()) {
        GTEST_SKIP() << "No core type stores kernels in the config buffer on this architecture.";
    }

    auto small_binary_specs = make_program_specs(0, 30, BinaryPlacement::InConfig);
    auto medium_binary_specs = make_program_specs(0, 40, BinaryPlacement::InConfig);
    auto large_program_specs = make_program_specs(60, 40, BinaryPlacement::InConfig);
    auto reused_program = make_program(small_binary_specs);
    std::vector<TraceNode> trace_nodes = {
        make_trace_node(small_binary_specs, SubDeviceId{0}, 1, reused_program),
        make_trace_node(medium_binary_specs, SubDeviceId{0}, 1),
        make_trace_node(large_program_specs, SubDeviceId{0}, 1),
        make_trace_node(small_binary_specs, SubDeviceId{0}, 1, reused_program)};
    auto trace_node_ptrs = make_trace_node_ptrs(trace_nodes);

    auto allocator = make_allocator(100);
    allocator.allocate_trace_programs(hal_, trace_node_ptrs);

    EXPECT_EQ(trace_nodes[2].dispatch_metadata.nonbinary_kernel_config_addrs[*core_index].addr, 0u);
    EXPECT_EQ(trace_nodes[2].dispatch_metadata.binary_kernel_config_addrs[*core_index].addr, 60u);
    EXPECT_TRUE(trace_nodes[2].dispatch_metadata.send_binary);
    EXPECT_EQ(trace_nodes[2].dispatch_metadata.sync_count, 2u);
    EXPECT_TRUE(trace_nodes[2].dispatch_metadata.stall_first);
}

TEST_F(SimpleTraceAllocatorDeviceFixture, NonBinaryEvictionSetsStallFirst) {
    auto core_index = first_core_type_index(BinaryPlacement::InConfig);
    if (!core_index.has_value()) {
        GTEST_SKIP() << "No core type stores kernels in the config buffer on this architecture.";
    }

    auto specs = make_program_specs(80, 0, BinaryPlacement::InConfig);
    std::vector<TraceNode> trace_nodes = {
        make_trace_node(specs, SubDeviceId{0}, 2), make_trace_node(specs, SubDeviceId{0}, 5)};
    auto trace_node_ptrs = make_trace_node_ptrs(trace_nodes);

    auto allocator = make_allocator(100);
    allocator.allocate_trace_programs(hal_, trace_node_ptrs);

    EXPECT_EQ(trace_nodes[1].dispatch_metadata.nonbinary_kernel_config_addrs[*core_index].addr, 0u);
    EXPECT_EQ(trace_nodes[1].dispatch_metadata.sync_count, 2u);
    EXPECT_TRUE(trace_nodes[1].dispatch_metadata.stall_first);
    EXPECT_FALSE(trace_nodes[1].dispatch_metadata.stall_before_program);
}

TEST_F(SimpleTraceAllocatorDeviceFixture, BinaryOnlyEvictionSetsStallBeforeProgram) {
    auto core_index = first_core_type_index(BinaryPlacement::InConfig);
    if (!core_index.has_value()) {
        GTEST_SKIP() << "No core type stores kernels in the config buffer on this architecture.";
    }

    auto specs = make_program_specs(0, 40, BinaryPlacement::InConfig);
    std::vector<TraceNode> trace_nodes = {
        make_trace_node(specs, SubDeviceId{0}, 2), make_trace_node(specs, SubDeviceId{0}, 5)};
    auto trace_node_ptrs = make_trace_node_ptrs(trace_nodes);

    auto allocator = make_allocator(40);
    allocator.allocate_trace_programs(hal_, trace_node_ptrs);

    EXPECT_EQ(trace_nodes[1].dispatch_metadata.binary_kernel_config_addrs[*core_index].addr, 0u);
    EXPECT_EQ(trace_nodes[1].dispatch_metadata.sync_count, 2u);
    EXPECT_FALSE(trace_nodes[1].dispatch_metadata.stall_first);
    EXPECT_TRUE(trace_nodes[1].dispatch_metadata.stall_before_program);
}

TEST_F(SimpleTraceAllocatorDeviceFixture, FixedL1AddressBinarySync) {
    auto core_index = first_core_type_index(BinaryPlacement::FixedAddress);
    if (!core_index.has_value()) {
        GTEST_SKIP() << "All programmable core types store kernels in the config buffer on this architecture.";
    }

    auto specs = make_program_specs(0, 0, BinaryPlacement::FixedAddress);
    auto program0 = make_program(specs);
    auto program1 = make_program(specs);
    populate_kernel_groups(*program0, BinaryPlacement::FixedAddress);
    populate_kernel_groups(*program1, BinaryPlacement::FixedAddress);
    std::vector<TraceNode> trace_nodes = {
        make_trace_node(specs, SubDeviceId{0}, 3, program0),
        make_trace_node(specs, SubDeviceId{0}, 4, program1)};
    auto trace_node_ptrs = make_trace_node_ptrs(trace_nodes);

    auto allocator = make_allocator(64);
    allocator.allocate_trace_programs(hal_, trace_node_ptrs);

    EXPECT_TRUE(trace_nodes[0].dispatch_metadata.send_binary);
    EXPECT_TRUE(trace_nodes[1].dispatch_metadata.send_binary);
    EXPECT_EQ(trace_nodes[1].dispatch_metadata.binary_kernel_config_addrs[*core_index].addr, 0u);
    EXPECT_EQ(trace_nodes[1].dispatch_metadata.sync_count, 3u);
    EXPECT_FALSE(trace_nodes[1].dispatch_metadata.stall_first);
    EXPECT_TRUE(trace_nodes[1].dispatch_metadata.stall_before_program);
}

TEST_F(SimpleTraceAllocatorDeviceFixture, LaunchWindowOverflow) {
    constexpr uint32_t max_queued_programs = dev_msgs::launch_msg_buffer_num_entries - 1;
    std::vector<TraceNode> trace_nodes;
    trace_nodes.reserve(max_queued_programs + 1);
    auto specs = std::vector<ProgramSpec>(hal_.get_programmable_core_type_count());
    for (uint32_t index = 0; index < max_queued_programs + 1; ++index) {
        trace_nodes.push_back(make_trace_node(specs, SubDeviceId{0}, 1));
    }
    auto trace_node_ptrs = make_trace_node_ptrs(trace_nodes);

    auto allocator = make_allocator(64);
    allocator.allocate_trace_programs(hal_, trace_node_ptrs);

    EXPECT_TRUE(trace_nodes[0].dispatch_metadata.stall_first);
    EXPECT_FALSE(trace_nodes[max_queued_programs].dispatch_metadata.stall_first);
    EXPECT_TRUE(trace_nodes[max_queued_programs].dispatch_metadata.stall_before_program);
    EXPECT_EQ(trace_nodes[max_queued_programs].dispatch_metadata.sync_count, 1u);
}

TEST_F(SimpleTraceAllocatorDeviceFixture, SubDeviceFiltering) {
    auto core_index = first_core_type_index(BinaryPlacement::InConfig);
    if (!core_index.has_value()) {
        GTEST_SKIP() << "No core type stores kernels in the config buffer on this architecture.";
    }

    auto specs = make_program_specs(32, 16, BinaryPlacement::InConfig);
    std::vector<TraceNode> trace_nodes = {
        make_trace_node(specs, SubDeviceId{0}, 1), make_trace_node(specs, SubDeviceId{1}, 1)};
    trace_nodes[1].dispatch_metadata.send_binary = false;
    trace_nodes[1].dispatch_metadata.sync_count = 99;
    trace_nodes[1].dispatch_metadata.stall_first = true;

    auto allocator = make_allocator(256);
    allocate_on_subdevice(allocator, trace_nodes, SubDeviceId{0});

    EXPECT_EQ(trace_nodes[0].dispatch_metadata.nonbinary_kernel_config_addrs[*core_index].addr, 0u);
    EXPECT_TRUE(trace_nodes[0].dispatch_metadata.send_binary);
    EXPECT_EQ(trace_nodes[0].dispatch_metadata.sync_count, 0u);
    EXPECT_TRUE(trace_nodes[0].dispatch_metadata.stall_first);

    EXPECT_FALSE(trace_nodes[1].dispatch_metadata.send_binary);
    EXPECT_EQ(trace_nodes[1].dispatch_metadata.sync_count, 99u);
    EXPECT_TRUE(trace_nodes[1].dispatch_metadata.stall_first);
    EXPECT_TRUE(trace_nodes[1].dispatch_metadata.binary_kernel_config_addrs.empty());
    EXPECT_TRUE(trace_nodes[1].dispatch_metadata.nonbinary_kernel_config_addrs.empty());
}

TEST_F(SimpleTraceAllocatorDeviceFixture, LargeTraceSequence) {
    auto core_index = first_core_type_index(BinaryPlacement::InConfig);
    if (!core_index.has_value()) {
        GTEST_SKIP() << "No core type stores kernels in the config buffer on this architecture.";
    }

    auto specs = make_program_specs(24, 16, BinaryPlacement::InConfig);
    auto program_a = make_program(specs);
    auto program_b = make_program(specs);
    auto program_c = make_program(specs);

    std::vector<TraceNode> trace_nodes;
    trace_nodes.reserve(20);
    for (uint32_t index = 0; index < 20; ++index) {
        std::shared_ptr<detail::ProgramImpl> program = program_c;
        if (index % 3 == 0) {
            program = program_a;
        } else if (index % 3 == 1) {
            program = program_b;
        }
        trace_nodes.push_back(make_trace_node(specs, SubDeviceId{0}, (index % 4) + 1, program));
    }
    auto trace_node_ptrs = make_trace_node_ptrs(trace_nodes);

    auto allocator = make_allocator(128);
    allocator.allocate_trace_programs(hal_, trace_node_ptrs);

    uint32_t workers_completed_before = 0;
    for (size_t index = 0; index < trace_nodes.size(); ++index) {
        const auto& metadata = trace_nodes[index].dispatch_metadata;
        ASSERT_EQ(metadata.nonbinary_kernel_config_addrs.size(), hal_.get_programmable_core_type_count());
        ASSERT_EQ(metadata.binary_kernel_config_addrs.size(), hal_.get_programmable_core_type_count());
        EXPECT_LT(metadata.nonbinary_kernel_config_addrs[*core_index].addr, 128u);
        EXPECT_LT(metadata.binary_kernel_config_addrs[*core_index].addr, 128u);
        EXPECT_LE(metadata.sync_count, workers_completed_before);
        workers_completed_before += trace_nodes[index].num_workers;
    }
}

}  // namespace tt::tt_metal
