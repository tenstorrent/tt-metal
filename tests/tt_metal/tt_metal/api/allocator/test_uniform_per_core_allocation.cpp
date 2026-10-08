// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Allocation behaviour of per_core_allocation::set_uniform_address: one address on every core of
// the shard grid, reserved only in those cores' per-core allocators. Needs a device in HYBRID mode,
// the only mode with per-core allocators.
//
// LeavesTheAddressFreeOnOtherCores is the test to read to see what it buys over range lockstep:
// a range-lockstep allocation sits in the lockstep allocator, which every core's per-core
// allocator avoids, so that full-bank allocation would fail.

#include <cstdint>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/experimental/per_core_allocation/global_semaphore.hpp>
#include <tt-metalium/experimental/per_core_allocation/mesh_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tests/tt_metal/tt_metal/api/allocator/hybrid_allocator_fixture.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal {

namespace {

namespace per_core = experimental::per_core_allocation;

bool hybrid_mode_active() { return MetalContext::instance().rtoptions().get_allocator_mode_hybrid(); }

#define SKIP_UNLESS_HYBRID()                                                                                 \
    if (!hybrid_mode_active()) {                                                                             \
        GTEST_SKIP() << "HYBRID allocator mode is not active in this process (it is latched at the first "   \
                        "MetalContext construction); run this binary with TT_METAL_ALLOCATOR_MODE_HYBRID=1"; \
    }

BufferShardingArgs per_core_args(const CoreRangeSet& cores, bool uniform) {
    const uint32_t num_cores = cores.num_cores();
    auto args = BufferShardingArgs(
        ShardSpecBuffer(cores, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_cores, 1}),
        TensorMemoryLayout::HEIGHT_SHARDED);
    per_core::set_per_core_allocation(args, true);
    per_core::set_uniform_address(args, uniform);
    return args;
}

// One `page_size` page on each core of `cores`.
std::shared_ptr<distributed::MeshBuffer> allocate_per_core(
    distributed::MeshDevice& md, const CoreRangeSet& cores, DeviceAddr page_size, bool uniform) {
    const distributed::DeviceLocalBufferConfig local_config{
        .page_size = page_size,
        .buffer_type = BufferType::L1,
        .sharding_args = per_core_args(cores, uniform),
        .bottom_up = false,
    };
    return distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = page_size * cores.num_cores()}, local_config, &md);
}

std::shared_ptr<distributed::MeshBuffer> allocate_lockstep(
    distributed::MeshDevice& md, const CoreCoord& core, DeviceAddr page_size) {
    const distributed::DeviceLocalBufferConfig local_config{
        .page_size = page_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(
            ShardSpecBuffer(CoreRangeSet(core), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1}),
            TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false,
    };
    return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = page_size}, local_config, &md);
}

// The largest page that fits on a core with nothing else allocated, rounded to the test page.
DeviceAddr whole_bank(distributed::MeshDevice& md) {
    const auto stats = md.allocator()->get_statistics(BufferType::L1);
    return stats.largest_free_block_bytes / HYBRID_TEST_PAGE_SIZE * HYBRID_TEST_PAGE_SIZE;
}

const Buffer& device_buffer(const distributed::MeshBuffer& buffer) {
    return *buffer.get_device_buffer(distributed::MeshCoordinate(0, 0));
}

}  // namespace

TEST_F(HybridAllocatorTest, UniformPerCoreTakesOneAddressOnEveryCore) {
    SKIP_UNLESS_HYBRID();
    auto& md = *this->devices_[0];
    ASSERT_GE(md.compute_with_storage_grid_size().x, 3u);
    const CoreCoord skewed(1, 0);
    const CoreRangeSet cores(CoreRange(CoreCoord(0, 0), CoreCoord(2, 0)));

    // Take the top of one core only, so a per-core allocation would place that core lower than
    // the others; the uniform buffer must instead find one address free on all three.
    auto skew = allocate_per_core(md, CoreRangeSet(skewed), HYBRID_TEST_PAGE_SIZE, /*uniform=*/false);
    const DeviceAddr skew_address = per_core::get_per_core_address(device_buffer(*skew), skewed);

    auto buffer = allocate_per_core(md, cores, HYBRID_TEST_PAGE_SIZE, /*uniform=*/true);
    ASSERT_TRUE(per_core::is_uniform_address(device_buffer(*buffer)));
    const DeviceAddr address = buffer->address();
    EXPECT_NE(address, 0u) << "A uniform per-core MeshBuffer must report its one address";
    for (const auto& core : corerange_to_cores(cores)) {
        EXPECT_EQ(per_core::get_per_core_address(device_buffer(*buffer), core), address) << "core " << core.str();
    }
    EXPECT_TRUE(address + HYBRID_TEST_PAGE_SIZE <= skew_address || skew_address + HYBRID_TEST_PAGE_SIZE <= address)
        << "Uniform address " << address << " overlaps the per-core allocation at " << skew_address << " on "
        << skewed.str();
}

TEST_F(HybridAllocatorTest, UniformPerCoreLeavesTheAddressFreeOnOtherCores) {
    SKIP_UNLESS_HYBRID();
    auto& md = *this->devices_[0];
    ASSERT_GE(md.compute_with_storage_grid_size().x, 2u);
    const DeviceAddr bank = whole_bank(md);
    auto small = allocate_per_core(md, CoreRangeSet(CoreCoord(0, 0)), HYBRID_TEST_PAGE_SIZE, /*uniform=*/true);

    // The whole bank still fits on a core outside the uniform buffer's grid.
    std::shared_ptr<distributed::MeshBuffer> other;
    ASSERT_NO_THROW(other = allocate_per_core(md, CoreRangeSet(CoreCoord(1, 0)), bank, /*uniform=*/false))
        << "A uniform per-core buffer on (0,0) blocked a whole-bank per-core allocation on (1,0)";
    EXPECT_TRUE(other->is_allocated());
}

TEST_F(HybridAllocatorTest, UniformPerCoreIsAvoidedOnItsOwnCores) {
    SKIP_UNLESS_HYBRID();
    auto& md = *this->devices_[0];
    const DeviceAddr bank = whole_bank(md);
    auto small = allocate_per_core(md, CoreRangeSet(CoreCoord(0, 0)), HYBRID_TEST_PAGE_SIZE, /*uniform=*/true);
    EXPECT_ANY_THROW(allocate_per_core(md, CoreRangeSet(CoreCoord(0, 0)), bank, /*uniform=*/false))
        << "A whole-bank per-core allocation on (0,0) overlapped the uniform per-core buffer there";
}

TEST_F(HybridAllocatorTest, UniformPerCoreIsAvoidedByLockstep) {
    SKIP_UNLESS_HYBRID();
    auto& md = *this->devices_[0];
    ASSERT_GE(md.compute_with_storage_grid_size().x, 2u);
    const DeviceAddr bank = whole_bank(md);
    auto small = allocate_per_core(md, CoreRangeSet(CoreCoord(0, 0)), HYBRID_TEST_PAGE_SIZE, /*uniform=*/true);
    // Default lockstep keeps its address clear on every core, including the uniform buffer's.
    EXPECT_ANY_THROW(allocate_lockstep(md, CoreCoord(1, 0), bank))
        << "A whole-bank lockstep allocation overlapped a uniform per-core buffer";
}

TEST_F(HybridAllocatorTest, UniformPerCoreBuffersOnDisjointCoresShareAnAddress) {
    SKIP_UNLESS_HYBRID();
    auto& md = *this->devices_[0];
    ASSERT_GE(md.compute_with_storage_grid_size().x, 2u);
    auto first = allocate_per_core(md, CoreRangeSet(CoreCoord(0, 0)), HYBRID_TEST_PAGE_SIZE, /*uniform=*/true);
    auto second = allocate_per_core(md, CoreRangeSet(CoreCoord(1, 0)), HYBRID_TEST_PAGE_SIZE, /*uniform=*/true);
    EXPECT_EQ(first->address(), second->address())
        << "Two uniform per-core buffers on disjoint cores should be able to take the same address";
}

TEST_F(HybridAllocatorTest, UniformPerCoreReleasesEveryCoreOnDeallocation) {
    SKIP_UNLESS_HYBRID();
    auto& md = *this->devices_[0];
    ASSERT_GE(md.compute_with_storage_grid_size().x, 2u);
    const DeviceAddr bank = whole_bank(md);
    const CoreRangeSet cores(CoreRange(CoreCoord(0, 0), CoreCoord(1, 0)));
    auto small = allocate_per_core(md, cores, HYBRID_TEST_PAGE_SIZE, /*uniform=*/true);
    small->deallocate();
    small.reset();

    std::shared_ptr<distributed::MeshBuffer> whole;
    ASSERT_NO_THROW(whole = allocate_per_core(md, cores, bank, /*uniform=*/false))
        << "A freed uniform per-core buffer still held its address on one of its cores";
    EXPECT_TRUE(whole->is_allocated());
}

TEST_F(HybridAllocatorTest, ScopedGlobalSemaphoreIsReservedOnlyOnItsCores) {
    SKIP_UNLESS_HYBRID();
    auto& md = *this->devices_[0];
    ASSERT_GE(md.compute_with_storage_grid_size().x, 3u);
    const DeviceAddr bank = whole_bank(md);
    const CoreRangeSet cores(CoreRange(CoreCoord(0, 0), CoreCoord(1, 0)));
    constexpr uint32_t initial_value = 0x5e3a;

    auto semaphore = per_core::create_global_semaphore(md, cores, initial_value);
    const DeviceAddr address = semaphore.address();
    ASSERT_NE(address, 0u);

    // The initial value went out through the mesh write path; read it back at the one address.
    auto* device = md.get_devices()[0];
    for (const auto& core : corerange_to_cores(cores)) {
        std::vector<uint32_t> value;
        ASSERT_TRUE(detail::ReadFromDeviceL1(device, core, address, sizeof(uint32_t), value));
        EXPECT_EQ(value.at(0), initial_value) << "core " << core.str();
    }

    std::shared_ptr<distributed::MeshBuffer> other;
    ASSERT_NO_THROW(other = allocate_per_core(md, CoreRangeSet(CoreCoord(2, 0)), bank, /*uniform=*/false))
        << "A scoped global semaphore on (0,0)-(1,0) blocked a whole-bank per-core allocation on (2,0)";
}

TEST_F(HybridAllocatorTest, RejectsUniformAddressWithoutPerCoreAllocation) {
    auto args = BufferShardingArgs(
        ShardSpecBuffer(CoreRangeSet(CoreCoord(0, 0)), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1}),
        TensorMemoryLayout::HEIGHT_SHARDED);
    EXPECT_ANY_THROW(per_core::set_uniform_address(args, true));
    per_core::set_per_core_allocation(args, true);
    per_core::set_uniform_address(args, true);
    per_core::set_per_core_allocation(args, false);
    EXPECT_FALSE(per_core::is_uniform_address(args)) << "Turning per-core off must also turn the uniform address off";
}

}  // namespace tt::tt_metal
