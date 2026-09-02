// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Allocation behaviour of experimental::range_lockstep_allocation, which needs a device: the
// narrowed scan only exists in HYBRID mode, and its effect is only visible as whether a placement
// succeeds. The guards and the flag's survival through config conversions are host-side and live
// in test_range_lockstep_allocation_host.cpp.
//
// PlacesBesideAPerCoreHogWhenScoped and RefusesToPlaceBesideAPerCoreHogWhenNotScoped are the same
// scenario with the opt-in on and off, and are the pair to read to see what the flag changes.

#include <functional>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/experimental/per_core_allocation/mesh_buffer.hpp>
#include <tt-metalium/experimental/range_lockstep_allocation/buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include "tests/tt_metal/tt_metal/api/allocator/hybrid_allocator_fixture.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal {

namespace per_core = experimental::per_core_allocation;
namespace range_lockstep = experimental::range_lockstep_allocation;

// A lockstep buffer takes one address. By default the allocator keeps that address clear of
// per-core allocations on EVERY bank, because an op may reach the buffer on a core outside its
// own shard grid (a multicast writes to its whole rectangle). Opting into range lockstep narrows
// that to the cores the buffer occupies.
//
// `expect_range_lockstep` selects which behaviour is asserted, so the same scenario covers both:
// hog most of one core's L1 with a per-core allocation, then ask for the same size on a DIFFERENT
// core. Scoped to its own cores that fits; scanning chip-wide it does not.
//
// Goes through MeshBuffer: hybrid_device_allocators_ is only populated for mesh allocations, and
// the ranges are only gathered when it is non-empty. A buffer states its cores either through a
// shard spec or a distribution spec, and both must scope the scan, so each gets its own test --
// a failed allocation perturbs the allocator, so they cannot share one.
void run_lockstep_beside_per_core_hog(
    const std::shared_ptr<distributed::MeshDevice>& md,
    const std::function<BufferShardingArgs(const CoreCoord&)>& make_lockstep_args,
    bool expect_range_lockstep) {
    // Under LOCKSTEP the ranges are never gathered, so this would pass without proving anything.
    // The mode is latched at the first MetalContext construction, so skip loudly.
    if (!MetalContext::instance().rtoptions().get_allocator_mode_hybrid()) {
        GTEST_SKIP() << "HYBRID allocator mode is not active in this process (it is latched at the first "
                        "MetalContext construction); run this binary with TT_METAL_ALLOCATOR_MODE_HYBRID=1";
    }
    auto* device = md->get_devices()[0];
    ASSERT_GE(device->compute_with_storage_grid_size().x, 2u) << "Need two disjoint compute cores";

    const CoreCoord hogged_core(0, 0);
    const CoreCoord free_core(1, 0);

    // Over half of L1, so a chip-wide scan cannot fit the second allocation.
    const auto stats = device->allocator()->get_statistics(BufferType::L1);
    const DeviceAddr alloc_size =
        (stats.largest_free_block_bytes * 6 / 10) / HYBRID_TEST_PAGE_SIZE * HYBRID_TEST_PAGE_SIZE;
    ASSERT_GT(alloc_size, stats.largest_free_block_bytes / 2)
        << "Sized under half of L1; both allocations would fit even under a chip-wide scan";

    // Occupy most of hogged_core's L1 with a per-core allocation.
    auto hog_args = BufferShardingArgs(
        ShardSpecBuffer(CoreRangeSet(hogged_core), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1}),
        TensorMemoryLayout::HEIGHT_SHARDED);
    per_core::set_per_core_allocation(hog_args, true);
    const distributed::DeviceLocalBufferConfig hog_local_config{
        .page_size = alloc_size,
        .buffer_type = BufferType::L1,
        .sharding_args = hog_args,
        .bottom_up = false,
    };
    auto hog_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = alloc_size}, hog_local_config, md.get());
    ASSERT_TRUE(per_core::is_per_core_allocation(*hog_buffer));

    // free_core holds nothing, so the same size must still fit there.
    const distributed::DeviceLocalBufferConfig lockstep_local_config{
        .page_size = alloc_size,
        .buffer_type = BufferType::L1,
        .sharding_args = make_lockstep_args(free_core),
        .bottom_up = false,
    };
    auto allocate_lockstep = [&] {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = alloc_size}, lockstep_local_config, md.get());
    };

    if (!expect_range_lockstep) {
        // Default lockstep still avoids per-core allocations everywhere, so this cannot be placed.
        EXPECT_ANY_THROW(allocate_lockstep())
            << "Default lockstep placed on " << free_core.str() << " despite a per-core allocation on "
            << hogged_core.str() << "; the chip-wide scan is what makes a multicast past its own cores safe";
        return;
    }

    std::shared_ptr<distributed::MeshBuffer> lockstep_buffer;
    ASSERT_NO_THROW(lockstep_buffer = allocate_lockstep())
        << "Range lockstep allocation on " << free_core.str() << " was blocked by a per-core allocation on "
        << hogged_core.str();
    ASSERT_NE(lockstep_buffer, nullptr);
    EXPECT_TRUE(lockstep_buffer->is_allocated());
}

namespace {
BufferShardingArgs shard_spec_args(const CoreCoord& core) {
    return BufferShardingArgs(
        ShardSpecBuffer(CoreRangeSet(core), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1}),
        TensorMemoryLayout::HEIGHT_SHARDED);
}

// shard_spec_ is unset on this path, and cores_with_data() is what Buffer::num_cores() reports.
BufferShardingArgs distribution_spec_args(const CoreCoord& core) {
    return BufferShardingArgs(BufferDistributionSpec(Shape({1, 1}), Shape({1, 1}), std::vector<CoreCoord>{core}));
}
}  // namespace

TEST_F(HybridAllocatorTest, PlacesBesideAPerCoreHogWhenScoped) {
    run_lockstep_beside_per_core_hog(
        this->devices_[0],
        [](const CoreCoord& core) {
            auto args = shard_spec_args(core);
            range_lockstep::set_range_lockstep_allocation(args, true);
            return args;
        },
        /*expect_range_lockstep=*/true);
}

TEST_F(HybridAllocatorTest, PlacesBesideAPerCoreHogWhenScopedByADistributionSpec) {
    run_lockstep_beside_per_core_hog(
        this->devices_[0],
        [](const CoreCoord& core) {
            auto args = distribution_spec_args(core);
            range_lockstep::set_range_lockstep_allocation(args, true);
            return args;
        },
        /*expect_range_lockstep=*/true);
}

// Without the opt-in the chip-wide scan stays, which is what keeps a multicast past a buffer's

TEST_F(HybridAllocatorTest, RefusesToPlaceBesideAPerCoreHogWhenNotScoped) {
    run_lockstep_beside_per_core_hog(this->devices_[0], shard_spec_args, /*expect_range_lockstep=*/false);
}

// The mesh path and the direct path learn about per-core allocations through different
// mechanisms: a mesh allocator gathers them from the device allocators it was handed, while a
// device allocator has them in its own dependency graph. Narrowing only the first would make the
// same BufferShardingArgs behave as range lockstep through a mesh and as default lockstep through
// a device, so the dependency subtraction is scoped too. This exercises that second path.
TEST_F(HybridAllocatorTest, ScopesDependenciesOnADirectBufferCreate) {
    auto* device = this->devices_[0]->get_devices()[0];
    ASSERT_GE(device->compute_with_storage_grid_size().x, 2u);
    const CoreCoord hogged_core(0, 0);
    const CoreCoord free_core(1, 0);

    const auto stats = device->allocator()->get_statistics(BufferType::L1);
    const DeviceAddr alloc_size =
        (stats.largest_free_block_bytes * 6 / 10) / HYBRID_TEST_PAGE_SIZE * HYBRID_TEST_PAGE_SIZE;
    ASSERT_GT(alloc_size, stats.largest_free_block_bytes / 2)
        << "sized under half of L1; both allocations would fit even without scoping";

    auto single_core_args = [](const CoreCoord& core) {
        return BufferShardingArgs(
            ShardSpecBuffer(CoreRangeSet(core), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1}),
            TensorMemoryLayout::HEIGHT_SHARDED);
    };

    auto hog_args = single_core_args(hogged_core);
    per_core::set_per_core_allocation(hog_args, true);
    auto hog = Buffer::create(device, alloc_size, alloc_size, BufferType::L1, hog_args);
    ASSERT_TRUE(per_core::is_per_core_allocation(*hog));

    auto scoped_args = single_core_args(free_core);
    range_lockstep::set_range_lockstep_allocation(scoped_args, true);
    std::shared_ptr<Buffer> scoped;
    ASSERT_NO_THROW(scoped = Buffer::create(device, alloc_size, alloc_size, BufferType::L1, scoped_args))
        << "range lockstep on " << free_core.str() << " was blocked by a per-core allocation on " << hogged_core.str()
        << "; the dependency subtraction is not being scoped on the direct path";
    EXPECT_TRUE(scoped->is_allocated());
}

// Only the L1 branch of allocate_buffer reads the flag, so anywhere else it would be a no-op that
// is_range_lockstep_allocation() still reports as enabled.
TEST_F(HybridAllocatorTest, RejectsNonL1Buffers) {
    auto* device = this->devices_[0]->get_devices()[0];
    auto args = BufferShardingArgs(
        ShardSpecBuffer(CoreRangeSet(CoreCoord(0, 0)), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1}),
        TensorMemoryLayout::HEIGHT_SHARDED);
    range_lockstep::set_range_lockstep_allocation(args, true);
    EXPECT_ANY_THROW(Buffer::create(device, HYBRID_TEST_PAGE_SIZE, HYBRID_TEST_PAGE_SIZE, BufferType::DRAM, args));
}

}  // namespace tt::tt_metal
