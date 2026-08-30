// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

#include <cstdint>
#include <memory>
#include <set>

#include "device_fixture.hpp"
#include "gtest/gtest.h"

// A sharded buffer's cores are validated when the buffer is constructed. Nothing further down the
// allocation path can do it: the allocator is handed a shard *count*, not the coordinates, so an
// out-of-range core survives allocation and is only caught later by whichever op happens to resolve
// a bank id for it -- if any does. These tests pin the check to construction.
//
// The invalid coordinates are derived from the device rather than hardcoded, because both grids
// shrink under harvesting: a core that is out of range on a harvested part is legal on a stock one.
//
// The one core that has no bank and is still legal is a claimed service core; that case needs the
// arch and fast-dispatch gating those fixtures carry, so it lives in
// tests/tt_metal/tt_metal/dispatch/test_service_core_manager.cpp.

namespace tt::tt_metal {
namespace {

constexpr uint32_t kPageSize = 1024;

ShardedBufferConfig one_shard_config(IDevice* device, BufferType buffer_type, const CoreCoord& core) {
    // One shard, on one core. Size is irrelevant to the core validation, but must be page-aligned.
    const CoreRangeSet grid(std::set<CoreRange>({CoreRange(core, core)}));
    return ShardedBufferConfig{
        .device = device,
        .size = kPageSize,
        .page_size = kPageSize,
        .buffer_type = buffer_type,
        .buffer_layout = TensorMemoryLayout::WIDTH_SHARDED,
        .shard_parameters = ShardSpecBuffer(
            grid,
            {1, kPageSize / sizeof(uint32_t)},
            ShardOrientation::ROW_MAJOR,
            {1, 1},
            {1, kPageSize / sizeof(uint32_t)})};
}

std::shared_ptr<Buffer> make_width_sharded_buffer(IDevice* device, BufferType buffer_type, const CoreCoord& core) {
    return CreateBuffer(one_shard_config(device, buffer_type, core));
}

// Constructs the buffer without allocating it: the explicit-address overload skips allocate_impl().
// L1_SMALL needs this because the fixture opens the device with the default l1_small_size of 0, so
// an allocation would fail on the missing small region alone and say nothing about the shard grid.
// It is also the shape of the path this first went wrong on -- graph capture hooks the allocation
// out, so an L1_SMALL config tensor there is constructed and never allocated.
std::shared_ptr<Buffer> make_unallocated_width_sharded_buffer(
    IDevice* device, BufferType buffer_type, const CoreCoord& core) {
    return CreateBuffer(
        one_shard_config(device, buffer_type, core), device->allocator()->get_base_allocator_addr(HalMemType::L1));
}

}  // namespace

TEST_F(AnyDispatchMeshDeviceSingleCardFixture, ShardGridValidationL1RejectsCoreOutsideGrid) {
    for (auto& mesh_device : this->devices_) {
        auto* device = mesh_device->get_devices()[0];
        // Shards live on the compute-with-storage rectangle; a core outside it has no L1 bank.
        const CoreCoord grid = device->compute_with_storage_grid_size();

        EXPECT_ANY_THROW(make_width_sharded_buffer(device, BufferType::L1, CoreCoord(grid.x, 0)));
        EXPECT_ANY_THROW(make_width_sharded_buffer(device, BufferType::L1, CoreCoord(0, grid.y)));
    }
}

TEST_F(AnyDispatchMeshDeviceSingleCardFixture, ShardGridValidationL1AcceptsCoreInsideGrid) {
    for (auto& mesh_device : this->devices_) {
        auto* device = mesh_device->get_devices()[0];
        const CoreCoord grid = device->compute_with_storage_grid_size();

        EXPECT_NO_THROW(make_width_sharded_buffer(device, BufferType::L1, CoreCoord(grid.x - 1, grid.y - 1)));
    }
}

TEST_F(AnyDispatchMeshDeviceSingleCardFixture, ShardGridValidationL1SmallAcceptsCoreInsideGrid) {
    for (auto& mesh_device : this->devices_) {
        auto* device = mesh_device->get_devices()[0];
        const CoreCoord grid = device->compute_with_storage_grid_size();

        // A compute core is a legal L1_SMALL shard core whether or not this device has a small
        // region -- the coordinate is what is under test, and the L1_SMALL bank map is empty when
        // l1_small_size is 0, so consulting it directly would reject this.
        EXPECT_NO_THROW(
            make_unallocated_width_sharded_buffer(device, BufferType::L1_SMALL, CoreCoord(grid.x - 1, grid.y - 1)));
    }
}

TEST_F(AnyDispatchMeshDeviceSingleCardFixture, ShardGridValidationL1SmallRejectsCoreOutsideGrid) {
    for (auto& mesh_device : this->devices_) {
        auto* device = mesh_device->get_devices()[0];
        const CoreCoord grid = device->compute_with_storage_grid_size();

        // L1_SMALL banks sit on the same cores as L1 banks, so the grid bound still applies.
        EXPECT_ANY_THROW(make_unallocated_width_sharded_buffer(device, BufferType::L1_SMALL, CoreCoord(grid.x, 0)));
        EXPECT_ANY_THROW(make_unallocated_width_sharded_buffer(device, BufferType::L1_SMALL, CoreCoord(0, grid.y)));
    }
}

TEST_F(AnyDispatchMeshDeviceSingleCardFixture, ShardGridValidationDramRejectsCoreBeyondBankCount) {
    for (auto& mesh_device : this->devices_) {
        auto* device = mesh_device->get_devices()[0];
        // bank_id == logical x, so x == the bank count names a bank that does not exist. A harvested
        // device has fewer banks here than an unharvested one.
        const uint32_t num_banks = device->dram_grid_size().x;

        EXPECT_ANY_THROW(make_width_sharded_buffer(device, BufferType::DRAM, CoreCoord(num_banks, 0)));
    }
}

TEST_F(AnyDispatchMeshDeviceSingleCardFixture, ShardGridValidationDramRejectsCoreOffRowZero) {
    for (auto& mesh_device : this->devices_) {
        auto* device = mesh_device->get_devices()[0];
        // (0, 1) is a real DRAM coordinate -- logical y indexes a view's subchannels -- but banks are
        // keyed {bank_id, 0}, so sharding there would alias onto bank 0.
        EXPECT_ANY_THROW(make_width_sharded_buffer(device, BufferType::DRAM, CoreCoord(0, 1)));
    }
}

TEST_F(AnyDispatchMeshDeviceSingleCardFixture, ShardGridValidationDramAcceptsCoreWithinBankCount) {
    for (auto& mesh_device : this->devices_) {
        auto* device = mesh_device->get_devices()[0];
        const uint32_t num_banks = device->dram_grid_size().x;

        EXPECT_NO_THROW(make_width_sharded_buffer(device, BufferType::DRAM, CoreCoord(num_banks - 1, 0)));
    }
}

}  // namespace tt::tt_metal
