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

namespace tt::tt_metal {
namespace {

constexpr uint32_t kPageSize = 1024;

std::shared_ptr<Buffer> make_width_sharded_buffer(IDevice* device, BufferType buffer_type, const CoreCoord& core) {
    // One shard, on one core. Size is irrelevant to the core validation, but must be page-aligned.
    const CoreRangeSet grid(std::set<CoreRange>({CoreRange(core, core)}));
    return CreateBuffer(ShardedBufferConfig{
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
            {1, kPageSize / sizeof(uint32_t)})});
}

}  // namespace

TEST_F(AnyDispatchMeshDeviceSingleCardFixture, ShardGridValidationL1RejectsCoreOutsideGrid) {
    for (auto& mesh_device : this->devices_) {
        auto* device = mesh_device->get_devices()[0];
        // Past the Tensix grid entirely: these coordinates name no core at all.
        const CoreCoord tensix_grid = device->logical_grid_size();

        EXPECT_ANY_THROW(make_width_sharded_buffer(device, BufferType::L1, CoreCoord(tensix_grid.x, 0)));
        EXPECT_ANY_THROW(make_width_sharded_buffer(device, BufferType::L1, CoreCoord(0, tensix_grid.y)));
    }
}

TEST_F(AnyDispatchMeshDeviceSingleCardFixture, ShardGridValidationL1RejectsCoreWithoutBank) {
    for (auto& mesh_device : this->devices_) {
        auto* device = mesh_device->get_devices()[0];
        // A Tensix core that exists but is not ComputeAndStore -- dispatch owns its L1, so the
        // allocator hands out no bank there. This one is worth pinning separately from the case
        // above: the core translates fine, so a raw write to it succeeds and silently lands on
        // memory the allocator does not manage.
        const CoreCoord tensix_grid = device->logical_grid_size();
        const CoreCoord compute_grid = device->compute_with_storage_grid_size();
        if (compute_grid.x >= tensix_grid.x) {
            GTEST_SKIP() << "No Tensix column outside the compute grid on this device";
        }

        EXPECT_ANY_THROW(make_width_sharded_buffer(device, BufferType::L1, CoreCoord(compute_grid.x, 0)));
    }
}

TEST_F(AnyDispatchMeshDeviceSingleCardFixture, ShardGridValidationL1AcceptsCoreInsideGrid) {
    for (auto& mesh_device : this->devices_) {
        auto* device = mesh_device->get_devices()[0];
        const CoreCoord compute_grid = device->compute_with_storage_grid_size();

        EXPECT_NO_THROW(
            make_width_sharded_buffer(device, BufferType::L1, CoreCoord(compute_grid.x - 1, compute_grid.y - 1)));
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
