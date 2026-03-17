// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Integration tests for per-core L1 allocation via Buffer::set_per_core_shard_sizes().
// These tests require a real device (slow dispatch).

#include <gtest/gtest.h>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/device.hpp>
#include "tests/tt_metal/tt_metal/common/device_fixture.hpp"

namespace tt::tt_metal {

class PerCoreAllocationTest : public MeshDeviceSingleCardBufferFixture {};

// Use 1024-byte page size to be safely above all alignment requirements
// (FreeListOpt internally uses DRAM alignment which may be larger than L1 alignment)
static constexpr DeviceAddr PAGE_SIZE = 1024;

TEST_F(PerCoreAllocationTest, DifferentSizesPerCore) {
    auto* device = this->devices_[0]->get_devices()[0];
    auto compute_grid = device->compute_with_storage_grid_size();
    uint32_t num_cores = static_cast<uint32_t>(std::min<size_t>(4, compute_grid.x));
    ASSERT_GE(num_cores, 2u) << "Need at least 2 compute cores";

    CoreRange core_range(CoreCoord(0, 0), CoreCoord(num_cores - 1, 0));
    // shard_shape and page_shape chosen so that num_pages per shard = 1
    std::array<uint32_t, 2> shard_shape = {32, 32};
    std::array<uint32_t, 2> page_shape = {32, 32};
    std::array<uint32_t, 2> tensor2d_shape = {num_cores, 1};

    ShardSpecBuffer shard_spec(
        CoreRangeSet(core_range), shard_shape, ShardOrientation::ROW_MAJOR, page_shape, tensor2d_shape);

    // Different sizes per core: 1KB, 2KB, 3KB, 4KB
    std::vector<DeviceAddr> per_core_sizes;
    for (uint32_t i = 0; i < num_cores; i++) {
        per_core_sizes.push_back(static_cast<DeviceAddr>((i + 1) * PAGE_SIZE));
    }

    DeviceAddr max_size = *std::max_element(per_core_sizes.begin(), per_core_sizes.end());
    DeviceAddr total_size = max_size * num_cores;

    auto shard_args =
        BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED).set_per_core_shard_sizes(per_core_sizes);

    auto buf = Buffer::create(device, total_size, PAGE_SIZE, BufferType::L1, shard_args);

    ASSERT_TRUE(buf->has_per_core_addresses());

    // Verify each core has an address within L1 range
    auto cores = corerange_to_cores(CoreRangeSet(core_range), std::nullopt, true);
    for (uint32_t i = 0; i < num_cores; i++) {
        auto addr = buf->per_core_address(cores[i]);
        EXPECT_GT(addr, 0u) << "Core " << i << " address should be non-zero (above L1 base)";
        EXPECT_LT(addr, device->l1_size_per_core()) << "Core " << i << " address exceeds L1 size";
    }
}

TEST_F(PerCoreAllocationTest, PerCoreAndLockstepCoexist) {
    auto* device = this->devices_[0]->get_devices()[0];
    auto compute_grid = device->compute_with_storage_grid_size();
    uint32_t num_cores = static_cast<uint32_t>(std::min<size_t>(4, compute_grid.x));
    ASSERT_GE(num_cores, 2u);

    CoreRange core_range(CoreCoord(0, 0), CoreCoord(num_cores - 1, 0));
    std::array<uint32_t, 2> shard_shape = {32, 32};
    std::array<uint32_t, 2> page_shape = {32, 32};
    std::array<uint32_t, 2> tensor2d_shape = {num_cores, 1};

    ShardSpecBuffer shard_spec(
        CoreRangeSet(core_range), shard_shape, ShardOrientation::ROW_MAJOR, page_shape, tensor2d_shape);

    std::vector<DeviceAddr> per_core_sizes(num_cores, 2 * PAGE_SIZE);

    DeviceAddr total_size = 2 * PAGE_SIZE * num_cores;

    // Create per-core buffer
    auto shard_args =
        BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED).set_per_core_shard_sizes(per_core_sizes);
    auto per_core_buf = Buffer::create(device, total_size, PAGE_SIZE, BufferType::L1, shard_args);

    // Create lockstep buffer on same cores
    auto lockstep_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED);
    auto lockstep_buf = Buffer::create(device, total_size, PAGE_SIZE, BufferType::L1, lockstep_args);

    // Both should be allocated successfully
    EXPECT_TRUE(per_core_buf->is_allocated());
    EXPECT_TRUE(lockstep_buf->is_allocated());

    // Lockstep address should not overlap any per-core address
    auto lockstep_addr = lockstep_buf->address();
    auto cores = corerange_to_cores(CoreRangeSet(core_range), std::nullopt, true);
    for (uint32_t i = 0; i < num_cores; i++) {
        auto pc_addr = per_core_buf->per_core_address(cores[i]);
        EXPECT_NE(lockstep_addr, pc_addr) << "Lockstep address overlaps per-core address at core " << i;
    }
}

TEST_F(PerCoreAllocationTest, DeallocationFreesPerCoreSpace) {
    auto* device = this->devices_[0]->get_devices()[0];
    auto compute_grid = device->compute_with_storage_grid_size();
    uint32_t num_cores = static_cast<uint32_t>(std::min<size_t>(4, compute_grid.x));
    ASSERT_GE(num_cores, 2u);

    CoreRange core_range(CoreCoord(0, 0), CoreCoord(num_cores - 1, 0));
    std::array<uint32_t, 2> shard_shape = {32, 32};
    std::array<uint32_t, 2> page_shape = {32, 32};
    std::array<uint32_t, 2> tensor2d_shape = {num_cores, 1};

    ShardSpecBuffer shard_spec(
        CoreRangeSet(core_range), shard_shape, ShardOrientation::ROW_MAJOR, page_shape, tensor2d_shape);

    std::vector<DeviceAddr> per_core_sizes(num_cores, 2 * PAGE_SIZE);

    DeviceAddr total_size = 2 * PAGE_SIZE * num_cores;

    auto shard_args =
        BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED).set_per_core_shard_sizes(per_core_sizes);

    // Create and destroy a buffer
    {
        auto buf1 = Buffer::create(device, total_size, PAGE_SIZE, BufferType::L1, shard_args);
        EXPECT_TRUE(buf1->has_per_core_addresses());
        // buf1 destroyed here, freeing per-core allocations
    }

    // Create another buffer on same cores — should succeed (space was freed)
    auto buf2 = Buffer::create(device, total_size, PAGE_SIZE, BufferType::L1, shard_args);
    EXPECT_TRUE(buf2->has_per_core_addresses());
    EXPECT_TRUE(buf2->is_allocated());
}

TEST_F(PerCoreAllocationTest, SizeMismatchFatals) {
    auto* device = this->devices_[0]->get_devices()[0];
    auto compute_grid = device->compute_with_storage_grid_size();
    uint32_t num_cores = static_cast<uint32_t>(std::min<size_t>(4, compute_grid.x));
    ASSERT_GE(num_cores, 2u);

    CoreRange core_range(CoreCoord(0, 0), CoreCoord(num_cores - 1, 0));
    std::array<uint32_t, 2> shard_shape = {32, 32};
    std::array<uint32_t, 2> page_shape = {32, 32};
    std::array<uint32_t, 2> tensor2d_shape = {num_cores, 1};

    ShardSpecBuffer shard_spec(
        CoreRangeSet(core_range), shard_shape, ShardOrientation::ROW_MAJOR, page_shape, tensor2d_shape);

    // Wrong count: num_cores - 1 sizes for num_cores grid
    std::vector<DeviceAddr> wrong_sizes(num_cores - 1, 2 * PAGE_SIZE);

    DeviceAddr total_size = 2 * PAGE_SIZE * num_cores;

    auto shard_args =
        BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED).set_per_core_shard_sizes(wrong_sizes);

    EXPECT_ANY_THROW(Buffer::create(device, total_size, PAGE_SIZE, BufferType::L1, shard_args));
}

}  // namespace tt::tt_metal
