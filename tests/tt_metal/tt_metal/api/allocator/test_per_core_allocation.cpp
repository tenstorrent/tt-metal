// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Integration tests for per-core L1 allocation via Buffer::per_core_allocation().
// These tests require a real device (slow dispatch).

#include <gtest/gtest.h>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/device.hpp>
#include "tests/tt_metal/tt_metal/common/device_fixture.hpp"

namespace tt::tt_metal {

class PerCoreAllocationTest : public MeshDeviceSingleCardBufferFixture {
protected:
    void SetUp() override {
        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        std::vector<ChipId> ids;
        for (ChipId id : tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids()) {
            ids.push_back(id);
        }
        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        id_to_device_ = distributed::MeshDevice::create_unit_meshes(
            ids,
            l1_small_size_,
            trace_region_size_,
            1,
            dispatch_core_config,
            {},
            DEFAULT_WORKER_L1_SIZE,
            AllocatorMode::HYBRID);
        devices_.clear();
        for (const auto& [device_id, device] : id_to_device_) {
            devices_.push_back(device);
        }
        this->num_devices_ = this->devices_.size();
        init_max_cbs();
    }
};

// Use 1024-byte page size to be safely above all alignment requirements
// (FreeListOpt internally uses DRAM alignment which may be larger than L1 alignment)
static constexpr DeviceAddr PAGE_SIZE = 1024;

TEST_F(PerCoreAllocationTest, BasicPerCoreAllocation) {
    auto* device = this->devices_[0]->get_devices()[0];
    auto compute_grid = device->compute_with_storage_grid_size();
    uint32_t num_cores = static_cast<uint32_t>(std::min<size_t>(4, compute_grid.x));
    ASSERT_GE(num_cores, 2u) << "Need at least 2 compute cores";

    CoreRange core_range(CoreCoord(0, 0), CoreCoord(num_cores - 1, 0));
    std::array<uint32_t, 2> shard_shape = {32, 32};
    std::array<uint32_t, 2> page_shape = {32, 32};
    std::array<uint32_t, 2> tensor2d_shape = {num_cores, 1};

    ShardSpecBuffer shard_spec(
        CoreRangeSet(core_range), shard_shape, ShardOrientation::ROW_MAJOR, page_shape, tensor2d_shape);

    DeviceAddr total_size = PAGE_SIZE * num_cores;

    auto shard_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED).set_per_core_allocation(true);

    auto buf = Buffer::create(device, total_size, PAGE_SIZE, BufferType::L1, shard_args);

    ASSERT_TRUE(buf->per_core_allocation());

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

    DeviceAddr total_size = 2 * PAGE_SIZE * num_cores;

    // Create per-core buffer
    auto shard_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED).set_per_core_allocation(true);
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

    DeviceAddr total_size = 2 * PAGE_SIZE * num_cores;

    auto shard_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED).set_per_core_allocation(true);

    // Create and destroy a buffer
    {
        auto buf1 = Buffer::create(device, total_size, PAGE_SIZE, BufferType::L1, shard_args);
        EXPECT_TRUE(buf1->per_core_allocation());
        // buf1 destroyed here, freeing per-core allocations
    }

    // Create another buffer on same cores — should succeed (space was freed)
    auto buf2 = Buffer::create(device, total_size, PAGE_SIZE, BufferType::L1, shard_args);
    EXPECT_TRUE(buf2->per_core_allocation());
    EXPECT_TRUE(buf2->is_allocated());
}

}  // namespace tt::tt_metal
