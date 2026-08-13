// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Integration tests for per-core L1 allocation via experimental::per_core_allocation.
// These tests require a real device (slow dispatch).

#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <distributed/mesh_io.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/experimental/per_core_allocation/mesh_buffer.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/mesh_device.hpp>
#include "tests/tt_metal/tt_metal/common/device_fixture.hpp"
#include "tt_metal/hw/inc/hostdev/socket.h"

namespace tt::tt_metal {

namespace per_core = experimental::per_core_allocation;

class PerCoreAllocationTest : public MeshDeviceSingleCardBufferFixture {
protected:
    void SetUp() override {
        // Enable HYBRID allocator mode before device creation.
        setenv("TT_METAL_ALLOCATOR_MODE_HYBRID", "1", /*overwrite=*/1);

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
            ids, l1_small_size_, trace_region_size_, 1, dispatch_core_config, {}, DEFAULT_WORKER_L1_SIZE);
        devices_.clear();
        for (const auto& [device_id, device] : id_to_device_) {
            devices_.push_back(device);
        }
        init_max_cbs();
    }

    void TearDown() override {
        MeshDeviceSingleCardBufferFixture::TearDown();
        unsetenv("TT_METAL_ALLOCATOR_MODE_HYBRID");
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

    auto shard_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED);
    experimental::per_core_allocation::set_per_core_allocation(shard_args, true);

    auto buf = Buffer::create(device, total_size, PAGE_SIZE, BufferType::L1, shard_args);

    ASSERT_TRUE(per_core::is_per_core_allocation(*buf));

    // Verify each core has an address within L1 range
    auto cores = corerange_to_cores(CoreRangeSet(core_range), std::nullopt, true);
    for (uint32_t i = 0; i < num_cores; i++) {
        auto addr = per_core::get_per_core_address(*buf, cores[i]);
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
    auto shard_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED);
    experimental::per_core_allocation::set_per_core_allocation(shard_args, true);
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
        auto pc_addr = per_core::get_per_core_address(*per_core_buf, cores[i]);
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

    auto shard_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED);
    experimental::per_core_allocation::set_per_core_allocation(shard_args, true);

    // Create and destroy a buffer
    {
        auto buf1 = Buffer::create(device, total_size, PAGE_SIZE, BufferType::L1, shard_args);
        EXPECT_TRUE(per_core::is_per_core_allocation(*buf1));
        // buf1 destroyed here, freeing per-core allocations
    }

    // Create another buffer on same cores — should succeed (space was freed)
    auto buf2 = Buffer::create(device, total_size, PAGE_SIZE, BufferType::L1, shard_args);
    EXPECT_TRUE(per_core::is_per_core_allocation(*buf2));
    EXPECT_TRUE(buf2->is_allocated());
}

// ================== Per-core socket data-buffer allocation (Phase B) ==================
// Exercises SocketMemoryConfig{..., per_core_allocation=true}: the receiver FIFO should be allocated with the
// per-core allocator (only on the receiver connection core) instead of lockstep across every worker core, and
// the handshake metadata should carry that per-core address.

namespace {

// Builds a single-connection, single-device (md == md) per-core socket on the fixture's first unit mesh.
std::pair<distributed::MeshSocket, distributed::MeshSocket> make_per_core_socket(
    const std::shared_ptr<distributed::MeshDevice>& md,
    const CoreCoord& sender_core,
    const CoreCoord& receiver_core,
    uint32_t fifo_size) {
    distributed::SocketConnection connection(
        distributed::MeshCoreCoord(distributed::MeshCoordinate(0, 0), sender_core),
        distributed::MeshCoreCoord(distributed::MeshCoordinate(0, 0), receiver_core));
    distributed::SocketMemoryConfig mem_config(
        BufferType::L1,
        fifo_size,
        /*sender_sub_device=*/std::nullopt,
        /*receiver_sub_device=*/std::nullopt,
        /*per_core_allocation=*/true);
    distributed::SocketConfig socket_config({connection}, mem_config);
    return distributed::MeshSocket::create_socket_pair(md, md, socket_config);
}

}  // namespace

TEST_F(PerCoreAllocationTest, PerCoreSocketDataBufferPlacement) {
    auto md = this->devices_[0];
    auto* device = md->get_devices()[0];

    const CoreCoord sender_core(0, 0);
    const CoreCoord receiver_core(0, 1);
    const uint32_t fifo_size = 2048;

    auto [send_socket, recv_socket] = make_per_core_socket(md, sender_core, receiver_core, fifo_size);

    auto data_buffer = recv_socket.get_data_buffer();
    ASSERT_TRUE(per_core::is_per_core_allocation(*data_buffer));

    // The FIFO occupies L1 only on the receiver core, at a valid per-core address.
    auto pc_addr = per_core::get_per_core_address(*data_buffer, distributed::MeshCoordinate(0, 0), receiver_core);
    EXPECT_GT(pc_addr, 0u) << "Receiver per-core address should be above the L1 base";
    EXPECT_LT(pc_addr, device->l1_size_per_core()) << "Receiver per-core address exceeds L1 size";

    // A per-core buffer has no single lockstep address.
    EXPECT_EQ(data_buffer->address(), 0u);
}

TEST_F(PerCoreAllocationTest, PerCoreSocketCoexistsWithLockstep) {
    auto md = this->devices_[0];
    auto* device = md->get_devices()[0];

    const CoreCoord sender_core(0, 0);
    const CoreCoord receiver_core(0, 1);
    const uint32_t fifo_size = 2048;

    auto [send_socket, recv_socket] = make_per_core_socket(md, sender_core, receiver_core, fifo_size);
    auto pc_addr = per_core::get_per_core_address(
        *recv_socket.get_data_buffer(), distributed::MeshCoordinate(0, 0), receiver_core);

    // A subsequent lockstep L1 buffer spanning the receiver core must not alias the per-core socket FIFO.
    CoreRange grid(CoreCoord(0, 0), CoreCoord(1, 0));
    ShardSpecBuffer shard_spec(CoreRangeSet(grid), {32, 32}, ShardOrientation::ROW_MAJOR, {32, 32}, {2, 1});
    auto lockstep_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED);
    auto lockstep_buf = Buffer::create(device, 2 * PAGE_SIZE, PAGE_SIZE, BufferType::L1, lockstep_args);
    EXPECT_TRUE(lockstep_buf->is_allocated());
    EXPECT_NE(lockstep_buf->address(), pc_addr) << "Lockstep buffer aliases the per-core socket FIFO";
}

TEST_F(PerCoreAllocationTest, PerCoreSocketConfigMetadataUsesPerCoreAddress) {
    auto md = this->devices_[0];

    const CoreCoord sender_core(0, 0);
    const CoreCoord receiver_core(0, 1);
    const uint32_t fifo_size = 2048;

    auto [send_socket, recv_socket] = make_per_core_socket(md, sender_core, receiver_core, fifo_size);
    auto pc_addr = per_core::get_per_core_address(
        *recv_socket.get_data_buffer(), distributed::MeshCoordinate(0, 0), receiver_core);

    // The receiver's on-device config must point read_ptr/fifo_addr at the receiver core's per-core address.
    std::vector<receiver_socket_md> recv_config_readback;
    distributed::ReadShard(
        md->mesh_command_queue(),
        recv_config_readback,
        recv_socket.get_config_buffer(),
        distributed::MeshCoordinate(0, 0));
    ASSERT_EQ(recv_config_readback.size(), 1u);
    EXPECT_EQ(recv_config_readback[0].fifo_addr, pc_addr);
    EXPECT_EQ(recv_config_readback[0].read_ptr, pc_addr);
    EXPECT_EQ(recv_config_readback[0].fifo_total_size, fifo_size);

    // The sender's downstream_fifo_addr must match the same per-core address.
    std::vector<uint8_t> sender_config_bytes;
    distributed::ReadShard(
        md->mesh_command_queue(),
        sender_config_bytes,
        send_socket.get_config_buffer(),
        distributed::MeshCoordinate(0, 0));
    ASSERT_GE(sender_config_bytes.size(), sizeof(sender_socket_md));
    sender_socket_md sender_md{};
    std::memcpy(&sender_md, sender_config_bytes.data(), sizeof(sender_socket_md));
    EXPECT_EQ(sender_md.downstream_fifo_addr, pc_addr);
    EXPECT_EQ(sender_md.downstream_fifo_total_size, fifo_size);
}

}  // namespace tt::tt_metal
