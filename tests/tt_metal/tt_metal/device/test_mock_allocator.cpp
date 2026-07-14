// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Pure-metal unit tests for experimental::MockAllocator.
// Drives allocations via MeshBuffer::create — no dependency on the TTNN layer.

#include <gtest/gtest.h>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/mock_device/mock_allocator.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/system_mesh.hpp>

#include <umd/device/types/arch.hpp>

#include "impl/device/mock_device_util.hpp"

namespace tt::tt_metal {

// ============================================================================
// MockAllocatorTest — single mock device fixture
// ============================================================================

class MockAllocatorTest : public ::testing::Test {
protected:
    std::unique_ptr<MetalEnv> mock_env_;
    std::shared_ptr<distributed::MeshDevice> mock_device_;

    void SetUp() override {
        mock_env_ = std::make_unique<MetalEnv>(
            MetalEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1)));
        auto mesh_shape = mock_env_->get_system_mesh().shape();
        mock_device_ = mock_env_->create_mesh_device(distributed::MeshDeviceConfig(mesh_shape));
        ASSERT_GT(mock_device_->num_devices(), 0u);
    }

    void TearDown() override {
        mock_device_.reset();
        mock_env_.reset();
    }
};

TEST_F(MockAllocatorTest, MockDeviceCreationAndQueries) {
    EXPECT_EQ(mock_env_->get_arch(), tt::ARCH::WORMHOLE_B0);
    EXPECT_GT(mock_env_->get_l1_size(), 0u);
    EXPECT_GT(mock_env_->get_dram_alignment(), 0u);
    EXPECT_GT(mock_env_->get_l1_alignment(), 0u);
    EXPECT_EQ(mock_device_->num_devices(), 1u);
}

TEST_F(MockAllocatorTest, BufferAllocationOnMockDevice) {
    constexpr size_t page_size = 4096;
    constexpr size_t buffer_size = page_size * 12;
    distributed::DeviceLocalBufferConfig local_config{.page_size = buffer_size, .buffer_type = BufferType::L1};
    distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};
    auto buffer = distributed::MeshBuffer::create(buffer_config, local_config, mock_device_.get());
    EXPECT_GT(buffer->address(), 0u);
    EXPECT_TRUE(buffer->is_allocated());
    buffer->deallocate();
    EXPECT_FALSE(buffer->is_allocated());
}

TEST_F(MockAllocatorTest, GetMockAllocatorReturnsNonNull) {
    auto* mock_alloc = experimental::get_mock_allocator(*mock_device_);
    ASSERT_NE(mock_alloc, nullptr);
}

TEST_F(MockAllocatorTest, ExtractAndOverrideRoundtrip) {
    // Fresh mock device — extracted state should be empty.
    auto empty_state = experimental::extract_mock_allocator_state(*mock_device_);
    EXPECT_TRUE(empty_state.is_empty(BufferType::L1));
    EXPECT_FALSE(empty_state.lowest_occupied(BufferType::L1).has_value());

    // Allocate a buffer, capture state while buffer is alive.
    constexpr size_t buffer_size = 4096 * 8;
    experimental::MockAllocatorState state_with_buffer;
    {
        distributed::DeviceLocalBufferConfig local{.page_size = buffer_size, .buffer_type = BufferType::L1};
        distributed::ReplicatedBufferConfig mesh{.size = buffer_size};
        auto buffer = distributed::MeshBuffer::create(mesh, local, mock_device_.get());
        state_with_buffer = experimental::extract_mock_allocator_state(*mock_device_);
        EXPECT_FALSE(state_with_buffer.is_empty(BufferType::L1));
        EXPECT_GT(state_with_buffer.total_allocated_size(BufferType::L1), 0u);
        EXPECT_TRUE(state_with_buffer.lowest_occupied(BufferType::L1).has_value());
    }

    // Buffer destroyed — allocator is clean.
    auto stats_clean = mock_device_->allocator()->get_statistics(BufferType::L1);
    EXPECT_EQ(stats_clean.total_allocated_bytes, 0u);

    // Restore the with-buffer snapshot — allocator reports the same occupancy again.
    experimental::override_mock_allocator_state(*mock_device_, state_with_buffer);
    auto stats_restored = mock_device_->allocator()->get_statistics(BufferType::L1);
    EXPECT_GT(stats_restored.total_allocated_bytes, 0u);

    // Restore the empty snapshot — allocator is clean.
    experimental::override_mock_allocator_state(*mock_device_, empty_state);
    auto stats_empty = mock_device_->allocator()->get_statistics(BufferType::L1);
    EXPECT_EQ(stats_empty.total_allocated_bytes, 0u);
}

TEST_F(MockAllocatorTest, CheckpointRestoreViaMeshBuffer) {
    constexpr size_t small_size = 4096 * 4;
    constexpr size_t large_size = 4096 * 16;

    // Take a checkpoint with one small buffer allocated.
    experimental::MockAllocatorState checkpoint;
    {
        distributed::DeviceLocalBufferConfig local{.page_size = small_size, .buffer_type = BufferType::L1};
        distributed::ReplicatedBufferConfig mesh{.size = small_size};
        auto buf = distributed::MeshBuffer::create(mesh, local, mock_device_.get());
        checkpoint = experimental::extract_mock_allocator_state(*mock_device_);
    }

    // Branch A: restore checkpoint, allocate a large buffer on top.
    experimental::MockAllocatorState state_a;
    {
        experimental::override_mock_allocator_state(*mock_device_, checkpoint);
        distributed::DeviceLocalBufferConfig local{.page_size = large_size, .buffer_type = BufferType::L1};
        distributed::ReplicatedBufferConfig mesh{.size = large_size};
        auto buf = distributed::MeshBuffer::create(mesh, local, mock_device_.get());
        state_a = experimental::extract_mock_allocator_state(*mock_device_);
    }

    // Branch B: restore the same checkpoint, allocate a smaller buffer instead.
    experimental::MockAllocatorState state_b;
    {
        experimental::override_mock_allocator_state(*mock_device_, checkpoint);
        distributed::DeviceLocalBufferConfig local{.page_size = small_size, .buffer_type = BufferType::L1};
        distributed::ReplicatedBufferConfig mesh{.size = small_size};
        auto buf = distributed::MeshBuffer::create(mesh, local, mock_device_.get());
        state_b = experimental::extract_mock_allocator_state(*mock_device_);
    }

    // Branch A allocated more bytes than branch B.
    EXPECT_GT(state_a.total_allocated_size(BufferType::L1), state_b.total_allocated_size(BufferType::L1));
}

}  // namespace tt::tt_metal
