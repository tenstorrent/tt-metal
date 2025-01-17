// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <memory>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/mesh_device_view.hpp>

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tt_metal/distributed/mesh_buffer.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

using MeshBufferTest = T3000MultiDeviceFixture;

TEST_F(MeshBufferTest, ConfigValidation) {
    const DeviceLocalBufferConfig device_local_config{
        .page_size = 1024,
        .buffer_type = BufferType::DRAM,
        .buffer_layout = TensorMemoryLayout::INTERLEAVED,
        .bottom_up = false};

    ASSERT_EQ(mesh_device_->num_rows(), 2);
    ASSERT_EQ(mesh_device_->num_cols(), 4);

    // Unaligned shard shape
    EXPECT_ANY_THROW(MeshBuffer::create(
        ShardedBufferConfig{.global_size = 16 << 10, .global_buffer_shape = {64, 128}, .shard_shape = {32, 120}},
        device_local_config,
        mesh_device_.get()));

    // Number of shards exceeds the number of devices
    EXPECT_ANY_THROW(MeshBuffer::create(
        ShardedBufferConfig{.global_size = 16 << 10, .global_buffer_shape = {64, 128}, .shard_shape = {16, 16}},
        device_local_config,
        mesh_device_.get()));

    // 32x32 shards distributed across 2x4 mesh, resulting in 64x128 global shape.
    auto buffer = MeshBuffer::create(
        ShardedBufferConfig{.global_size = 16 << 10, .global_buffer_shape = {64, 128}, .shard_shape = {32, 32}},
        device_local_config,
        mesh_device_.get());
}

TEST_F(MeshBufferTest, ShardedBufferInitialization) {
    const DeviceLocalBufferConfig device_local_config{
        .page_size = 1024,
        .buffer_type = BufferType::DRAM,
        .buffer_layout = TensorMemoryLayout::INTERLEAVED,
        .bottom_up = false};

    const ShardedBufferConfig buffer_config{
        .global_size = 16 << 10, .global_buffer_shape = {64, 128}, .shard_shape = {32, 32}};
    EXPECT_EQ(buffer_config.compute_datum_size_bytes(), 2);
    auto sharded_buffer = MeshBuffer::create(buffer_config, device_local_config, mesh_device_.get());

    EXPECT_EQ(sharded_buffer->size(), 16 << 10);
    EXPECT_EQ(sharded_buffer->global_layout(), MeshBufferLayout::SHARDED);
    EXPECT_EQ(sharded_buffer->device_local_size(), 2 << 10);
}

TEST_F(MeshBufferTest, ReplicatedBufferInitialization) {
    const DeviceLocalBufferConfig device_local_config{
        .page_size = 1024,
        .buffer_type = BufferType::DRAM,
        .buffer_layout = TensorMemoryLayout::INTERLEAVED,
        .bottom_up = false};

    const ReplicatedBufferConfig buffer_config{.size = 16 << 10};
    auto replicated_buffer = MeshBuffer::create(buffer_config, device_local_config, mesh_device_.get());

    EXPECT_EQ(replicated_buffer->size(), 16 << 10);
    EXPECT_EQ(replicated_buffer->global_layout(), MeshBufferLayout::REPLICATED);
    EXPECT_EQ(replicated_buffer->device_local_size(), 16 << 10);
}

TEST_F(MeshBufferTest, Deallocation) {
    const DeviceLocalBufferConfig device_local_config{
        .page_size = 1024,
        .buffer_type = BufferType::DRAM,
        .buffer_layout = TensorMemoryLayout::INTERLEAVED,
        .bottom_up = false};

    const ReplicatedBufferConfig buffer_config{.size = 16 << 10};
    std::shared_ptr<Buffer> buffer;
    Allocator* allocator = nullptr;
    {
        auto replicated_buffer = MeshBuffer::create(buffer_config, device_local_config, mesh_device_.get());
        buffer = replicated_buffer->get_device_buffer(Coordinate{0, 0});
        allocator = buffer->allocator();
        EXPECT_TRUE(allocator->allocated_buffers.contains(buffer.get()));
    }
    EXPECT_FALSE(allocator->allocated_buffers.contains(buffer.get()));
}

TEST_F(MeshBufferTest, GetDeviceBuffer) {
    const DeviceLocalBufferConfig device_local_config{
        .page_size = 1024,
        .buffer_type = BufferType::DRAM,
        .buffer_layout = TensorMemoryLayout::INTERLEAVED,
        .bottom_up = false};

    auto replicated_buffer =
        MeshBuffer::create(ReplicatedBufferConfig{.size = 16 << 10}, device_local_config, mesh_device_.get());

    // Out of bounds coordinates.
    EXPECT_ANY_THROW(replicated_buffer->get_device_buffer(Coordinate{2, 4}));

    EXPECT_NO_THROW(replicated_buffer->get_device_buffer(Coordinate{1, 3}));
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
