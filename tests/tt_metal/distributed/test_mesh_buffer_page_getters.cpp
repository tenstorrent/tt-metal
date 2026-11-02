// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/shape2d.hpp>
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

using MeshBufferPageGettersTest = GenericMeshDeviceFixture;

void expect_page_getters_match_device_buffers(const MeshBuffer& mesh_buffer) {
    for (const auto& coord : mesh_buffer.device()->get_view().get_local_mesh_coord_range()) {
        const Buffer& buffer = *mesh_buffer.get_device_buffer(coord);
        EXPECT_EQ(mesh_buffer.alignment(), buffer.alignment());
        EXPECT_EQ(mesh_buffer.aligned_page_size(), buffer.aligned_page_size());
        EXPECT_EQ(mesh_buffer.num_dev_pages(), buffer.num_dev_pages());
        EXPECT_EQ(mesh_buffer.aligned_size(), buffer.aligned_size());
        EXPECT_EQ(mesh_buffer.aligned_size_per_bank(), buffer.aligned_size_per_bank());
    }
}

TEST_F(MeshBufferPageGettersTest, Interleaved) {
    // A page size that isn't a multiple of the alignment checks that pages are padded.
    constexpr DeviceAddr page_size = 100;
    constexpr DeviceAddr size = 40 * page_size;
    for (auto buffer_type : {BufferType::DRAM, BufferType::L1}) {
        const DeviceLocalBufferConfig device_local_config{
            .page_size = page_size, .buffer_type = buffer_type, .bottom_up = false};
        auto mesh_buffer =
            MeshBuffer::create(ReplicatedBufferConfig{.size = size}, device_local_config, mesh_device_.get());

        const uint32_t alignment = mesh_device_->allocator()->get_alignment(buffer_type);
        EXPECT_EQ(mesh_buffer->alignment(), alignment);
        EXPECT_EQ(mesh_buffer->aligned_page_size(), tt::round_up(page_size, alignment));
        EXPECT_EQ(mesh_buffer->num_dev_pages(), size / page_size);
        expect_page_getters_match_device_buffers(*mesh_buffer);
    }
}

TEST_F(MeshBufferPageGettersTest, ShardedWithPadding) {
    // 5 pages height-sharded 2 per core over 3 cores: the last shard is padded to 2 pages.
    const Shape2D page_shape{32, 32};
    constexpr uint32_t page_size = 32 * 32 * 2;
    const ShardSpecBuffer shard_spec(
        CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 2))),
        {2 * page_shape.height(), page_shape.width()},
        ShardOrientation::ROW_MAJOR,
        page_shape,
        {5, 1});
    const DeviceLocalBufferConfig device_local_config{
        .page_size = page_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false};
    auto mesh_buffer =
        MeshBuffer::create(ReplicatedBufferConfig{.size = 5 * page_size}, device_local_config, mesh_device_.get());

    EXPECT_EQ(mesh_buffer->num_dev_pages(), 6);
    EXPECT_EQ(mesh_buffer->aligned_size(), 6 * mesh_buffer->aligned_page_size());
    expect_page_getters_match_device_buffers(*mesh_buffer);
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
