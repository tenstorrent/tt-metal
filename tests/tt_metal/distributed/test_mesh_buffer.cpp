// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <memory>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <tt-metalium/distributed.hpp>

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

using MeshBufferTest = T3000MultiDeviceFixture;

struct DeviceLocalShardedBufferTestConfig {
    Shape2D num_pages_per_core;
    Shape2D num_cores;
    Shape2D page_shape;
    uint32_t element_size = 1;
    TensorMemoryLayout mem_config = TensorMemoryLayout::HEIGHT_SHARDED;
    ShardOrientation shard_orientation = ShardOrientation::ROW_MAJOR;

    Shape2D tensor2d_shape() {
        return {num_pages_per_core.height() * num_cores.height(), num_pages_per_core.width() * num_cores.width()};
    }

    uint32_t num_pages() { return tensor2d_shape().height() * tensor2d_shape().width(); }

    std::array<uint32_t, 2> shard_shape() {
        return {num_pages_per_core.height() * page_shape.height(), num_pages_per_core.width() * page_shape.width()};
    }

    CoreRangeSet shard_grid() {
        return CoreRangeSet(std::set<CoreRange>(
            {CoreRange(CoreCoord(0, 0), CoreCoord(this->num_cores.height() - 1, this->num_cores.width() - 1))}));
    }

    uint32_t page_size() { return page_shape.height() * page_shape.width() * element_size; }

    ShardSpecBuffer shard_parameters() {
        return ShardSpecBuffer(
            this->shard_grid(), this->shard_shape(), this->shard_orientation, this->page_shape, this->tensor2d_shape());
    }
};

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
    // Verify that a buffer is deallocated on the MeshDevice when it goes
    // out of scope on host. Create a buffer with a certain config in limited
    // scope. Record its address. Create another buffer with the same config
    // outside the scope. Verify that addresses match.
    const DeviceLocalBufferConfig device_local_config{
        .page_size = 1024,
        .buffer_type = BufferType::DRAM,
        .buffer_layout = TensorMemoryLayout::INTERLEAVED,
        .bottom_up = false};

    const ReplicatedBufferConfig buffer_config{.size = 16 << 10};

    // Fetch an address that the allocator provides on first allocation.
    const uint32_t expected_address = [&]() {
        auto buffer = MeshBuffer::create(buffer_config, device_local_config, mesh_device_.get());
        EXPECT_TRUE(buffer->is_allocated());
        return buffer->address();
    }();

    // Test that creating and deallocating a MeshBuffer frees the address.
    auto buffer1 = MeshBuffer::create(buffer_config, device_local_config, mesh_device_.get());
    EXPECT_TRUE(buffer1->is_allocated());
    EXPECT_EQ(buffer1->address(), expected_address);

    buffer1->deallocate();
    EXPECT_FALSE(buffer1->is_allocated());

    auto buffer2 = MeshBuffer::create(buffer_config, device_local_config, mesh_device_.get());
    EXPECT_TRUE(buffer2->is_allocated());
    EXPECT_EQ(buffer2->address(), expected_address);

    // Test deallocation of the view also works.
    auto buffer_view = MeshBuffer::create(buffer_config, device_local_config, mesh_device_.get(), buffer2->address());
    EXPECT_TRUE(buffer_view->is_allocated());
    EXPECT_EQ(buffer_view->address(), expected_address);

    buffer_view->deallocate();
    EXPECT_FALSE(buffer_view->is_allocated());
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

TEST_F(MeshBufferTest, InterleavedShardsReadWrite) {
    constexpr uint32_t NUM_ITERS = 100;
    uint32_t seed = tt::parse_env("TT_METAL_SEED", 0);
    uint32_t single_tile_size = ::tt::tt_metal::detail::TileSize(DataFormat::UInt32);

    for (auto buffer_type : {BufferType::L1, BufferType::DRAM}) {
        DeviceLocalBufferConfig per_device_buffer_config{
            .page_size = single_tile_size,
            .buffer_type = BufferType::L1,
            .buffer_layout = TensorMemoryLayout::INTERLEAVED,
            .bottom_up = false};

        std::uniform_int_distribution<int> gen_num_tiles(1, 1024);
        std::mt19937 rng(seed);
        for (int i = 0; i < NUM_ITERS; i++) {
            uint32_t num_random_tiles = gen_num_tiles(rng);
            ReplicatedBufferConfig global_buffer_config = {
                .size = num_random_tiles * single_tile_size,
            };

            std::shared_ptr<MeshBuffer> buf =
                MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());

            std::vector<uint32_t> src_vec(num_random_tiles * single_tile_size / sizeof(uint32_t), 0);
            std::iota(src_vec.begin(), src_vec.end(), i);
            for (std::size_t logical_x = 0; logical_x < buf->device()->num_cols(); logical_x++) {
                for (std::size_t logical_y = 0; logical_y < buf->device()->num_rows(); logical_y++) {
                    WriteShard(mesh_device_->mesh_command_queue(), buf, src_vec, Coordinate(logical_y, logical_x));
                }
            }

            for (std::size_t logical_x = 0; logical_x < buf->device()->num_cols(); logical_x++) {
                for (std::size_t logical_y = 0; logical_y < buf->device()->num_rows(); logical_y++) {
                    std::vector<uint32_t> dst_vec = {};
                    ReadShard(mesh_device_->mesh_command_queue(), dst_vec, buf, Coordinate(logical_y, logical_x));
                    EXPECT_EQ(dst_vec, src_vec);
                }
            }
        }
    }
}

class DeviceLocalMeshBufferShardingTest
    : public MeshBufferTest,
      public testing::WithParamInterface<
          std::tuple<std::array<uint32_t, 2>, std::array<uint32_t, 2>, TensorMemoryLayout>> {};

TEST_P(DeviceLocalMeshBufferShardingTest, ShardingTest) {
    auto [num_pages_per_core, page_shape, shard_strategy] = GetParam();
    CoreCoord core_grid_size = mesh_device_->compute_with_storage_grid_size();

    DeviceLocalShardedBufferTestConfig test_config{
        .num_pages_per_core = num_pages_per_core,
        .num_cores = {core_grid_size.x, core_grid_size.y},
        .page_shape = page_shape,
        .mem_config = shard_strategy};
    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = test_config.page_size(),
        .buffer_type = BufferType::L1,
        .buffer_layout = test_config.mem_config,
        .shard_parameters = test_config.shard_parameters(),
        .bottom_up = false};

    uint32_t buf_size = test_config.num_pages() * test_config.page_size();
    ReplicatedBufferConfig global_buffer_config{
        .size = buf_size,
    };

    auto buf = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());
    std::vector<uint32_t> src_vec(buf_size / sizeof(uint32_t), 0);
    std::iota(src_vec.begin(), src_vec.end(), 0);

    for (std::size_t logical_x = 0; logical_x < buf->device()->num_cols(); logical_x++) {
        for (std::size_t logical_y = 0; logical_y < buf->device()->num_rows(); logical_y++) {
            WriteShard(mesh_device_->mesh_command_queue(), buf, src_vec, Coordinate(logical_y, logical_x));
        }
    }

    for (std::size_t logical_x = 0; logical_x < buf->device()->num_cols(); logical_x++) {
        for (std::size_t logical_y = 0; logical_y < buf->device()->num_rows(); logical_y++) {
            std::vector<uint32_t> dst_vec = {};
            ReadShard(mesh_device_->mesh_command_queue(), dst_vec, buf, Coordinate(logical_y, logical_x));
            EXPECT_EQ(dst_vec, src_vec);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    DeviceLocalMeshBufferShardingTests,
    DeviceLocalMeshBufferShardingTest,
    ::testing::Combine(
        // num_pages_per_core
        ::testing::Values(
            std::array<uint32_t, 2>{1, 1},
            std::array<uint32_t, 2>{3, 137},
            std::array<uint32_t, 2>{67, 4},
            std::array<uint32_t, 2>{7, 11},
            std::array<uint32_t, 2>{2, 2}),
        // page_shape
        ::testing::Values(
            std::array<uint32_t, 2>{1, 1024},
            std::array<uint32_t, 2>{1, 2048},
            std::array<uint32_t, 2>{1, 4},
            std::array<uint32_t, 2>{32, 32},
            std::array<uint32_t, 2>{1, 120}),
        // shard_strategy
        ::testing::Values(
            TensorMemoryLayout::HEIGHT_SHARDED, TensorMemoryLayout::WIDTH_SHARDED, TensorMemoryLayout::BLOCK_SHARDED)));

TEST_F(MeshBufferTest, SweepShardAndConcat) {
    uint32_t single_tile_size = ::tt::tt_metal::detail::TileSize(DataFormat::UInt32);

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size,
        .buffer_type = BufferType::DRAM,
        .buffer_layout = TensorMemoryLayout::INTERLEAVED,
        .bottom_up = true};
    std::vector<Shape2D> global_buffer_shapes = {
        {64, 128}, {128, 128}, {32, 1024}, {1024, 32}, {512, 64}, {2048, 2048}};
    std::vector<Shape2D> shard_shapes = {{32, 32}, {32, 64}, {32, 128}, {128, 32}, {128, 32}, {512, 1024}};
    for (auto shard_orientation : {ShardOrientation::COL_MAJOR, ShardOrientation::ROW_MAJOR}) {
        for (int i = 0; i < global_buffer_shapes.size(); i++) {
            Shape2D global_buffer_shape = global_buffer_shapes[i];
            Shape2D shard_shape = shard_shapes[i];

            uint32_t global_buffer_size = global_buffer_shape.height() * global_buffer_shape.width() * sizeof(uint32_t);

            ShardedBufferConfig sharded_config{
                .global_size = global_buffer_size,
                .global_buffer_shape = global_buffer_shape,
                .shard_shape = shard_shape,
                .shard_orientation = shard_orientation,
            };

            auto mesh_buffer = MeshBuffer::create(sharded_config, per_device_buffer_config, mesh_device_.get());
            std::vector<uint32_t> src_vec =
                std::vector<uint32_t>(global_buffer_shape.height() * global_buffer_shape.width(), 0);
            std::iota(src_vec.begin(), src_vec.end(), 0);
            EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(), mesh_buffer, src_vec);
            std::vector<uint32_t> dst_vec = {};
            EnqueueReadMeshBuffer(mesh_device_->mesh_command_queue(), dst_vec, mesh_buffer);

            EXPECT_EQ(dst_vec, src_vec);
        }
    }
}

TEST_F(MeshBufferTest, RowMajorShardingAndReplication) {
    uint32_t single_tile_size = ::tt::tt_metal::detail::TileSize(DataFormat::UInt32);

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size,
        .buffer_type = BufferType::DRAM,
        .buffer_layout = TensorMemoryLayout::INTERLEAVED,
        .bottom_up = true};

    std::vector<Shape2D> global_buffer_shapes = {{64, 256}, {128, 128}, {256, 2048}, {32, 512}, {512, 1024}};

    for (int i = 0; i < global_buffer_shapes.size(); i++) {
        auto global_buffer_shape = global_buffer_shapes[i];
        Shape2D shard_shape = {0, global_buffer_shape.width() / mesh_device_->num_cols()};
        // Mesh-Level Sharding Parameters for the MeshBufferView that will be read to verify correctness
        Shape2D global_buffer_read_shape = {
            global_buffer_shape.height() * mesh_device_->num_rows(), global_buffer_shape.width()};
        Shape2D shard_read_shape = {
            global_buffer_shape.height(), global_buffer_shape.width() / mesh_device_->num_cols()};

        uint32_t global_buffer_size = global_buffer_shape.height() * global_buffer_shape.width() * sizeof(uint32_t);
        auto shard_orientation = ShardOrientation::ROW_MAJOR;

        ShardedBufferConfig sharded_config{
            .global_size = global_buffer_size,
            .global_buffer_shape = global_buffer_shape,
            .shard_shape = shard_shape,
            .shard_orientation = shard_orientation,
        };
        // Initialize the ShardedBufferConfig for reading and verifying replicated data
        ShardedBufferConfig sharded_read_view_config{
            .global_size = global_buffer_read_shape.height() * global_buffer_read_shape.width() * sizeof(uint32_t),
            .global_buffer_shape = global_buffer_read_shape,
            .shard_shape = shard_read_shape,
            .shard_orientation = shard_orientation};

        auto mesh_buffer = MeshBuffer::create(sharded_config, per_device_buffer_config, mesh_device_.get());
        std::vector<uint32_t> src_vec =
            std::vector<uint32_t>(global_buffer_shape.height() * global_buffer_shape.width(), 0);
        std::iota(src_vec.begin(), src_vec.end(), 0);

        auto mesh_buffer_read_view = MeshBuffer::create(
            sharded_read_view_config, per_device_buffer_config, mesh_device_.get(), mesh_buffer->address());
        EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(), mesh_buffer, src_vec);
        std::vector<uint32_t> dst_vec =
            std::vector<uint32_t>(global_buffer_read_shape.height() * global_buffer_read_shape.width(), 0);
        EnqueueReadMeshBuffer(mesh_device_->mesh_command_queue(), dst_vec, mesh_buffer_read_view);

        for (int i = 0; i < dst_vec.size(); i++) {
            EXPECT_EQ(dst_vec[i], i % (src_vec.size()));
        }
    }
}

TEST_F(MeshBufferTest, ColMajorShardingAndReplication) {
    uint32_t single_tile_size = ::tt::tt_metal::detail::TileSize(DataFormat::UInt32);

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size,
        .buffer_type = BufferType::DRAM,
        .buffer_layout = TensorMemoryLayout::INTERLEAVED,
        .bottom_up = true};

    std::vector<Shape2D> global_buffer_shapes = {{256, 64}, {1024, 1024}, {128, 32}, {512, 64}, {2048, 256}};

    for (int i = 0; i < global_buffer_shapes.size(); i++) {
        auto global_buffer_shape = global_buffer_shapes[i];
        Shape2D shard_shape = {global_buffer_shape.height() / mesh_device_->num_rows(), 0};
        uint32_t global_buffer_size = global_buffer_shape.height() * global_buffer_shape.width() * sizeof(uint32_t);
        Shape2D global_buffer_read_shape = {
            global_buffer_shape.height(), global_buffer_shape.width() * mesh_device_->num_cols()};
        Shape2D shard_read_shape = {
            global_buffer_shape.height() / mesh_device_->num_rows(), global_buffer_shape.width()};

        ShardOrientation shard_orientation = ShardOrientation::COL_MAJOR;

        ShardedBufferConfig sharded_config{
            .global_size = global_buffer_size,
            .global_buffer_shape = global_buffer_shape,
            .shard_shape = shard_shape,
            .shard_orientation = shard_orientation,
        };

        ShardedBufferConfig sharded_read_view_config{
            .global_size = global_buffer_read_shape.height() * global_buffer_read_shape.width() * sizeof(uint32_t),
            .global_buffer_shape = global_buffer_read_shape,
            .shard_shape = shard_read_shape,
            .shard_orientation = ShardOrientation::ROW_MAJOR};

        auto mesh_buffer = MeshBuffer::create(sharded_config, per_device_buffer_config, mesh_device_.get());
        std::vector<uint32_t> src_vec =
            std::vector<uint32_t>(global_buffer_shape.height() * global_buffer_shape.width(), 0);
        std::iota(src_vec.begin(), src_vec.end(), 0);

        auto mesh_buffer_read_view = MeshBuffer::create(
            sharded_read_view_config, per_device_buffer_config, mesh_device_.get(), mesh_buffer->address());

        EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(), mesh_buffer, src_vec);
        std::vector<uint32_t> dst_vec =
            std::vector<uint32_t>(global_buffer_read_shape.height() * global_buffer_read_shape.width(), 0);
        EnqueueReadMeshBuffer(mesh_device_->mesh_command_queue(), dst_vec, mesh_buffer_read_view);
        for (int i = 0; i < dst_vec.size(); i++) {
            EXPECT_EQ(
                (i / global_buffer_read_shape.width()) * global_buffer_shape.width() + i % global_buffer_shape.width(),
                dst_vec[i]);
        }
    }
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
