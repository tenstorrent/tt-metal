// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <boost/move/utility_core.hpp>
#include <gtest/gtest.h>
#include <cstdint>
#include <tt-metalium/distributed.hpp>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include "env_lib.hpp"
#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/shape2d.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/memory_pin.hpp>
#include <tt-metalium/host_buffer.hpp>
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_metal/distributed/mesh_device_impl.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

using MeshBufferTest2x4 = MeshDevice2x4Fixture;
using MeshBufferTestSuite = GenericMeshDeviceFixture;

struct DeviceLocalShardedBufferTestConfig {
    Shape2D num_pages_per_core;
    Shape2D num_cores;
    Shape2D page_shape;
    uint32_t element_size = 1;
    TensorMemoryLayout mem_config = TensorMemoryLayout::HEIGHT_SHARDED;
    ShardOrientation shard_orientation = ShardOrientation::ROW_MAJOR;

    Shape2D tensor2d_shape_in_pages() const {
        return {num_pages_per_core.height() * num_cores.height(), num_pages_per_core.width() * num_cores.width()};
    }

    uint32_t num_pages() const { return tensor2d_shape_in_pages().height() * tensor2d_shape_in_pages().width(); }

    std::array<uint32_t, 2> shard_shape() const {
        return {num_pages_per_core.height() * page_shape.height(), num_pages_per_core.width() * page_shape.width()};
    }

    CoreRangeSet shard_grid() const {
        return CoreRangeSet(std::set<CoreRange>(
            {CoreRange(CoreCoord(0, 0), CoreCoord(this->num_cores.height() - 1, this->num_cores.width() - 1))}));
    }

    uint32_t page_size() const { return page_shape.height() * page_shape.width() * element_size; }

    ShardSpecBuffer shard_parameters() const {
        return ShardSpecBuffer(
            this->shard_grid(),
            this->shard_shape(),
            this->shard_orientation,
            this->page_shape,
            this->tensor2d_shape_in_pages());
    }
};

TEST_F(MeshBufferTest2x4, ShardedBufferInitialization) {
    const DeviceLocalBufferConfig device_local_config{
        .page_size = 1024, .buffer_type = BufferType::DRAM, .bottom_up = false};

    const ShardedBufferConfig buffer_config{
        .global_size = 16 << 10, .global_buffer_shape = {64, 128}, .shard_shape = {32, 32}};
    EXPECT_EQ(buffer_config.compute_datum_size_bytes(), 2);
    auto sharded_buffer = MeshBuffer::create(buffer_config, device_local_config, mesh_device_.get());

    EXPECT_EQ(sharded_buffer->size(), 16 << 10);
    EXPECT_EQ(sharded_buffer->global_layout(), MeshBufferLayout::SHARDED);
    EXPECT_EQ(sharded_buffer->device_local_size(), 2 << 10);
}

TEST_F(MeshBufferTest2x4, ReplicatedBufferInitialization) {
    const DeviceLocalBufferConfig device_local_config{
        .page_size = 1024, .buffer_type = BufferType::DRAM, .bottom_up = false};

    const ReplicatedBufferConfig buffer_config{.size = 16 << 10};
    auto replicated_buffer = MeshBuffer::create(buffer_config, device_local_config, mesh_device_.get());

    EXPECT_EQ(replicated_buffer->size(), 16 << 10);
    EXPECT_EQ(replicated_buffer->global_layout(), MeshBufferLayout::REPLICATED);
    EXPECT_EQ(replicated_buffer->device_local_size(), 16 << 10);
}

TEST_F(MeshBufferTest2x4, Deallocation) {
    // Verify that a buffer is deallocated on the MeshDevice when it goes
    // out of scope on host. Create a buffer with a certain config in limited
    // scope. Record its address. Create another buffer with the same config
    // outside the scope. Verify that addresses match.
    const DeviceLocalBufferConfig device_local_config{
        .page_size = 1024, .buffer_type = BufferType::DRAM, .bottom_up = false};

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

TEST(MeshBufferTest, DeallocationWithoutMeshDevice) {
    // Repeated device init takes very long on TG. Lower the number of iterations.
    int iterations = 100;
    if (tt::tt_metal::MetalContext::instance().get_cluster().is_galaxy_cluster()) {
        iterations = 10;
    }

    for (int i = 0; i < iterations; i++) {
        MeshDeviceConfig config(MeshShape(1, 1));
        auto mesh_device =
            MeshDevice::create(config, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreType::WORKER);

        const DeviceLocalBufferConfig device_local_config{
            .page_size = 2048, .buffer_type = BufferType::DRAM, .bottom_up = false};
        const ReplicatedBufferConfig buffer_config{.size = 2048};
        auto buffer = MeshBuffer::create(buffer_config, device_local_config, mesh_device.get());

        mesh_device.reset();
    }
}

TEST(MeshBufferTest, DeallocationWithMeshDeviceClosed) {
    // Repeated device init takes very long on TG. Lower the number of iterations.
    int iterations = 100;
    if (tt::tt_metal::MetalContext::instance().get_cluster().is_galaxy_cluster()) {
        iterations = 10;
    }

    for (int i = 0; i < iterations; i++) {
        MeshDeviceConfig config(MeshShape(1, 1));
        auto mesh_device =
            MeshDevice::create(config, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreType::WORKER);

        const DeviceLocalBufferConfig device_local_config{
            .page_size = 2048, .buffer_type = BufferType::DRAM, .bottom_up = false};
        const ReplicatedBufferConfig buffer_config{.size = 2048};
        auto buffer = MeshBuffer::create(buffer_config, device_local_config, mesh_device.get());

        mesh_device->close();
    }
}

TEST_F(MeshBufferTest2x4, GetDeviceBuffer) {
    const DeviceLocalBufferConfig device_local_config{
        .page_size = 1024, .buffer_type = BufferType::DRAM, .bottom_up = false};

    auto replicated_buffer =
        MeshBuffer::create(ReplicatedBufferConfig{.size = 16 << 10}, device_local_config, mesh_device_.get());

    // Out of bounds coordinates.
    EXPECT_ANY_THROW(replicated_buffer->get_device_buffer(MeshCoordinate{2, 4}));

    EXPECT_NO_THROW(replicated_buffer->get_device_buffer(MeshCoordinate{1, 3}));
}

class DeviceLocalMeshBufferShardingTest
    : public MeshBufferTest2x4,
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
        .sharding_args = BufferShardingArgs(test_config.shard_parameters(), test_config.mem_config),
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
            WriteShard(mesh_device_->mesh_command_queue(), buf, src_vec, MeshCoordinate(logical_y, logical_x));
        }
    }

    for (std::size_t logical_x = 0; logical_x < buf->device()->num_cols(); logical_x++) {
        for (std::size_t logical_y = 0; logical_y < buf->device()->num_rows(); logical_y++) {
            std::vector<uint32_t> dst_vec = {};
            ReadShard(mesh_device_->mesh_command_queue(), dst_vec, buf, MeshCoordinate(logical_y, logical_x));
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

TEST_F(MeshBufferTest2x4, SweepShardAndConcat) {
    uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size, .buffer_type = BufferType::DRAM, .bottom_up = true};
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

TEST_F(MeshBufferTestSuite, ConfigValidation) {
    const DeviceLocalBufferConfig device_local_config{
        .page_size = 1024, .buffer_type = BufferType::DRAM, .bottom_up = false};

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

    // Buffer with a global shape of 64x128 distributed across a 2x4 or 2x1 Mesh.
    auto buffer = MeshBuffer::create(
        ShardedBufferConfig{
            .global_size = 16 << 10,
            .global_buffer_shape = {128, 256},
            .shard_shape = {128 / mesh_device_->num_rows(), 256 / mesh_device_->num_cols()}},
        device_local_config,
        mesh_device_.get());
}

TEST_F(MeshBufferTestSuite, InterleavedShardsReadWrite) {
    constexpr uint32_t NUM_ITERS = 100;
    uint32_t seed = tt::parse_env("TT_METAL_SEED", 0);
    uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);

    for ([[maybe_unused]] auto buffer_type : {BufferType::L1, BufferType::DRAM}) {
        DeviceLocalBufferConfig per_device_buffer_config{
            .page_size = single_tile_size, .buffer_type = BufferType::L1, .bottom_up = false};

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
                    WriteShard(mesh_device_->mesh_command_queue(), buf, src_vec, MeshCoordinate(logical_y, logical_x));
                }
            }

            for (std::size_t logical_x = 0; logical_x < buf->device()->num_cols(); logical_x++) {
                for (std::size_t logical_y = 0; logical_y < buf->device()->num_rows(); logical_y++) {
                    if (!mesh_device_->impl().is_local(MeshCoordinate(logical_y, logical_x))) {
                        continue;
                    }
                    std::vector<uint32_t> dst_vec = {};
                    ReadShard(mesh_device_->mesh_command_queue(), dst_vec, buf, MeshCoordinate(logical_y, logical_x));
                    EXPECT_EQ(dst_vec, src_vec);
                }
            }
        }
    }
}

TEST_F(MeshBufferTestSuite, RowMajorShardingAndReplication) {
    uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size, .buffer_type = BufferType::DRAM, .bottom_up = true};

    std::vector<Shape2D> global_buffer_shapes = {{64, 256}, {128, 128}, {256, 2048}, {32, 512}, {512, 1024}};

    for (auto global_buffer_shape : global_buffer_shapes) {
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

TEST_F(MeshBufferTestSuite, ColMajorShardingAndReplication) {
    uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size, .buffer_type = BufferType::DRAM, .bottom_up = true};

    std::vector<Shape2D> global_buffer_shapes = {{256, 64}, {1024, 1024}, {128, 32}, {512, 64}, {2048, 256}};

    for (auto global_buffer_shape : global_buffer_shapes) {
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

TEST_F(MeshBufferTestSuite, MultiShardReadWrite) {
    constexpr uint32_t NUM_ITERS = 50;
    uint32_t seed = tt::parse_env("TT_METAL_SEED", 0);
    uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);

    std::uniform_int_distribution<int> gen_num_datums(32, 128);
    std::mt19937 rng(seed);

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size, .buffer_type = BufferType::DRAM, .bottom_up = true};

    distributed::MeshCoordinateRange coord_range(mesh_device_->shape());

    uint32_t rows = mesh_device_->num_rows();
    uint32_t cols = mesh_device_->num_cols();
    uint32_t num_devices = rows * cols;

    for (auto shard_orientation : {ShardOrientation::COL_MAJOR, ShardOrientation::ROW_MAJOR}) {
        for (int i = 0; i < NUM_ITERS; i++) {
            Shape2D global_buffer_shape = {
                gen_num_datums(rng) * constants::TILE_HEIGHT * rows,
                gen_num_datums(rng) * constants::TILE_WIDTH * cols};
            Shape2D shard_shape = {global_buffer_shape.height() / rows, global_buffer_shape.width() / cols};
            uint32_t global_buffer_size = global_buffer_shape.height() * global_buffer_shape.width() * sizeof(uint32_t);
            ShardedBufferConfig sharded_config{
                .global_size = global_buffer_size,
                .global_buffer_shape = global_buffer_shape,
                .shard_shape = shard_shape,
                .shard_orientation = shard_orientation,
            };
            auto mesh_buffer = MeshBuffer::create(sharded_config, per_device_buffer_config, mesh_device_.get());

            std::vector<uint32_t> src_vec =
                std::vector<uint32_t>(global_buffer_size / num_devices / sizeof(uint32_t), 0);
            std::iota(src_vec.begin(), src_vec.end(), i);
            std::unordered_map<distributed::MeshCoordinate, std::vector<uint32_t>> dst_vec = {};
            std::vector<distributed::ShardDataTransfer> input_shards = {};
            std::vector<distributed::ShardDataTransfer> output_shards = {};

            for (const auto& coord : coord_range) {
                input_shards.push_back(distributed::ShardDataTransfer{coord}.host_data(src_vec.data()));
            }
            for (const auto& coord : coord_range) {
                dst_vec[coord] = std::vector<uint32_t>(global_buffer_size / num_devices / sizeof(uint32_t), 0);
                output_shards.push_back(distributed::ShardDataTransfer{coord}.host_data(dst_vec[coord].data()));
            }

            mesh_device_->mesh_command_queue().enqueue_write_shards(mesh_buffer, input_shards, false);
            mesh_device_->mesh_command_queue().enqueue_read_shards(output_shards, mesh_buffer, true);

            for (auto& dst : dst_vec) {
                EXPECT_EQ(dst.second, src_vec);
            }
        }
    }
}

TEST_F(MeshBufferTestSuite, MultiShardReadWriteMultiThread) {
    constexpr uint32_t NUM_ITERS = 50;
    uint32_t seed = tt::parse_env("TT_METAL_SEED", 0);
    uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);

    std::vector<std::thread> threads;
    for (int thread_idx = 0; thread_idx < 2; thread_idx += 1) {
        threads.push_back(std::thread([&]() {
            std::uniform_int_distribution<int> gen_num_datums(32, 128);
            std::mt19937 rng(seed);

            DeviceLocalBufferConfig per_device_buffer_config{
                .page_size = single_tile_size, .buffer_type = BufferType::DRAM, .bottom_up = true};

            distributed::MeshCoordinateRange coord_range(mesh_device_->shape());

            uint32_t rows = mesh_device_->num_rows();
            uint32_t cols = mesh_device_->num_cols();
            uint32_t num_devices = rows * cols;

            for (auto shard_orientation : {ShardOrientation::COL_MAJOR, ShardOrientation::ROW_MAJOR}) {
                for (int i = 0; i < NUM_ITERS; i++) {
                    Shape2D global_buffer_shape = {
                        gen_num_datums(rng) * constants::TILE_HEIGHT * rows,
                        gen_num_datums(rng) * constants::TILE_WIDTH * cols};
                    Shape2D shard_shape = {global_buffer_shape.height() / rows, global_buffer_shape.width() / cols};
                    uint32_t global_buffer_size =
                        global_buffer_shape.height() * global_buffer_shape.width() * sizeof(uint32_t);
                    ShardedBufferConfig sharded_config{
                        .global_size = global_buffer_size,
                        .global_buffer_shape = global_buffer_shape,
                        .shard_shape = shard_shape,
                        .shard_orientation = shard_orientation,
                    };
                    auto mesh_buffer = MeshBuffer::create(sharded_config, per_device_buffer_config, mesh_device_.get());

                    std::vector<uint32_t> src_vec =
                        std::vector<uint32_t>(global_buffer_size / num_devices / sizeof(uint32_t), 0);
                    std::iota(src_vec.begin(), src_vec.end(), i);
                    std::unordered_map<distributed::MeshCoordinate, std::vector<uint32_t>> dst_vec = {};
                    std::vector<distributed::ShardDataTransfer> input_shards = {};
                    std::vector<distributed::ShardDataTransfer> output_shards = {};

                    for (const auto& coord : coord_range) {
                        input_shards.push_back(distributed::ShardDataTransfer{coord}.host_data(src_vec.data()));
                    }
                    for (const auto& coord : coord_range) {
                        dst_vec[coord] = std::vector<uint32_t>(global_buffer_size / num_devices / sizeof(uint32_t), 0);
                        output_shards.push_back(distributed::ShardDataTransfer{coord}.host_data(dst_vec[coord].data()));
                    }

                    mesh_device_->mesh_command_queue().enqueue_write_shards(mesh_buffer, input_shards, false);
                    mesh_device_->mesh_command_queue().enqueue_read_shards(output_shards, mesh_buffer, true);

                    for (auto& dst : dst_vec) {
                        EXPECT_EQ(dst.second, src_vec);
                    }
                }
            }
        }));
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

TEST_F(MeshBufferTestSuite, EnqueueReadShardsWithPinnedMemoryFullRange) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
        return;
    }
    uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);

    // Use a replicated mesh buffer so per-device buffers are interleaved (not sharded)
    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size, .buffer_type = BufferType::DRAM, .bottom_up = true};

    // Make the buffer multiple tiles to exercise multi-page transfers
    const uint32_t tiles_per_device = 128;
    const uint32_t bytes_per_device = tiles_per_device * single_tile_size;

    ReplicatedBufferConfig global_buffer_config{.size = bytes_per_device};
    auto mesh_buffer = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());

    // Prepare source data and write it to a single shard (0,0)
    std::vector<uint32_t> src(bytes_per_device / sizeof(uint32_t), 0);
    std::iota(src.begin(), src.end(), 0);

    distributed::MeshCoordinate coord(0, 0);
    auto write_transfer = distributed::ShardDataTransfer{coord}
                              .host_data(static_cast<void*>(const_cast<uint32_t*>(src.data())))
                              .region(BufferRegion(0, bytes_per_device));
    mesh_device_->mesh_command_queue().enqueue_write_shards(mesh_buffer, {write_transfer}, /*blocking=*/true);

    // Prepare destination buffer and pin the entire destination range for the target shard
    auto dst = std::make_shared<vector_aligned<uint32_t>>(bytes_per_device / sizeof(uint32_t), 0);
    uint32_t* dst_ptr_aligned = reinterpret_cast<uint32_t*>(dst->data());

    // Create HostBuffer on top of dst
    HostBuffer host_buffer(
        tt::stl::Span<uint32_t>(dst_ptr_aligned, bytes_per_device / sizeof(uint32_t)), MemoryPin(dst));

    auto coordinate_range_set = MeshCoordinateRangeSet(MeshCoordinateRange(coord, coord));
    auto pinned_unique = experimental::PinnedMemory::Create(
        *mesh_device_,
        coordinate_range_set,
        host_buffer,
        /*map_to_noc=*/true);
    std::shared_ptr<experimental::PinnedMemory> pinned_shared = std::move(pinned_unique);

    // Read back using enqueue_read_shards with pinned_memory populated
    auto read_transfer = distributed::ShardDataTransfer{coord}
                             .host_data(static_cast<void*>(dst_ptr_aligned))
                             .region(BufferRegion(0, bytes_per_device));
    experimental::ShardDataTransferSetPinnedMemory(read_transfer, pinned_shared);
    mesh_device_->mesh_command_queue().enqueue_read_shards({read_transfer}, mesh_buffer, /*blocking=*/true);

    std::vector<uint32_t> dst_aligned(dst_ptr_aligned, dst_ptr_aligned + (bytes_per_device / sizeof(uint32_t)));

    EXPECT_EQ(dst_aligned, src);
}

TEST_F(MeshBufferTestSuite, EnqueueReadWithDistributedHostBufferAndPinnedMemory) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
        return;
    }
    uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);

    // Use a replicated mesh buffer so per-device buffers are interleaved (not sharded)
    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size, .buffer_type = BufferType::DRAM, .bottom_up = true};

    // Make the buffer multiple tiles to exercise multi-page transfers
    const uint32_t tiles_per_device = 128;
    const uint32_t bytes_per_device = tiles_per_device * single_tile_size;

    ReplicatedBufferConfig global_buffer_config{.size = bytes_per_device};
    auto mesh_buffer = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());

    // Prepare source data and write it to a single shard (0,0)
    std::vector<uint32_t> src(bytes_per_device / sizeof(uint32_t), 0);
    std::iota(src.begin(), src.end(), 0);

    distributed::MeshCoordinate coord(0, 0);
    auto write_transfer = distributed::ShardDataTransfer{coord}
                              .host_data(static_cast<void*>(const_cast<uint32_t*>(src.data())))
                              .region(BufferRegion(0, bytes_per_device));
    mesh_device_->mesh_command_queue().enqueue_write_shards(mesh_buffer, {write_transfer}, /*blocking=*/true);

    // Prepare destination buffer and pin the entire destination range for the target shard
    auto dst = std::make_shared<vector_aligned<uint32_t>>(bytes_per_device / sizeof(uint32_t), 0);
    uint32_t* dst_ptr_aligned = reinterpret_cast<uint32_t*>(dst->data());

    // Create HostBuffer on top of dst
    HostBuffer host_buffer(
        tt::stl::Span<uint32_t>(dst_ptr_aligned, bytes_per_device / sizeof(uint32_t)), MemoryPin(dst));

    auto coordinate_range_set = MeshCoordinateRangeSet(MeshCoordinateRange(coord, coord));
    auto pinned_unique = experimental::PinnedMemory::Create(
        *mesh_device_,
        coordinate_range_set,
        host_buffer,
        /*map_to_noc=*/true);
    std::shared_ptr<experimental::PinnedMemory> pinned_shared = std::move(pinned_unique);

    // Attach pinned memory to HostBuffer
    experimental::HostBufferSetPinnedMemory(host_buffer, pinned_shared);

    // Create DistributedHostBuffer and add the HostBuffer as a shard
    auto distributed_host_buffer = DistributedHostBuffer::create(mesh_device_->shape());
    distributed_host_buffer.emplace_shard(coord, [&host_buffer]() { return host_buffer; });

    // Read back using enqueue_read with DistributedHostBuffer
    mesh_device_->mesh_command_queue().enqueue_read(
        mesh_buffer, distributed_host_buffer, std::nullopt, /*blocking=*/true);

    std::vector<uint32_t> dst_aligned(dst_ptr_aligned, dst_ptr_aligned + (bytes_per_device / sizeof(uint32_t)));

    EXPECT_EQ(dst_aligned, src);
}

TEST_F(MeshBufferTestSuite, EnqueueReadShardsWithPinnedMemoryFullRangeUnaligned) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
        return;
    }
    uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);

    // Use a replicated mesh buffer so per-device buffers are interleaved (not sharded)
    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size, .buffer_type = BufferType::DRAM, .bottom_up = true};

    // Make the buffer multiple tiles to exercise multi-page transfers
    const uint32_t tiles_per_device = 128;
    const uint32_t bytes_per_device = tiles_per_device * single_tile_size;

    ReplicatedBufferConfig global_buffer_config{.size = bytes_per_device};
    auto mesh_buffer = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());

    // Prepare source data and write it to a single shard (0,0)
    std::vector<uint8_t> src(bytes_per_device, 0);
    std::iota(src.begin(), src.end(), 0);

    distributed::MeshCoordinate coord(0, 0);
    auto write_transfer =
        distributed::ShardDataTransfer{coord}.host_data(src.data()).region(BufferRegion(0, bytes_per_device));
    mesh_device_->mesh_command_queue().enqueue_write_shards(mesh_buffer, {write_transfer}, /*blocking=*/true);

    constexpr size_t unaligned_shift = 3;
    // Prepare destination buffer and pin the entire destination range for the target shard
    auto dst = std::make_shared<vector_aligned<uint8_t>>(bytes_per_device + unaligned_shift, 0);
    uint8_t* dst_ptr_unaligned = dst->data() + unaligned_shift;

    // Create HostBuffer on top of dst
    HostBuffer host_buffer(tt::stl::Span<uint8_t>(dst_ptr_unaligned, bytes_per_device), MemoryPin(dst));

    auto coordinate_range_set = MeshCoordinateRangeSet(MeshCoordinateRange(coord, coord));
    auto pinned_unique = experimental::PinnedMemory::Create(
        *mesh_device_,
        coordinate_range_set,
        host_buffer,
        /*map_to_noc=*/true);
    std::shared_ptr<experimental::PinnedMemory> pinned_shared = std::move(pinned_unique);

    // Read back using enqueue_read_shards with pinned_memory populated
    auto read_transfer = distributed::ShardDataTransfer{coord}
                             .host_data(static_cast<void*>(dst_ptr_unaligned))
                             .region(BufferRegion(0, bytes_per_device));
    mesh_device_->mesh_command_queue().enqueue_read_shards({read_transfer}, mesh_buffer, /*blocking=*/true);

    std::vector<uint8_t> dst_aligned(dst_ptr_unaligned, dst_ptr_unaligned + bytes_per_device);

    EXPECT_EQ(dst_aligned, src);
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
