// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <boost/move/utility_core.hpp>
#include <gtest/gtest.h>
#include <stdint.h>
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
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/util.hpp>

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

    Shape2D tensor2d_shape_in_pages() {
        return {num_pages_per_core.height() * num_cores.height(), num_pages_per_core.width() * num_cores.width()};
    }

    uint32_t num_pages() { return tensor2d_shape_in_pages().height() * tensor2d_shape_in_pages().width(); }

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
    uint32_t single_tile_size = ::tt::tt_metal::detail::TileSize(DataFormat::UInt32);

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
    uint32_t single_tile_size = ::tt::tt_metal::detail::TileSize(DataFormat::UInt32);

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
                    if (!mesh_device_->is_local(MeshCoordinate(logical_y, logical_x))) {
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

TEST_F(MeshBufferTestSuite, InterleavedShardsReadWriteWithFloat32ToBfloat16Conversion) {
    constexpr uint32_t NUM_ITERS = 100;
    uint32_t seed = tt::parse_env("TT_METAL_SEED", 0);
    uint32_t single_tile_size = ::tt::tt_metal::detail::TileSize(DataFormat::UInt32);

    for (auto buffer_type : {BufferType::L1, BufferType::DRAM}) {
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

            std::vector<float> src_vec(num_random_tiles * single_tile_size / sizeof(uint16_t), 0);
            // Initialize src_vec with random values
            std::generate(src_vec.begin(), src_vec.end(), [&rng]() { return static_cast<float>(rng() % 100); });

            for (std::size_t logical_x = 0; logical_x < buf->device()->num_cols(); logical_x++) {
                for (std::size_t logical_y = 0; logical_y < buf->device()->num_rows(); logical_y++) {
                    WriteShardWithConversion(
                        mesh_device_->mesh_command_queue(),
                        buf,
                        DataFormat::Float16_b,
                        src_vec,
                        DataFormat::Float32,
                        MeshCoordinate(logical_y, logical_x));
                }
            }
            // Generate golden vector
            std::vector<bfloat16> golden_vec(num_random_tiles * single_tile_size / sizeof(uint16_t), 0);
            std::transform(src_vec.begin(), src_vec.end(), golden_vec.begin(), [](float f) { return bfloat16(f); });
            std::vector<uint16_t> golden_vec_uint16(num_random_tiles * single_tile_size / sizeof(uint16_t), 0);
            std::transform(golden_vec.begin(), golden_vec.end(), golden_vec_uint16.begin(), [](bfloat16 bf16) {
                return bf16.to_packed();
            });

            for (std::size_t logical_x = 0; logical_x < buf->device()->num_cols(); logical_x++) {
                for (std::size_t logical_y = 0; logical_y < buf->device()->num_rows(); logical_y++) {
                    std::vector<uint16_t> dst_vec = {};
                    ReadShard(mesh_device_->mesh_command_queue(), dst_vec, buf, MeshCoordinate(logical_y, logical_x));

                    EXPECT_EQ(dst_vec, golden_vec_uint16);
                }
            }
        }
    }
}

TEST_F(MeshBufferTestSuite, DistributedHostBufferReadWriteWithFloat32ToBfloat16Conversion) {
    constexpr uint32_t NUM_ITERS = 100;
    uint32_t seed = tt::parse_env("TT_METAL_SEED", 0);
    uint32_t single_tile_size = ::tt::tt_metal::detail::TileSize(DataFormat::UInt32);

    for (auto buffer_type : {BufferType::L1, BufferType::DRAM}) {
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

            std::vector<float> src_vec(num_random_tiles * single_tile_size / sizeof(uint16_t), 0);
            // Initialize src_vec with random values
            std::generate(src_vec.begin(), src_vec.end(), [&rng]() { return static_cast<float>(rng() % 100); });
            DistributedHostBuffer host_buffer = DistributedHostBuffer::create(mesh_device_->get_view());
            std::shared_ptr<std::vector<float>> shared_src_vec = std::make_shared<std::vector<float>>(src_vec);

            for (std::size_t logical_x = 0; logical_x < buf->device()->num_cols(); logical_x++) {
                for (std::size_t logical_y = 0; logical_y < buf->device()->num_rows(); logical_y++) {
                    host_buffer.emplace_shard(MeshCoordinate(logical_y, logical_x), [shared_src_vec]() {
                        return HostBuffer(shared_src_vec);
                    });
                }
            }
            mesh_device_->mesh_command_queue().enqueue_write_with_conversion(
                buf, DataFormat::Float16_b, host_buffer, DataFormat::Float32, false);
            // Generate golden vector
            std::vector<bfloat16> golden_vec(num_random_tiles * single_tile_size / sizeof(uint16_t), 0);
            std::transform(src_vec.begin(), src_vec.end(), golden_vec.begin(), [](float f) { return bfloat16(f); });
            std::vector<uint16_t> golden_vec_uint16(num_random_tiles * single_tile_size / sizeof(uint16_t), 0);
            std::transform(golden_vec.begin(), golden_vec.end(), golden_vec_uint16.begin(), [](bfloat16 bf16) {
                return bf16.to_packed();
            });

            for (std::size_t logical_x = 0; logical_x < buf->device()->num_cols(); logical_x++) {
                for (std::size_t logical_y = 0; logical_y < buf->device()->num_rows(); logical_y++) {
                    std::vector<uint16_t> dst_vec = {};
                    ReadShard(mesh_device_->mesh_command_queue(), dst_vec, buf, MeshCoordinate(logical_y, logical_x));

                    EXPECT_EQ(dst_vec, golden_vec_uint16);
                }
            }
        }
    }
}

TEST_F(MeshBufferTestSuite, ShardingTestWithFloat32ToBfloat16Conversion) {
    std::array<uint32_t, 2> num_pages_per_core = {1, 1};
    std::array<uint32_t, 2> page_shape = {1, 1024};
    auto shard_strategy = TensorMemoryLayout::HEIGHT_SHARDED;

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
    //   std::vector<uint32_t> src_vec(buf_size / sizeof(uint32_t), 0);
    //  std::iota(src_vec.begin(), src_vec.end(), 0);
    std::vector<float> src_vec(buf_size / sizeof(uint16_t), 0);
    std::iota(src_vec.begin(), src_vec.end(), 0);

    for (std::size_t logical_x = 0; logical_x < buf->device()->num_cols(); logical_x++) {
        for (std::size_t logical_y = 0; logical_y < buf->device()->num_rows(); logical_y++) {
            WriteShardWithConversion(
                mesh_device_->mesh_command_queue(),
                buf,
                DataFormat::Float16_b,
                src_vec,
                DataFormat::Float32,
                MeshCoordinate(logical_y, logical_x));
        }
    }
    std::vector<bfloat16> golden_vec(buf_size / sizeof(uint16_t), 0);
    std::transform(src_vec.begin(), src_vec.end(), golden_vec.begin(), [](float f) { return bfloat16(f); });
    std::vector<uint16_t> golden_vec_uint16(buf_size / sizeof(uint16_t), 0);
    std::transform(golden_vec.begin(), golden_vec.end(), golden_vec_uint16.begin(), [](bfloat16 bf16) {
        return bf16.to_packed();
    });

    for (std::size_t logical_x = 0; logical_x < buf->device()->num_cols(); logical_x++) {
        for (std::size_t logical_y = 0; logical_y < buf->device()->num_rows(); logical_y++) {
            std::vector<uint16_t> dst_vec = {};
            ReadShard(mesh_device_->mesh_command_queue(), dst_vec, buf, MeshCoordinate(logical_y, logical_x));
            EXPECT_EQ(dst_vec, golden_vec_uint16);
        }
    }
}

TEST_F(MeshBufferTestSuite, RowMajorShardingAndReplication) {
    uint32_t single_tile_size = ::tt::tt_metal::detail::TileSize(DataFormat::UInt32);

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size, .buffer_type = BufferType::DRAM, .bottom_up = true};

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

TEST_F(MeshBufferTestSuite, RowMajorShardingAndReplicationWithFloat32ToBfloat16Conversion) {
    uint32_t single_tile_size = ::tt::tt_metal::detail::TileSize(DataFormat::UInt32);

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size, .buffer_type = BufferType::DRAM, .bottom_up = true};

    std::vector<Shape2D> global_buffer_shapes = {{64, 256}, {128, 128}, {256, 2048}, {32, 512}, {512, 1024}};

    for (int i = 0; i < global_buffer_shapes.size(); i++) {
        auto global_buffer_shape = global_buffer_shapes[i];
        Shape2D shard_shape = {0, global_buffer_shape.width() / mesh_device_->num_cols()};
        // Mesh-Level Sharding Parameters for the MeshBufferView that will be read to verify correctness
        Shape2D global_buffer_read_shape = {
            global_buffer_shape.height() * mesh_device_->num_rows(), global_buffer_shape.width()};
        Shape2D shard_read_shape = {
            global_buffer_shape.height(), global_buffer_shape.width() / mesh_device_->num_cols()};

        uint32_t global_buffer_size = global_buffer_shape.height() * global_buffer_shape.width() * sizeof(uint16_t);
        auto shard_orientation = ShardOrientation::ROW_MAJOR;

        ShardedBufferConfig sharded_config{
            .global_size = global_buffer_size,
            .global_buffer_shape = global_buffer_shape,
            .shard_shape = shard_shape,
            .shard_orientation = shard_orientation,
        };
        // Initialize the ShardedBufferConfig for reading and verifying replicated data
        ShardedBufferConfig sharded_read_view_config{
            .global_size = global_buffer_read_shape.height() * global_buffer_read_shape.width() * sizeof(uint16_t),
            .global_buffer_shape = global_buffer_read_shape,
            .shard_shape = shard_read_shape,
            .shard_orientation = shard_orientation};

        auto mesh_buffer = MeshBuffer::create(sharded_config, per_device_buffer_config, mesh_device_.get());
        std::vector<float> src_vec = std::vector<float>(global_buffer_shape.height() * global_buffer_shape.width(), 0);
        std::iota(src_vec.begin(), src_vec.end(), 0);

        auto mesh_buffer_read_view = MeshBuffer::create(
            sharded_read_view_config, per_device_buffer_config, mesh_device_.get(), mesh_buffer->address());
        EnqueueWriteMeshBufferWithConversion(
            mesh_device_->mesh_command_queue(), mesh_buffer, DataFormat::Float16_b, src_vec, DataFormat::Float32);
        std::vector<bfloat16> golden_vec(src_vec.size(), 0);
        std::transform(src_vec.begin(), src_vec.end(), golden_vec.begin(), [](float f) { return bfloat16(f); });
        std::vector<uint16_t> golden_vec_uint16(src_vec.size(), 0);
        std::transform(golden_vec.begin(), golden_vec.end(), golden_vec_uint16.begin(), [](bfloat16 bf16) {
            return bf16.to_packed();
        });
        std::vector<uint16_t> dst_vec = std::vector<uint16_t>(src_vec.size(), 0);
        EnqueueReadMeshBuffer(mesh_device_->mesh_command_queue(), dst_vec, mesh_buffer_read_view);

        EXPECT_EQ(dst_vec, golden_vec_uint16);
    }
}

TEST_F(MeshBufferTestSuite, ColMajorShardingAndReplication) {
    uint32_t single_tile_size = ::tt::tt_metal::detail::TileSize(DataFormat::UInt32);

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size, .buffer_type = BufferType::DRAM, .bottom_up = true};

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

TEST_F(MeshBufferTestSuite, ColMajorShardingAndReplicationWithFloat32ToBfloat16Conversion) {
    uint32_t single_tile_size = ::tt::tt_metal::detail::TileSize(DataFormat::UInt32);

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size, .buffer_type = BufferType::DRAM, .bottom_up = true};

    std::vector<Shape2D> global_buffer_shapes = {{256, 64}, {1024, 1024}, {128, 32}, {512, 64}, {2048, 256}};

    for (int i = 0; i < global_buffer_shapes.size(); i++) {
        auto global_buffer_shape = global_buffer_shapes[i];
        Shape2D shard_shape = {global_buffer_shape.height() / mesh_device_->num_rows(), 0};
        uint32_t global_buffer_size = global_buffer_shape.height() * global_buffer_shape.width() * sizeof(uint16_t);
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
            .global_size = global_buffer_read_shape.height() * global_buffer_read_shape.width() * sizeof(uint16_t),
            .global_buffer_shape = global_buffer_read_shape,
            .shard_shape = shard_read_shape,
            .shard_orientation = ShardOrientation::ROW_MAJOR};

        auto mesh_buffer = MeshBuffer::create(sharded_config, per_device_buffer_config, mesh_device_.get());
        std::vector<float> src_vec = std::vector<float>(global_buffer_shape.height() * global_buffer_shape.width(), 0);
        std::iota(src_vec.begin(), src_vec.end(), 0);

        auto mesh_buffer_read_view = MeshBuffer::create(
            sharded_read_view_config, per_device_buffer_config, mesh_device_.get(), mesh_buffer->address());

        EnqueueWriteMeshBufferWithConversion(
            mesh_device_->mesh_command_queue(), mesh_buffer, DataFormat::Float16_b, src_vec, DataFormat::Float32);

        std::vector<bfloat16> golden_vec(src_vec.size(), 0);
        std::transform(src_vec.begin(), src_vec.end(), golden_vec.begin(), [](float f) { return bfloat16(f); });
        std::vector<uint16_t> golden_vec_uint16(src_vec.size(), 0);
        std::transform(golden_vec.begin(), golden_vec.end(), golden_vec_uint16.begin(), [](bfloat16 bf16) {
            return bf16.to_packed();
        });

        std::vector<uint16_t> dst_vec = std::vector<uint16_t>(src_vec.size(), 0);
        EnqueueReadMeshBuffer(mesh_device_->mesh_command_queue(), dst_vec, mesh_buffer_read_view);

        for (int i = 0; i < dst_vec.size(); i++) {
            uint32_t expected_index =
                (i / global_buffer_read_shape.width()) * global_buffer_shape.width() + i % global_buffer_shape.width();
            EXPECT_EQ(golden_vec_uint16[expected_index], dst_vec[i]);
        }
    }
}

TEST_F(MeshBufferTestSuite, MultiShardReadWrite) {
    constexpr uint32_t NUM_ITERS = 50;
    uint32_t seed = tt::parse_env("TT_METAL_SEED", 0);
    uint32_t single_tile_size = ::tt::tt_metal::detail::TileSize(DataFormat::UInt32);

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
            std::vector<MeshCommandQueue::ShardDataTransfer> input_shards = {};
            std::vector<MeshCommandQueue::ShardDataTransfer> output_shards = {};

            for (auto& coord : coord_range) {
                input_shards.push_back({coord, src_vec.data()});
            }
            for (auto& coord : coord_range) {
                dst_vec[coord] = std::vector<uint32_t>(global_buffer_size / num_devices / sizeof(uint32_t), 0);
                output_shards.push_back({coord, dst_vec[coord].data()});
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
    uint32_t single_tile_size = ::tt::tt_metal::detail::TileSize(DataFormat::UInt32);

    std::vector<std::thread> threads;
    for (int thread_idx = 0; thread_idx < 2; thread_idx += 1) {
        threads.push_back(std::thread([&, thread_idx]() {
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
                    std::vector<MeshCommandQueue::ShardDataTransfer> input_shards = {};
                    std::vector<MeshCommandQueue::ShardDataTransfer> output_shards = {};

                    for (auto& coord : coord_range) {
                        input_shards.push_back({coord, src_vec.data()});
                    }
                    for (auto& coord : coord_range) {
                        dst_vec[coord] = std::vector<uint32_t>(global_buffer_size / num_devices / sizeof(uint32_t), 0);
                        output_shards.push_back({coord, dst_vec[coord].data()});
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

}  // namespace
}  // namespace tt::tt_metal::distributed::test
