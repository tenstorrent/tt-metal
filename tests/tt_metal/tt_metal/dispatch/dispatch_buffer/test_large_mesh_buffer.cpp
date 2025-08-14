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
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <cstring>
#include "tt_metal/impl/dispatch/kernels/cq_commands.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

using MeshBufferTest2x4 = MeshDevice2x4Fixture;
using MeshBufferTestSuite = GenericMeshDeviceFixture;

class ReplicatedMeshBufferTestSuite : public MeshBufferTestSuite,
                                      public testing::WithParamInterface<std::tuple<uint64_t, uint32_t>> {
protected:
    static constexpr uint64_t max_num_elements_ = 4ull << 30;  // 8GB
    inline static std::vector<uint16_t> src_vec_;

    ReplicatedMeshBufferTestSuite() {
        if (src_vec_.empty()) {
            std::cout << "ReplicatedMeshBufferTestSuite ctor: initializing src_vec_" << std::endl;
            src_vec_.resize(max_num_elements_);
            std::iota(src_vec_.begin(), src_vec_.end(), 0);
            // std::shuffle(src_vec_.begin(), src_vec_.end(), std::mt19937(std::random_device{}()));
        }
    }
};

TEST_P(ReplicatedMeshBufferTestSuite, DRAMReadback) {
    // - REPLICATED layout for writing, SHARDED with ROW_MAJOR for reading
    // - DRAM, bottom up allocation
    auto [tensor_size, page_size] = GetParam();
    // auto num_devices = mesh_device_->num_devices();
    tensor_size = tensor_size * sizeof(uint16_t);

    const DeviceLocalBufferConfig device_local_config{
        .page_size = page_size, .buffer_type = BufferType::DRAM, .bottom_up = true};
    constexpr auto address = 32;

    // const auto dram_alignment = MetalContext::instance().hal().get_alignment(HalMemType::DRAM);
    // uint32_t length_adjust = ((std::random_device{}() % page_size) & ~(dram_alignment - 1)) &
    // CQ_PREFETCH_RELAY_PAGED_LENGTH_ADJUST_MASK;
    // tensor_size -= length_adjust;
    std::cout << "page_size: " << page_size << " tensor_size/device (MB): " << tensor_size / (1 << 20) << std::endl;

    const ReplicatedBufferConfig buffer_config{.size = tensor_size};

    // Create replicated buffer for writing with specific address 32
    auto mesh_buffer = MeshBuffer::create(buffer_config, device_local_config, mesh_device_.get(), address);
    // Verify buffer properties
    EXPECT_EQ(mesh_buffer->size(), tensor_size);
    EXPECT_EQ(mesh_buffer->global_layout(), MeshBufferLayout::REPLICATED);
    EXPECT_EQ(mesh_buffer->device_local_size(), tensor_size);
    EXPECT_EQ(mesh_buffer->address(), address);
    EXPECT_TRUE(mesh_buffer->is_allocated());

    // Create test data - use uint16_t for easy verification
    auto num_elements = tensor_size / sizeof(uint16_t);
    assert(num_elements <= max_num_elements_);
    std::vector<uint16_t> src_vec(src_vec_.begin(), src_vec_.begin() + num_elements);

    distributed::MeshCoordinateRange coord_range(mesh_device_->shape());
    std::vector<MeshCommandQueue::ShardDataTransfer> input_shards = {};
    for (auto& coord : coord_range) {
        input_shards.push_back({coord, src_vec.data()});
    }

    std::unordered_map<distributed::MeshCoordinate, std::vector<uint16_t>> dst_vec = {};
    std::vector<MeshCommandQueue::ShardDataTransfer> output_shards = {};
    for (auto& coord : coord_range) {
        dst_vec[coord] = std::vector<uint16_t>(num_elements, 0);
        output_shards.push_back({coord, dst_vec[coord].data()});
    }

    mesh_device_->mesh_command_queue().enqueue_write_shards(mesh_buffer, input_shards, false);
    mesh_device_->mesh_command_queue().enqueue_read_shards(output_shards, mesh_buffer, true);

    for (auto& dst : dst_vec) {
        EXPECT_EQ(dst.second, src_vec);
    }
}

INSTANTIATE_TEST_SUITE_P(
    LargeReplicatedReadback,
    ReplicatedMeshBufferTestSuite,
    ::testing::Combine(
        ::testing::Values(2ull << 30, 4ull << 30),        // tensor sizes
        ::testing::Values(1024, 2048, 16 << 10, 1 << 20)  // page sizes
        ));

class ShardedMeshBufferTestSuite
    : public MeshBufferTestSuite,
      public testing::WithParamInterface<std::tuple<Shape2D, uint32_t, TensorMemoryLayout>> {
protected:
    static constexpr uint64_t max_num_elements_ = 4ull << 30;  // 8GB
    inline static std::vector<uint16_t> src_vec_;

    ShardedMeshBufferTestSuite() {
        if (src_vec_.empty()) {
            std::cout << "ShardedMeshBufferTestSuite ctor: initializing src_vec_" << std::endl;
            src_vec_.resize(max_num_elements_);
            std::iota(src_vec_.begin(), src_vec_.end(), 0);
            // std::shuffle(src_vec_.begin(), src_vec_.end(), std::mt19937(std::random_device{}()));
        }
    }
};

TEST_P(ShardedMeshBufferTestSuite, DRAMReadback) {
    auto [shard_shape, page_size, tensor_layout] = GetParam();

    log_info(
        tt::LogTest,
        "shard_shape: {}, page_size: {}, tensor_layout: {}, device buffer size: {} MB",
        shard_shape,
        page_size,
        tensor_layout,
        shard_shape.height() * shard_shape.width() * sizeof(uint16_t) / (1 << 20));

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = page_size, .buffer_type = BufferType::DRAM, .bottom_up = true};

    distributed::MeshCoordinateRange coord_range(mesh_device_->shape());

    uint32_t rows = mesh_device_->num_rows();
    uint32_t cols = mesh_device_->num_cols();
    uint32_t num_devices = rows * cols;

    // Ensure buffer dimensions are divisible by tile dimensions
    ASSERT_TRUE(shard_shape.height() % constants::TILE_HEIGHT == 0);
    ASSERT_TRUE(shard_shape.width() % constants::TILE_WIDTH == 0);

    auto shard_orientation = ShardOrientation::ROW_MAJOR;

    // Configure so that every device loads the specified buffer size
    Shape2D tensor_shape = {shard_shape.height() * rows, shard_shape.width() * cols};
    uint64_t tensor_size = num_devices * shard_shape.height() * shard_shape.width() * sizeof(uint16_t);
    ShardedBufferConfig sharded_config{
        .global_size = tensor_size,
        .global_buffer_shape = tensor_shape,
        .shard_shape = shard_shape,
        .shard_orientation = shard_orientation,
    };
    auto mesh_buffer = MeshBuffer::create(sharded_config, per_device_buffer_config, mesh_device_.get());

    auto num_elements = tensor_size / num_devices / sizeof(uint16_t);
    assert(num_elements <= max_num_elements_);
    std::vector<uint16_t> src_vec(src_vec_.begin(), src_vec_.begin() + num_elements);
    std::vector<MeshCommandQueue::ShardDataTransfer> input_shards = {};
    for (auto& coord : coord_range) {
        input_shards.push_back({coord, src_vec.data()});
    }

    std::unordered_map<distributed::MeshCoordinate, std::vector<uint16_t>> dst_vec = {};
    std::vector<MeshCommandQueue::ShardDataTransfer> output_shards = {};
    for (auto& coord : coord_range) {
        dst_vec[coord] = std::vector<uint16_t>(tensor_size / num_devices / sizeof(uint16_t), 0);
        output_shards.push_back({coord, dst_vec[coord].data()});
    }

    mesh_device_->mesh_command_queue().enqueue_write_shards(mesh_buffer, input_shards, false);
    mesh_device_->mesh_command_queue().enqueue_read_shards(output_shards, mesh_buffer, true);

    for (auto& dst : dst_vec) {
        EXPECT_EQ(dst.second, src_vec);
    }
}

INSTANTIATE_TEST_SUITE_P(
    LargeShardedReadback,
    ShardedMeshBufferTestSuite,
    ::testing::Combine(
        // shard_shape, page_size, tensor_layout
        ::testing::Values(
            Shape2D(1 << 15, 1 << 15),                     // 2 GB with Uint16
            Shape2D(1 << 15, 1 << 16),                     // 4 GB with Uint16
            Shape2D(1 << 16, 1 << 16)),                    // 8 GB with Uint16
        ::testing::Values(1024, 2048, 16 << 10, 1 << 20),  // page size
        ::testing::Values(TensorMemoryLayout::BLOCK_SHARDED)));

}  // namespace
}  // namespace tt::tt_metal::distributed::test
