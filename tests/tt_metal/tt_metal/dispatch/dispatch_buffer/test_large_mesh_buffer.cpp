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
using ElementType = uint64_t;
constexpr uint32_t ElementSize = sizeof(ElementType);

class LargeMeshBufferTestSuiteBase : public MeshBufferTestSuite {
protected:
    static constexpr uint64_t max_num_elements_ = (8ull << 30) / ElementSize;  // 8GB
    // A large datatype is used to reduce random shuffle time
    inline static std::vector<ElementType> src_vec_;

    LargeMeshBufferTestSuiteBase() {
        if (src_vec_.empty()) {
            log_info(tt::LogTest, "LargeMeshBufferTestSuiteBase ctor: initializing src_vec_");
            src_vec_.resize(max_num_elements_);
            std::iota(src_vec_.begin(), src_vec_.end(), 0);
            std::shuffle(src_vec_.begin(), src_vec_.end(), std::mt19937(std::random_device{}()));
        }
    }
};

class ReplicatedMeshBufferTestSuite : public LargeMeshBufferTestSuiteBase,
                                      public testing::WithParamInterface<std::tuple<uint64_t, uint32_t>> {};

TEST_P(ReplicatedMeshBufferTestSuite, DRAMReadback) {
    // - REPLICATED layout for writing, SHARDED with ROW_MAJOR for reading
    // - DRAM, bottom up allocation
    auto [tensor_size, page_size] = GetParam();
    // auto num_devices = mesh_device_->num_devices();
    tensor_size = tensor_size * ElementSize;

    const DeviceLocalBufferConfig device_local_config{
        .page_size = page_size, .buffer_type = BufferType::DRAM, .bottom_up = true};
    constexpr auto address = 32;

    // const auto dram_alignment = MetalContext::instance().hal().get_alignment(HalMemType::DRAM);
    // uint32_t length_adjust = ((std::random_device{}() % page_size) & ~(dram_alignment - 1)) &
    // CQ_PREFETCH_RELAY_PAGED_LENGTH_ADJUST_MASK;
    // tensor_size -= length_adjust;
    log_info(tt::LogTest, "page_size: {}, tensor_size/device (MB): {}", page_size, tensor_size / (1 << 20));

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
    auto num_elements = tensor_size / ElementSize;
    assert(num_elements <= max_num_elements_);
    std::vector<ElementType> src_vec(src_vec_.begin(), src_vec_.begin() + num_elements);

    distributed::MeshCoordinateRange coord_range(mesh_device_->shape());
    std::vector<MeshCommandQueue::ShardDataTransfer> input_shards = {};
    for (auto& coord : coord_range) {
        input_shards.push_back({coord, src_vec.data()});
    }

    std::unordered_map<distributed::MeshCoordinate, std::vector<ElementType>> dst_vec = {};
    std::vector<MeshCommandQueue::ShardDataTransfer> output_shards = {};
    for (auto& coord : coord_range) {
        dst_vec[coord] = std::vector<ElementType>(num_elements, 0);
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
        ::testing::Values(
            (2ull << 30) / ElementSize, (4ull << 30) / ElementSize, (8ull << 30) / ElementSize),  // tensor sizes
        ::testing::Values(1024, 2048, 16 << 10, 1 << 20)                                          // page sizes
        ));

class ShardedMeshBufferTestSuite : public LargeMeshBufferTestSuiteBase,
                                   public testing::WithParamInterface<std::tuple<Shape2D, uint32_t>> {};

TEST_P(ShardedMeshBufferTestSuite, DRAMReadback) {
    auto [shard_shape, page_size] = GetParam();
    constexpr auto tensor_layout = TensorMemoryLayout::HEIGHT_SHARDED;

    // Ensure buffer dimensions are divisible by tile dimensions
    assert(shard_shape.height() % constants::TILE_HEIGHT == 0);
    assert(shard_shape.width() % constants::TILE_WIDTH == 0);

    CoreCoord core_grid_size = mesh_device_->compute_with_storage_grid_size();
    CoreCoord start(0, 0);
    CoreCoord end(core_grid_size.x - 1, core_grid_size.y - 1);
    CoreRange cores(start, end);
    // CoreCoord core_grid_size = cores.grid_size();

    // Map allocation for input tensor to on device core grid: per core dimensions H x W in elements
    Shape2D shard_core_shape{
        div_up(shard_shape.height(), core_grid_size.y), div_up(shard_shape.width(), core_grid_size.x)};

    // Determine page shape from page size and tile shape, in elements
    uint32_t page_height = page_size / constants::TILE_WIDTH / ElementSize;
    uint32_t page_width = constants::TILE_WIDTH;
    TT_ASSERT(
        page_height * page_width * ElementSize == page_size,
        "page_size:{}, page_height:{}, page_width:{}",
        page_size,
        page_height,
        page_width);
    Shape2D page_shape{page_height, page_width};

    // Device shard shape in pages
    Shape2D shard_device_shape{div_up(shard_shape.height(), page_height), div_up(shard_shape.width(), page_width)};

    auto shard_orientation = ShardOrientation::ROW_MAJOR;
    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = page_size,
        .buffer_type = BufferType::DRAM,
        .sharding_args = BufferShardingArgs(
            ShardSpecBuffer{CoreRangeSet(cores), shard_core_shape, shard_orientation, page_shape, shard_device_shape},
            tensor_layout),
        .bottom_up = true};

    // shard size in bytes
    uint64_t shard_size = shard_shape.height() * shard_shape.width() * ElementSize;
    log_info(
        tt::LogTest,
        "Core grid size:{}. on device shard size:{} MB, shape:{} pages, {} elements, shape on core:{} elements, page "
        "shape:{} elements, page_size:{} KB",
        core_grid_size,
        shard_size >> 20,
        shard_device_shape,
        shard_shape,
        shard_core_shape,
        page_shape,
        page_size / (1 << 10));

    uint32_t device_rows = mesh_device_->num_rows();
    uint32_t device_cols = mesh_device_->num_cols();
    uint32_t num_devices = device_rows * device_cols;
    // tensor (buffer loaded across devices) shape in elements
    Shape2D tensor_shape = {shard_shape.height() * device_rows, shard_shape.width() * device_cols};
    // tensor size in bytes
    uint64_t tensor_size = num_devices * shard_size;
    ShardedBufferConfig sharded_config{
        .global_size = tensor_size,
        .global_buffer_shape = tensor_shape,
        .shard_shape = shard_shape,
        .shard_orientation = shard_orientation,
    };
    log_info(
        tt::LogTest,
        "Mesh buffer (tensor) shape:{} elements, tensor size:{} MB",
        tensor_shape,
        tensor_size / (1 << 20));

    assert(sharded_config.compute_datum_size_bytes() == ElementSize);
    auto mesh_buffer = MeshBuffer::create(sharded_config, per_device_buffer_config, mesh_device_.get());

    distributed::MeshCoordinateRange coord_range(mesh_device_->shape());
    auto num_elements = tensor_size / num_devices / ElementSize;
    assert(num_elements <= max_num_elements_);
    std::vector<ElementType> src_vec(src_vec_.begin(), src_vec_.begin() + num_elements);
    std::vector<MeshCommandQueue::ShardDataTransfer> input_shards = {};
    for (auto& coord : coord_range) {
        input_shards.push_back({coord, src_vec.data()});
    }

    std::unordered_map<distributed::MeshCoordinate, std::vector<ElementType>> dst_vec = {};
    std::vector<MeshCommandQueue::ShardDataTransfer> output_shards = {};
    for (auto& coord : coord_range) {
        dst_vec[coord] = std::vector<ElementType>(tensor_size / num_devices / ElementSize, 0);
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
            Shape2D((1 << 14), (1 << 14)),                // 2 GB with uint64_t
            Shape2D((1 << 14), (1 << 15)),                // 4 GB with uint64_t
            Shape2D((1 << 15), (1 << 15))),               // 8 GB with uint64_t
        ::testing::Values(1024, 4096, 16 << 10, 1 << 20)  // page size
        ));

}  // namespace
}  // namespace tt::tt_metal::distributed::test
