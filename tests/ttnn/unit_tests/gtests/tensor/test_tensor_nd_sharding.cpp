// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include <tt_stl/reflection.hpp>
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn_test_fixtures.hpp"
#include <tt-metalium/distributed.hpp>

namespace {
struct NDShardingParams {
    Shape shape;
    Shape shard_shape;
    Layout layout = Layout::TILE;
};
struct NDShardingOpCompatParams {
    Shape shape;
    Shape shard_shape;
    CoreCoord grid_size;
};
struct NDShardingBufferSizeParams {
    Shape shape;
    Shape shard_shape;
    CoreCoord grid_size;
    size_t expected_buffer_size = 0;
    size_t expected_num_pages = 0;
    size_t expected_num_dev_pages = 0;
    size_t expected_aligned_size_per_bank = 0;
};
TensorSpec get_nd_sharding_tensor_spec(
    const NDShardingParams& params,
    BufferType buffer_type,
    ShardOrientation orientation,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    CoreRangeSet cores;
    if (buffer_type == BufferType::L1) {
        cores = CoreRangeSet(CoreRange(CoreCoord{0, 0}, CoreCoord{6, 6}));
    } else {
        auto dram_grid_size = mesh_device->dram_grid_size();
        cores = CoreRangeSet(CoreRange(CoreCoord{0, 0}, CoreCoord{dram_grid_size.x - 1, dram_grid_size.y - 1}));
    }
    MemoryConfig memory_config{buffer_type, NdShardSpec{params.shard_shape, cores, orientation}};
    TensorLayout tensor_layout(DataType::UINT16, PageConfig(params.layout), memory_config);
    return TensorSpec(params.shape, tensor_layout);
}
}  // namespace

class NDShardingTests
    : public ttnn::TTNNFixtureWithSuiteDevice<NDShardingTests>,
      public ::testing::WithParamInterface<std::tuple<NDShardingParams, BufferType, ShardOrientation>> {};

TEST_P(NDShardingTests, LoopbackTest) {
    const auto& [params, buffer_type, orientation] = GetParam();
    auto tensor_spec = get_nd_sharding_tensor_spec(params, buffer_type, orientation, device_holder_);

    size_t volume = params.shape.volume();
    std::vector<uint16_t> data(volume);
    for (size_t i = 0; i < volume; i++) {
        data[i] = static_cast<uint16_t>(i);
    }

    auto tensor = Tensor::from_vector(data, tensor_spec, device_);
    EXPECT_TRUE(tensor.buffer()->buffer_distribution_spec().has_value());
    auto readback_data = tensor.to_vector<uint16_t>();

    for (size_t i = 0; i < volume; i++) {
        EXPECT_EQ(data[i], readback_data[i]);
    }
}

TEST_P(NDShardingTests, RegionWriteReadTest) {
    const auto& [params, buffer_type, orientation] = GetParam();
    auto tensor_spec = get_nd_sharding_tensor_spec(params, buffer_type, orientation, device_holder_);

    size_t volume = params.shape.volume();
    std::vector<uint16_t> data(volume);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = static_cast<uint16_t>(i);
    }
    auto data_tensor = Tensor::from_vector(data, tensor_spec);
    auto tensor_data_span = host_buffer::get_as<uint16_t>(data_tensor);
    auto tensor_data = std::vector<uint16_t>(tensor_data_span.begin(), tensor_data_span.end());

    std::vector<uint16_t> empty_data(volume);
    auto tensor = Tensor::from_vector(empty_data, tensor_spec, device_);

    const auto& buffer = tensor.mesh_buffer();

    // TODO(#38691): Clean this up after we provide non-shared-ptr overloads for these methods.
    const auto& shared_mesh_buffer = tensor.device_storage().get_mesh_buffer_leak_ownership();

    size_t region_size = buffer.page_size();
    while (buffer.size() % (region_size * 2) == 0) {
        region_size *= 2;
    }

    std::vector<uint16_t> partial_readback_data(tensor_data.size());
    std::vector<uint16_t> full_readback_data(tensor_data.size());

    for (size_t region = 0; region < buffer.size() / region_size; region++) {
        size_t region_offset = region * region_size;
        auto buffer_region = BufferRegion{region_offset, region_size};
        auto write_shard_data_transfer =
            distributed::ShardDataTransfer{distributed::MeshCoordinate(0, 0)}
                .host_data(reinterpret_cast<std::byte*>(tensor_data.data()) + region_offset)
                .region(buffer_region);
        auto read_shard_data_transfer =
            distributed::ShardDataTransfer{distributed::MeshCoordinate(0, 0)}
                .host_data(reinterpret_cast<std::byte*>(partial_readback_data.data()) + region_offset)
                .region(buffer_region);
        device_->mesh_command_queue().enqueue_write_shards(shared_mesh_buffer, {write_shard_data_transfer}, true);
        device_->mesh_command_queue().enqueue_read_shards({read_shard_data_transfer}, shared_mesh_buffer, true);
    }
    EXPECT_EQ(tensor_data, partial_readback_data);

    distributed::ReadShard(
        device_->mesh_command_queue(), full_readback_data, shared_mesh_buffer, distributed::MeshCoordinate(0, 0), true);
    EXPECT_EQ(tensor_data, full_readback_data);
}

class BufferDistributionSpecCreationTests
    : public ttnn::TTNNFixtureWithSuiteDevice<BufferDistributionSpecCreationTests> {};

TEST_F(BufferDistributionSpecCreationTests, LegacyAndNdShardSpecCreateBufferDistributionSpec) {
    const Shape shape({3, 64, 64});
    const CoreRangeSet cores(CoreRange(CoreCoord{0, 0}, CoreCoord{1, 5}));
    const std::vector<uint16_t> data(shape.volume(), 1);

    {
        MemoryConfig memory_config{
            TensorMemoryLayout::BLOCK_SHARDED,
            BufferType::L1,
            ShardSpec{cores, Shape2D{32, 32}, ShardOrientation::ROW_MAJOR}};
        TensorLayout tensor_layout(DataType::UINT16, PageConfig(Layout::TILE), memory_config);
        TensorSpec tensor_spec(shape, tensor_layout);

        auto tensor = Tensor::from_vector(data, tensor_spec, device_);
        EXPECT_TRUE(tensor.buffer()->buffer_distribution_spec().has_value());
    }

    {
        MemoryConfig memory_config{BufferType::L1, NdShardSpec{Shape({2, 32, 32}), cores, ShardOrientation::ROW_MAJOR}};
        TensorLayout tensor_layout(DataType::UINT16, PageConfig(Layout::TILE), memory_config);
        TensorSpec tensor_spec(shape, tensor_layout);

        auto tensor = Tensor::from_vector(data, tensor_spec, device_);
        EXPECT_TRUE(tensor.buffer()->buffer_distribution_spec().has_value());
    }
}

class NdShardingOpCompatTests : public ttnn::TTNNFixtureWithSuiteDevice<NdShardingOpCompatTests>,
                                public ::testing::WithParamInterface<NDShardingOpCompatParams> {};

TEST_P(NdShardingOpCompatTests, TestAdd) {
    const auto& params = GetParam();

    CoreRangeSet cores(CoreRange(CoreCoord{0, 0}, CoreCoord{params.grid_size.x - 1, params.grid_size.y - 1}));
    NdShardSpec nd_shard_spec{params.shard_shape, cores, ShardOrientation::ROW_MAJOR};
    MemoryConfig memory_config{BufferType::L1, nd_shard_spec};
    // NOTE: currently binary op does not support interger data types with uneven shard size, so we use float32
    TensorLayout tensor_layout(DataType::FLOAT32, PageConfig(Layout::TILE), memory_config);
    TensorSpec tensor_spec(params.shape, tensor_layout);

    size_t volume = params.shape.volume();
    std::vector<float> data(volume);
    for (size_t i = 0; i < volume; i++) {
        data[i] = static_cast<float>(i);
    }
    auto tensor_a = Tensor::from_vector(data, tensor_spec, device_);
    for (auto& elem : data) {
        elem *= 2;
    }
    auto tensor_b = Tensor::from_vector(data, tensor_spec, device_);

    auto sum_tensor = ttnn::add(tensor_a, tensor_b);

    auto sum_vector = sum_tensor.to_vector<float>();
    for (size_t i = 0; i < volume; i++) {
        EXPECT_FLOAT_EQ(sum_vector[i], static_cast<float>(i * 3));
    }
}

class NDShardingPerfTests : public ttnn::TTNNFixtureWithDevice {};

TEST_F(NDShardingPerfTests, TestBatchShardingPerf) {
    CoreRangeSet cores(CoreRange(CoreCoord{0, 0}, CoreCoord{6, 6}));

    Shape tensor_shape{16, 1024, 1024};
    Shape shard_shape_nd_batch{16, 160, 160};
    Shape shard_shape_nd_small{1, 64, 64};
    Shape2D shard_shape_2d{2368, 160};

    size_t volume = tensor_shape.volume();
    std::vector<uint16_t> data(volume);
    for (size_t i = 0; i < volume; i++) {
        data[i] = static_cast<uint16_t>(i);
    }

    auto measure_to_device_time_ns = [&](const TensorSpec& tensor_spec) -> double {
        auto tensor = Tensor::from_vector(data, tensor_spec);

        auto start = std::chrono::high_resolution_clock::now();
        auto device_tensor = tensor.to_device(device_, tensor_spec.memory_config());
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        return duration.count();
    };

    double batch_nd_sharding_time_ns = [&]() {
        MemoryConfig memory_config{BufferType::L1, NdShardSpec{shard_shape_nd_batch, cores}};
        TensorLayout tensor_layout(DataType::UINT16, PageConfig(Layout::TILE), memory_config);
        TensorSpec tensor_spec(tensor_shape, tensor_layout);
        return measure_to_device_time_ns(tensor_spec);
    }();

    double small_shards_nd_sharding_time_ns = [&]() {
        MemoryConfig memory_config{BufferType::L1, NdShardSpec{shard_shape_nd_small, cores}};
        TensorLayout tensor_layout(DataType::UINT16, PageConfig(Layout::TILE), memory_config);
        TensorSpec tensor_spec(tensor_shape, tensor_layout);
        return measure_to_device_time_ns(tensor_spec);
    }();

    double block_2d_sharding_time_ns = [&]() {
        MemoryConfig memory_config{TensorMemoryLayout::BLOCK_SHARDED, BufferType::L1, ShardSpec{cores, shard_shape_2d}};
        TensorLayout tensor_layout(DataType::UINT16, PageConfig(Layout::TILE), memory_config);
        TensorSpec tensor_spec(tensor_shape, tensor_layout);
        return measure_to_device_time_ns(tensor_spec);
    }();

    log_info(tt::LogTest, "Batch ND sharding time: {} ns", batch_nd_sharding_time_ns);
    log_info(tt::LogTest, "Small shards ND sharding time: {} ns", small_shards_nd_sharding_time_ns);
    log_info(tt::LogTest, "Block 2D sharding time: {} ns", block_2d_sharding_time_ns);

    EXPECT_TRUE(batch_nd_sharding_time_ns < block_2d_sharding_time_ns * 6);
    EXPECT_TRUE(small_shards_nd_sharding_time_ns < block_2d_sharding_time_ns * 6);
}

class NDShardingBufferSizeTests : public ttnn::TTNNFixtureWithSuiteDevice<NDShardingBufferSizeTests>,
                                  public ::testing::WithParamInterface<NDShardingBufferSizeParams> {};

TEST_P(NDShardingBufferSizeTests, TestBufferSize) {
    const auto& params = GetParam();

    CoreRangeSet cores(CoreRange(CoreCoord{0, 0}, CoreCoord{params.grid_size.x - 1, params.grid_size.y - 1}));
    NdShardSpec nd_shard_spec{params.shard_shape, cores, ShardOrientation::ROW_MAJOR};
    MemoryConfig memory_config{BufferType::L1, nd_shard_spec};
    TensorLayout tensor_layout(DataType::UINT8, PageConfig(Layout::TILE), memory_config);
    TensorSpec tensor_spec(params.shape, tensor_layout);

    size_t volume = params.shape.volume();
    std::vector<uint8_t> data(volume);
    auto tensor = Tensor::from_vector(data, tensor_spec, device_);

    auto* buffer = tensor.buffer();
    EXPECT_EQ(buffer->size(), params.expected_buffer_size);
    EXPECT_EQ(buffer->num_pages(), params.expected_num_pages);
    EXPECT_EQ(buffer->num_dev_pages(), params.expected_num_dev_pages);
    EXPECT_EQ(buffer->aligned_size_per_bank(), params.expected_aligned_size_per_bank);
}

INSTANTIATE_TEST_SUITE_P(
    TensorShardingTests,
    NDShardingTests,
    ::testing::Combine(
        ::testing::Values(
            NDShardingParams{
                .shape = Shape({320, 320}),
                .shard_shape = Shape({32, 32}),
                .layout = Layout::TILE,
            },
            NDShardingParams{
                .shape = Shape({32, 32, 32}),
                .shard_shape = Shape({32, 32, 32}),
                .layout = Layout::TILE,
            },
            NDShardingParams{
                .shape = Shape({3 * 32, 4 * 32, 5 * 32}),
                .shard_shape = Shape({32, 4 * 32, 5 * 32}),
                .layout = Layout::TILE,
            },
            NDShardingParams{
                .shape = Shape({3 * 32, 4 * 32, 5 * 32}),
                .shard_shape = Shape({3 * 32, 32, 5 * 32}),
                .layout = Layout::TILE,
            },
            NDShardingParams{
                .shape = Shape({3 * 32, 4 * 32, 5 * 32}),
                .shard_shape = Shape({3 * 32, 4 * 32, 32}),
                .layout = Layout::TILE,
            },
            NDShardingParams{
                .shape = Shape({3 * 32, 4 * 32, 5 * 32}),
                .shard_shape = Shape({3 * 32, 32, 32}),
                .layout = Layout::TILE,
            },
            NDShardingParams{
                .shape = Shape({3 * 32, 4 * 32, 5 * 32}),
                .shard_shape = Shape({32, 4 * 32, 32}),
                .layout = Layout::TILE,
            },
            NDShardingParams{
                .shape = Shape({3 * 32, 4 * 32, 5 * 32}),
                .shard_shape = Shape({32, 32, 5 * 32}),
                .layout = Layout::TILE,
            },
            NDShardingParams{
                .shape = Shape({3 * 32, 4 * 32, 5 * 32}),
                .shard_shape = Shape({32, 32, 32}),
                .layout = Layout::TILE,
            },
            NDShardingParams{
                .shape = Shape({3 * 32 + 5, 4 * 32, 5 * 32}),
                .shard_shape = Shape({32, 4 * 32, 5 * 32}),
                .layout = Layout::TILE,
            },
            NDShardingParams{
                .shape = Shape({3 * 32, 4 * 32 + 5, 5 * 32}),
                .shard_shape = Shape({3 * 32, 32, 5 * 32}),
                .layout = Layout::TILE,
            },
            NDShardingParams{
                .shape = Shape({3 * 32, 4 * 32, 5 * 32 + 5}),
                .shard_shape = Shape({3 * 32, 4 * 32, 32}),
                .layout = Layout::TILE,
            },
            NDShardingParams{
                .shape = Shape({3 * 32, 4 * 32 + 5, 5 * 32 + 5}),
                .shard_shape = Shape({3 * 32, 32, 32}),
                .layout = Layout::TILE,
            },
            NDShardingParams{
                .shape = Shape({3 * 32 + 5, 4 * 32, 5 * 32 + 5}),
                .shard_shape = Shape({32, 4 * 32, 32}),
                .layout = Layout::TILE,
            },
            NDShardingParams{
                .shape = Shape({3 * 32 + 5, 4 * 32 + 5, 5 * 32}),
                .shard_shape = Shape({32, 32, 5 * 32}),
                .layout = Layout::TILE,
            },
            NDShardingParams{
                .shape = Shape({3 * 32 + 5, 4 * 32 + 5, 5 * 32 + 5}),
                .shard_shape = Shape({32, 32, 32}),
                .layout = Layout::TILE,
            },
            NDShardingParams{
                .shape = Shape({30, 40, 50}),
                .shard_shape = Shape({30, 40, 50}),
                .layout = Layout::ROW_MAJOR,
            },
            NDShardingParams{
                .shape = Shape({30, 40, 50}),
                .shard_shape = Shape({10, 40, 50}),
                .layout = Layout::ROW_MAJOR,
            },
            NDShardingParams{
                .shape = Shape({30, 40, 50}),
                .shard_shape = Shape({30, 10, 50}),
                .layout = Layout::ROW_MAJOR,
            },
            NDShardingParams{
                .shape = Shape({30, 40, 50}),
                .shard_shape = Shape({30, 40, 10}),
                .layout = Layout::ROW_MAJOR,
            },
            NDShardingParams{
                .shape = Shape({30, 40, 50}),
                .shard_shape = Shape({10, 10, 50}),
                .layout = Layout::ROW_MAJOR,
            },
            NDShardingParams{
                .shape = Shape({30, 40, 50}),
                .shard_shape = Shape({10, 40, 10}),
                .layout = Layout::ROW_MAJOR,
            },
            NDShardingParams{
                .shape = Shape({30, 40, 50}),
                .shard_shape = Shape({30, 10, 10}),
                .layout = Layout::ROW_MAJOR,
            },
            NDShardingParams{
                .shape = Shape({30, 40, 50}),
                .shard_shape = Shape({10, 10, 10}),
                .layout = Layout::ROW_MAJOR,
            },
            NDShardingParams{
                .shape = Shape({3, 4, 5}),
                .shard_shape = Shape({1, 1, 1}),
                .layout = Layout::ROW_MAJOR,
            },
            NDShardingParams{
                .shape = Shape({35, 40, 50}),
                .shard_shape = Shape({10, 40, 50}),
                .layout = Layout::ROW_MAJOR,
            },
            NDShardingParams{
                .shape = Shape({30, 45, 50}),
                .shard_shape = Shape({30, 10, 50}),
                .layout = Layout::ROW_MAJOR,
            },
            NDShardingParams{
                .shape = Shape({30, 40, 55}),
                .shard_shape = Shape({30, 40, 10}),
                .layout = Layout::ROW_MAJOR,
            },
            NDShardingParams{
                .shape = Shape({35, 45, 50}),
                .shard_shape = Shape({10, 10, 50}),
                .layout = Layout::ROW_MAJOR,
            },
            NDShardingParams{
                .shape = Shape({35, 40, 55}),
                .shard_shape = Shape({10, 40, 10}),
                .layout = Layout::ROW_MAJOR,
            },
            NDShardingParams{
                .shape = Shape({30, 45, 55}),
                .shard_shape = Shape({30, 10, 10}),
                .layout = Layout::ROW_MAJOR,
            },
            NDShardingParams{
                .shape = Shape({35, 45, 55}),
                .shard_shape = Shape({10, 10, 10}),
                .layout = Layout::ROW_MAJOR,
            },
            NDShardingParams{
                .shape = Shape({3, 5, 7}),
                .shard_shape = Shape({2, 2, 2}),
                .layout = Layout::ROW_MAJOR,
            }),
        ::testing::Values(BufferType::L1, BufferType::DRAM),
        ::testing::Values(ShardOrientation::ROW_MAJOR, ShardOrientation::COL_MAJOR)));
INSTANTIATE_TEST_SUITE_P(
    TensorShardingTests,
    NdShardingOpCompatTests,
    ::testing::Values(
        NDShardingOpCompatParams{
            .shape = Shape({32 * 5, 32 * 5}),
            .shard_shape = Shape({32, 32}),
            .grid_size = CoreCoord{5, 5},
        },
        NDShardingOpCompatParams{
            .shape = Shape({32 * 5, 32 * 5}),
            .shard_shape = Shape({32 * 2, 32 * 2}),
            .grid_size = CoreCoord{3, 3},
        },
        NDShardingOpCompatParams{
            .shape = Shape({1, 32 * 5, 32 * 5}),
            .shard_shape = Shape({1, 32 * 2, 32 * 2}),
            .grid_size = CoreCoord{3, 3},
        },
        NDShardingOpCompatParams{
            .shape = Shape({1, 1, 32 * 5, 32 * 5}),
            .shard_shape = Shape({1, 1, 32 * 2, 32 * 2}),
            .grid_size = CoreCoord{3, 3},
        },
        NDShardingOpCompatParams{
            .shape = Shape({2, 32 * 4, 32 * 5}),
            .shard_shape = Shape({1, 32 * 2, 32 * 2}),
            .grid_size = CoreCoord{3, 4},
        },
        NDShardingOpCompatParams{
            .shape = Shape({1, 2, 32 * 4, 32 * 5}),
            .shard_shape = Shape({1, 1, 32 * 2, 32 * 2}),
            .grid_size = CoreCoord{3, 4},
        }));

INSTANTIATE_TEST_SUITE_P(
    TensorShardingTests,
    NDShardingBufferSizeTests,
    ::testing::Values(
        NDShardingBufferSizeParams{
            .shape = Shape({4, 4, 32 * 2, 32 * 2}),
            .shard_shape = Shape({1, 1, 32, 32}),
            .grid_size = CoreCoord{4, 4},
            .expected_buffer_size = 64 * 32 * 32,
            .expected_num_pages = 64,
            .expected_num_dev_pages = 64,
            .expected_aligned_size_per_bank = 4 * 32 * 32,
        },
        NDShardingBufferSizeParams{
            .shape = Shape({4, 7, 32 * 2, 32 * 2}),
            .shard_shape = Shape({2, 2, 32, 32}),
            .grid_size = CoreCoord{1, 3},
            .expected_buffer_size = 112 * 32 * 32,
            .expected_num_pages = 112,
            .expected_num_dev_pages = 3 * 11 * 4,
            .expected_aligned_size_per_bank = 11 * 4 * 32 * 32,
        }));
