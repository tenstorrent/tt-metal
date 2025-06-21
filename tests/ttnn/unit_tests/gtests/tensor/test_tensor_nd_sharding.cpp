// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn_test_fixtures.hpp"

namespace {
struct NDShardingParams {
    Shape shape;
    Shape shard_shape;
    Layout layout = Layout::TILE;
};
struct LegacyToNdShardingParams {
    Shape shape;
    TensorMemoryLayout memory_layout = TensorMemoryLayout::BLOCK_SHARDED;
    std::optional<Shape2D> shard_shape_2d;
    Layout layout = Layout::TILE;

    std::optional<Shape> shard_shape_nd;
    std::optional<CoreCoord> grid_size;
};
struct NdToLegacyShardingParams {
    Shape shape;
    Shape shard_shape_nd;
    Layout layout = Layout::TILE;
    CoreCoord grid_size;

    TensorMemoryLayout memory_layout = TensorMemoryLayout::BLOCK_SHARDED;
    std::optional<Shape2D> shard_shape_2d;
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
    const NDShardingParams& params, BufferType buffer_type, ShardOrientation orientation, IDevice* device) {
    CoreRangeSet cores;
    if (buffer_type == BufferType::L1) {
        cores = CoreRangeSet(CoreRange(CoreCoord{0, 0}, CoreCoord{6, 6}));
    } else {
        auto dram_grid_size = device->dram_grid_size();
        cores = CoreRangeSet(CoreRange(CoreCoord{0, 0}, CoreCoord{dram_grid_size.x - 1, dram_grid_size.y - 1}));
    }
    MemoryConfig memory_config{buffer_type, NdShardSpec{params.shard_shape, cores, orientation}};
    TensorLayout tensor_layout(DataType::UINT16, PageConfig(params.layout), memory_config);
    return TensorSpec(params.shape, tensor_layout);
}
}  // namespace

class NDShardingTests
    : public ttnn::TTNNFixtureWithDevice,
      public ::testing::WithParamInterface<std::tuple<NDShardingParams, BufferType, ShardOrientation>> {};

TEST_P(NDShardingTests, LoopbackTest) {
    const auto& [params, buffer_type, orientation] = GetParam();
    auto tensor_spec = get_nd_sharding_tensor_spec(params, buffer_type, orientation, device_);

    size_t volume = params.shape.volume();
    std::vector<uint16_t> data(volume);
    for (size_t i = 0; i < volume; i++) {
        data[i] = static_cast<uint16_t>(i);
    }

    auto tensor = Tensor::from_vector(data, tensor_spec, device_);
    auto readback_data = tensor.to_vector<uint16_t>();

    for (size_t i = 0; i < volume; i++) {
        ASSERT_EQ(data[i], readback_data[i]);
    }
}

TEST_P(NDShardingTests, RegionWriteReadTest) {
    const auto& [params, buffer_type, orientation] = GetParam();
    auto tensor_spec = get_nd_sharding_tensor_spec(params, buffer_type, orientation, device_);

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

    auto& storage = std::get<DeviceStorage>(tensor.storage());
    auto buffer = storage.get_buffer();
    auto page_size = buffer->page_size();
    auto device = buffer->device();

    size_t region_size = buffer->page_size();
    while (buffer->size() % (region_size * 2) == 0) {
        region_size *= 2;
    }

    std::vector<uint16_t> partial_readback_data(tensor_data.size());
    std::vector<uint16_t> full_readback_data(tensor_data.size());

    for (size_t region = 0; region < buffer->size() / region_size; region++) {
        size_t region_offset = region * region_size;
        auto buffer_view = buffer->view(BufferRegion{region_offset, region_size});
        EnqueueWriteBuffer(
            device->command_queue(),
            buffer_view,
            reinterpret_cast<const std::byte*>(tensor_data.data()) + region_offset,
            true);
        EnqueueReadBuffer(
            device->command_queue(),
            buffer_view,
            reinterpret_cast<std::byte*>(partial_readback_data.data()) + region_offset,
            true);
    }
    EXPECT_EQ(tensor_data, partial_readback_data);

    EnqueueReadBuffer(device->command_queue(), *buffer, full_readback_data.data(), true);
    EXPECT_EQ(tensor_data, full_readback_data);
}

class LegacyToNdShardingTests : public ::testing::TestWithParam<LegacyToNdShardingParams> {};

TEST_P(LegacyToNdShardingTests, LegacyToNdSharding) {
    const auto& params = GetParam();

    CoreRangeSet cores(CoreRange(CoreCoord{0, 0}, CoreCoord{6, 6}));
    std::optional<ShardSpec> shard_spec;
    if (params.shard_shape_2d.has_value()) {
        shard_spec = ShardSpec{cores, params.shard_shape_2d.value(), ShardOrientation::ROW_MAJOR};
    }
    MemoryConfig memory_config{params.memory_layout, BufferType::L1, shard_spec};
    TensorLayout tensor_layout(DataType::UINT16, PageConfig(params.layout), memory_config);
    TensorSpec tensor_spec(params.shape, tensor_layout);

    auto nd_shard_spec = tensor_spec.memory_config().nd_shard_spec();
    ASSERT_EQ(nd_shard_spec.has_value(), params.shard_shape_nd.has_value());
    if (nd_shard_spec.has_value()) {
        ASSERT_EQ(nd_shard_spec->shard_shape, params.shard_shape_nd.value());
        if (params.grid_size.has_value()) {
            ASSERT_EQ(nd_shard_spec->grid.ranges().size(), 1);
            ASSERT_EQ(nd_shard_spec->grid.ranges()[0].grid_size(), params.grid_size.value());
        } else {
            ASSERT_EQ(nd_shard_spec->grid, cores);
        }
    }
}

class NdToLegacyShardingTests : public ::testing::TestWithParam<NdToLegacyShardingParams> {};

TEST_P(NdToLegacyShardingTests, NdToLegacySharding) {
    const auto& params = GetParam();

    CoreRangeSet cores(CoreRange(CoreCoord{0, 0}, CoreCoord{params.grid_size.x - 1, params.grid_size.y - 1}));
    NdShardSpec nd_shard_spec{params.shard_shape_nd, cores, ShardOrientation::ROW_MAJOR};
    MemoryConfig memory_config{BufferType::L1, nd_shard_spec};
    TensorLayout tensor_layout(DataType::UINT16, PageConfig(params.layout), memory_config);
    TensorSpec tensor_spec(params.shape, tensor_layout);

    ASSERT_EQ(tensor_spec.memory_config().memory_layout(), params.memory_layout);

    auto shard_spec_2d = tensor_spec.memory_config().shard_spec();
    ASSERT_EQ(shard_spec_2d.has_value(), params.shard_shape_2d.has_value());
    if (shard_spec_2d.has_value()) {
        ASSERT_EQ(shard_spec_2d->shape, params.shard_shape_2d.value());
    }
}

class NdShardingOpCompatTests : public ttnn::TTNNFixtureWithDevice,
                                public ::testing::WithParamInterface<NDShardingOpCompatParams> {};

TEST_P(NdShardingOpCompatTests, TestAdd) {
    const auto& params = GetParam();

    CoreRangeSet cores(CoreRange(CoreCoord{0, 0}, CoreCoord{params.grid_size.x - 1, params.grid_size.y - 1}));
    NdShardSpec nd_shard_spec{params.shard_shape, cores, ShardOrientation::ROW_MAJOR};
    MemoryConfig memory_config{BufferType::L1, nd_shard_spec};
    TensorLayout tensor_layout(DataType::UINT32, PageConfig(Layout::TILE), memory_config);
    TensorSpec tensor_spec(params.shape, tensor_layout);

    size_t volume = params.shape.volume();
    std::vector<uint32_t> data(volume);
    for (size_t i = 0; i < volume; i++) {
        data[i] = static_cast<uint32_t>(i);
    }

    auto tensor_a = Tensor::from_vector(data, tensor_spec, device_);
    for (auto& elem : data) {
        elem *= 2;
    }
    auto tensor_b = Tensor::from_vector(data, tensor_spec, device_);

    auto sum_tensor = ttnn::add(tensor_a, tensor_b);

    auto sum_vector = sum_tensor.to_vector<uint32_t>();
    for (size_t i = 0; i < volume; i++) {
        ASSERT_EQ(sum_vector[i], i * 3);
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

class NDShardingBufferSizeTests : public ttnn::TTNNFixtureWithDevice,
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

    auto buffer = tensor.buffer();
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
    LegacyToNdShardingTests,
    ::testing::Values(
        LegacyToNdShardingParams{
            .shape = Shape({2, 32 * 2, 32 * 2}),
            .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
            .shard_shape_2d = Shape2D{32, 32},
            .layout = Layout::TILE,
            .shard_shape_nd = Shape({1, 32, 32}),
            .grid_size = CoreCoord{2, 4},
        },
        LegacyToNdShardingParams{
            .shape = Shape({2, 32 * 2, 32 * 2}),
            .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
            .shard_shape_2d = Shape2D{32 * 2, 32},
            .layout = Layout::TILE,
            .shard_shape_nd = Shape({1, 32 * 2, 32}),
            .grid_size = CoreCoord{2, 2},
        },
        LegacyToNdShardingParams{
            .shape = Shape({2, 32 * 2, 32 * 2}),
            .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
            .shard_shape_2d = Shape2D{32 * 3, 32},
            .layout = Layout::TILE,
            .shard_shape_nd = std::nullopt,  // Can't convert, because sharding across higher dimensions
        },
        LegacyToNdShardingParams{
            .shape = Shape({2, 4, 4}),
            .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
            .shard_shape_2d = Shape2D{2, 2},
            .layout = Layout::ROW_MAJOR,
            .shard_shape_nd = Shape({1, 2, 2}),
            .grid_size = CoreCoord{2, 4},
        },
        LegacyToNdShardingParams{
            .shape = Shape({2, 4}),
            .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
            .shard_shape_2d = Shape2D{2, 2},
            .layout = Layout::ROW_MAJOR,
            .shard_shape_nd = Shape({2, 2}),
            .grid_size = CoreCoord{2, 1},
        },
        LegacyToNdShardingParams{
            .shape = Shape({4}),
            .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
            .shard_shape_2d = Shape2D{1, 2},
            .layout = Layout::ROW_MAJOR,
            .shard_shape_nd = Shape({2}),
            .grid_size = CoreCoord{2, 1},
        },
        LegacyToNdShardingParams{
            .shape = Shape({}),
            .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
            .shard_shape_2d = Shape2D{1, 1},
            .layout = Layout::ROW_MAJOR,
            .shard_shape_nd = Shape({1}),
            .grid_size = CoreCoord{1, 1},
        },
        LegacyToNdShardingParams{
            .shape = Shape({2, 32 * 2, 32 * 2}),
            .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
            .shard_shape_2d = Shape2D{32, 32 * 2},
            .layout = Layout::TILE,
            .shard_shape_nd = Shape({1, 32, 32 * 2}),
        },
        LegacyToNdShardingParams{
            .shape = Shape({2, 32 * 2, 32 * 2}),
            .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
            .shard_shape_2d = Shape2D{32 * 3, 32 * 2},
            .layout = Layout::TILE,
            .shard_shape_nd = std::nullopt,  // Can't convert, because sharding across higher dimensions
        },
        LegacyToNdShardingParams{
            .shape = Shape({2, 3, 4}),
            .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
            .shard_shape_2d = Shape2D{3, 4},
            .layout = Layout::ROW_MAJOR,
            .shard_shape_nd = Shape({1, 3, 4}),
        },
        LegacyToNdShardingParams{
            .shape = Shape({2, 3, 4}),
            .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
            .shard_shape_2d = Shape2D{4, 4},
            .layout = Layout::ROW_MAJOR,
            .shard_shape_nd = std::nullopt,  // Can't convert, because sharding across higher dimensions
        },
        LegacyToNdShardingParams{
            .shape = Shape({2, 32 * 2, 32 * 2}),
            .memory_layout = TensorMemoryLayout::WIDTH_SHARDED,
            .shard_shape_2d = Shape2D{32 * 4, 32},
            .layout = Layout::TILE,
            .shard_shape_nd = Shape({2, 32 * 2, 32}),
        },
        LegacyToNdShardingParams{
            .shape = Shape({2, 32 * 2, 32 * 2}),
            .memory_layout = TensorMemoryLayout::WIDTH_SHARDED,
            .shard_shape_2d = Shape2D{32 * 4, 32 * 3},
            .layout = Layout::TILE,
            .shard_shape_nd = Shape({2, 32 * 2, 32 * 3}),
        },
        LegacyToNdShardingParams{
            .shape = Shape({2, 3, 4}),
            .memory_layout = TensorMemoryLayout::WIDTH_SHARDED,
            .shard_shape_2d = Shape2D{6, 4},
            .layout = Layout::ROW_MAJOR,
            .shard_shape_nd = Shape({2, 3, 4}),
        },
        LegacyToNdShardingParams{
            .shape = Shape({2, 3, 4}),
            .memory_layout = TensorMemoryLayout::WIDTH_SHARDED,
            .shard_shape_2d = Shape2D{6, 5},
            .layout = Layout::ROW_MAJOR,
            .shard_shape_nd = Shape({2, 3, 5}),
        },
        LegacyToNdShardingParams{
            .shape = Shape({2, 32 * 2, 32 * 2}),
            .memory_layout = TensorMemoryLayout::INTERLEAVED,
            .shard_shape_2d = std::nullopt,
            .layout = Layout::TILE,
            .shard_shape_nd = std::nullopt,
        },
        LegacyToNdShardingParams{
            .shape = Shape({3, 4, 5}),
            .memory_layout = TensorMemoryLayout::INTERLEAVED,
            .shard_shape_2d = std::nullopt,
            .layout = Layout::ROW_MAJOR,
            .shard_shape_nd = std::nullopt,
        }));

INSTANTIATE_TEST_SUITE_P(
    TensorShardingTests,
    NdToLegacyShardingTests,
    ::testing::Values(
        NdToLegacyShardingParams{
            .shape = Shape({2, 32 * 2, 32 * 2}),
            .shard_shape_nd = Shape({1, 32, 32}),
            .layout = Layout::TILE,
            .grid_size = CoreCoord{2, 4},
            .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
            .shard_shape_2d = Shape2D{32, 32},
        },
        NdToLegacyShardingParams{
            .shape = Shape({2, 32 * 2, 32 * 2}),
            .shard_shape_nd = Shape({1, 32, 32}),
            .layout = Layout::TILE,
            .grid_size = CoreCoord{2, 5},
            .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
            .shard_shape_2d = Shape2D{32, 32},
        },
        NdToLegacyShardingParams{
            .shape = Shape({2, 32 * 2, 32 * 2}),
            .shard_shape_nd = Shape({1, 32, 32}),
            .layout = Layout::TILE,
            .grid_size = CoreCoord{3, 4},
            .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
            .shard_shape_2d = std::nullopt,  // Can't convert, different shard distribution
        },
        NdToLegacyShardingParams{
            .shape = Shape({2, 2, 4}),
            .shard_shape_nd = Shape({1, 1, 2}),
            .layout = Layout::ROW_MAJOR,
            .grid_size = CoreCoord{2, 4},
            .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
            .shard_shape_2d = Shape2D{1, 2},
        },
        NdToLegacyShardingParams{
            .shape = Shape({4 * 32, 4 * 32}),
            .shard_shape_nd = Shape({2 * 32, 2 * 32}),
            .layout = Layout::TILE,
            .grid_size = CoreCoord{2, 2},
            .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
            .shard_shape_2d = Shape2D{2 * 32, 2 * 32},
        },
        NdToLegacyShardingParams{
            .shape = Shape({4, 4}),
            .shard_shape_nd = Shape({2, 2}),
            .layout = Layout::ROW_MAJOR,
            .grid_size = CoreCoord{2, 2},
            .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
            .shard_shape_2d = Shape2D{2, 2},
        },
        NdToLegacyShardingParams{
            .shape = Shape({4 * 32}),
            .shard_shape_nd = Shape({32, 2 * 32}),
            .layout = Layout::TILE,
            .grid_size = CoreCoord{7, 7},
            .memory_layout = TensorMemoryLayout::WIDTH_SHARDED,
            .shard_shape_2d = Shape2D{32, 2 * 32},
        },
        NdToLegacyShardingParams{
            .shape = Shape({4}),
            .shard_shape_nd = Shape({2}),
            .layout = Layout::ROW_MAJOR,
            .grid_size = CoreCoord{7, 7},
            .memory_layout = TensorMemoryLayout::WIDTH_SHARDED,
            .shard_shape_2d = Shape2D{1, 2},
        },
        NdToLegacyShardingParams{
            .shape = Shape({}),
            .shard_shape_nd = Shape({32, 2 * 32}),
            .layout = Layout::TILE,
            .grid_size = CoreCoord{7, 7},
            .memory_layout = TensorMemoryLayout::WIDTH_SHARDED,
            .shard_shape_2d = Shape2D{32, 2 * 32},
        },
        NdToLegacyShardingParams{
            .shape = Shape({}),
            .shard_shape_nd = Shape({2}),
            .layout = Layout::ROW_MAJOR,
            .grid_size = CoreCoord{7, 7},
            .memory_layout = TensorMemoryLayout::WIDTH_SHARDED,
            .shard_shape_2d = Shape2D{1, 2},
        },
        NdToLegacyShardingParams{
            .shape = Shape({2, 3, 4}),
            .shard_shape_nd = Shape({2, 3, 2}),
            .layout = Layout::ROW_MAJOR,
            .grid_size = CoreCoord{7, 7},
            .memory_layout = TensorMemoryLayout::WIDTH_SHARDED,
            .shard_shape_2d = Shape2D{6, 2},
        },
        NdToLegacyShardingParams{
            .shape = Shape({2, 3 * 32, 4 * 32}),
            .shard_shape_nd = Shape({2, 3 * 32, 2 * 32}),
            .layout = Layout::TILE,
            .grid_size = CoreCoord{7, 7},
            .memory_layout = TensorMemoryLayout::WIDTH_SHARDED,
            .shard_shape_2d = Shape2D{6 * 32, 2 * 32},
        },
        NdToLegacyShardingParams{
            .shape = Shape({2, 3, 4}),
            .shard_shape_nd = Shape({1, 1, 4}),
            .layout = Layout::ROW_MAJOR,
            .grid_size = CoreCoord{7, 7},
            .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
            .shard_shape_2d = Shape2D{1, 4},
        },
        NdToLegacyShardingParams{
            .shape = Shape({2, 4, 4}),
            .shard_shape_nd = Shape({1, 2, 4}),
            .layout = Layout::ROW_MAJOR,
            .grid_size = CoreCoord{7, 7},
            .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
            .shard_shape_2d = Shape2D{2, 4},
        },
        NdToLegacyShardingParams{
            .shape = Shape({2, 3 * 32, 4 * 32}),
            .shard_shape_nd = Shape({1, 32, 4 * 32}),
            .layout = Layout::TILE,
            .grid_size = CoreCoord{7, 7},
            .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
            .shard_shape_2d = Shape2D{32, 4 * 32},
        },
        NdToLegacyShardingParams{
            .shape = Shape({2, 4 * 32, 4 * 32}),
            .shard_shape_nd = Shape({1, 2 * 32, 4 * 32}),
            .layout = Layout::ROW_MAJOR,
            .grid_size = CoreCoord{7, 7},
            .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
            .shard_shape_2d = Shape2D{2 * 32, 4 * 32},
        },
        NdToLegacyShardingParams{
            .shape = Shape({2 * 32, 2 * 32, 2 * 32}),
            .shard_shape_nd = Shape({2 * 32, 32, 32}),  // sharding along the batch
            .layout = Layout::TILE,
            .grid_size = CoreCoord{2, 2},
            .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
            .shard_shape_2d = std::nullopt,
        },
        NdToLegacyShardingParams{
            .shape = Shape({2, 2, 2}),
            .shard_shape_nd = Shape({2, 1, 1}),  // sharding along the batch
            .layout = Layout::ROW_MAJOR,
            .grid_size = CoreCoord{2, 2},
            .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
            .shard_shape_2d = std::nullopt,
        },
        NdToLegacyShardingParams{
            .shape = Shape({30 * 32, 80 * 32}),
            .shard_shape_nd = Shape({20 * 32, 10 * 32}),
            .layout = Layout::TILE,
            .grid_size = CoreCoord{7, 7},
            .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
            .shard_shape_2d = std::nullopt,  // not enough cores, requires 2 x 8, but only 7 x 7 are present
        },
        NdToLegacyShardingParams{
            .shape = Shape({30 * 32, 80 * 32}),
            .shard_shape_nd = Shape({30 * 32, 10 * 32}),
            .layout = Layout::TILE,
            .grid_size = CoreCoord{7, 7},
            .memory_layout = TensorMemoryLayout::WIDTH_SHARDED,
            .shard_shape_2d = Shape2D{30 * 32, 10 * 32},  // enough cores, linearizes to 8 cores
        },
        NdToLegacyShardingParams{
            .shape = Shape({30 * 32, 500 * 32}),
            .shard_shape_nd = Shape({30 * 32, 10 * 32}),
            .layout = Layout::TILE,
            .grid_size = CoreCoord{7, 7},
            .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
            .shard_shape_2d =
                std::nullopt,  // not enough cores, tries to linearize to 50 cores, but only 49 are present
        },
        NdToLegacyShardingParams{
            .shape = Shape({3, 8}),
            .shard_shape_nd = Shape({2, 1}),
            .layout = Layout::ROW_MAJOR,
            .grid_size = CoreCoord{7, 7},
            .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
            .shard_shape_2d = std::nullopt,  // not enough cores, requires 2 x 8, but only 7 x 7 are present
        },
        NdToLegacyShardingParams{
            .shape = Shape({3, 8}),
            .shard_shape_nd = Shape({3, 1}),
            .layout = Layout::ROW_MAJOR,
            .grid_size = CoreCoord{7, 7},
            .memory_layout = TensorMemoryLayout::WIDTH_SHARDED,
            .shard_shape_2d = Shape2D{3, 1},  // enough cores, linearizes to 8 cores
        },
        NdToLegacyShardingParams{
            .shape = Shape({3, 50}),
            .shard_shape_nd = Shape({3, 1}),
            .layout = Layout::ROW_MAJOR,
            .grid_size = CoreCoord{7, 7},
            .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
            .shard_shape_2d =
                std::nullopt,  // not enough cores, tries to linearize to 50 cores, but only 49 are present
        }));

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
