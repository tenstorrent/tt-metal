// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

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
struct LegacyToNdShardingParams {
    Shape shape;
    TensorMemoryLayout memory_layout = TensorMemoryLayout::BLOCK_SHARDED;
    std::optional<Shape2D> shard_shape_2d;
    Layout layout = Layout::TILE;

    std::optional<Shape> shard_shape_nd;
};
struct NdToLegacyShardingParams {
    Shape shape;
    Shape shard_shape_nd;
    Layout layout = Layout::TILE;
    CoreCoord grid_size;
    ShardDistributionStrategy shard_distribution_strategy = ShardDistributionStrategy::ROUND_ROBIN_1D;

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
struct NDShardingCoreInfoParams {
    Shape shape_in_pages;
    Shape shard_shape_in_pages;
    CoreCoord grid_size;

    size_t expected_max_num_shards_per_core = 0;
    std::vector<size_t> expected_num_shards_per_core;
    BufferDistributionSpec::CoreGroups expected_groups;
};
struct NDShardingSqueezeRankParams {
    Shape tensor_shape_pages;
    Shape shard_shape_pages;
    Shape expected_tensor_shape_pages;
    Shape expected_shard_shape_pages;
};
enum class ShardingTensorSpecMethod {
    ShardedAcrossDims,
    ShardedAcrossDimsExcept,
    WidthSharded,
    HeightSharded,
    BlockSharded,
};
struct NDShardingTensorSpecParams {
    Shape tensor_shape;
    Layout layout = Layout::TILE;
    std::optional<Tile> tile;
    ShardingTensorSpecMethod method = ShardingTensorSpecMethod::ShardedAcrossDims;
    std::vector<int32_t> dims;
    Shape expected_shard_shape;
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

    auto& storage = std::get<DeviceStorage>(tensor.storage());
    auto buffer = storage.get_mesh_buffer();

    size_t region_size = buffer->page_size();
    while (buffer->size() % (region_size * 2) == 0) {
        region_size *= 2;
    }

    std::vector<uint16_t> partial_readback_data(tensor_data.size());
    std::vector<uint16_t> full_readback_data(tensor_data.size());

    for (size_t region = 0; region < buffer->size() / region_size; region++) {
        size_t region_offset = region * region_size;
        auto buffer_region = BufferRegion{region_offset, region_size};
        auto write_shard_data_transfer = distributed::MeshCommandQueue::ShardDataTransfer{
            .shard_coord = distributed::MeshCoordinate(0, 0),
            .host_data = reinterpret_cast<std::byte*>(tensor_data.data()) + region_offset,
            .region = buffer_region,
        };
        auto read_shard_data_transfer = distributed::MeshCommandQueue::ShardDataTransfer{
            .shard_coord = distributed::MeshCoordinate(0, 0),
            .host_data = reinterpret_cast<std::byte*>(partial_readback_data.data()) + region_offset,
            .region = buffer_region,
        };
        device_->mesh_command_queue().enqueue_write_shards(buffer, {write_shard_data_transfer}, true);
        device_->mesh_command_queue().enqueue_read_shards({read_shard_data_transfer}, buffer, true);
    }
    EXPECT_EQ(tensor_data, partial_readback_data);

    distributed::ReadShard(
        device_->mesh_command_queue(), full_readback_data, buffer, distributed::MeshCoordinate(0, 0), true);
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
        ASSERT_EQ(nd_shard_spec->grid, cores);
    }
}

class NdToLegacyShardingTests : public ::testing::TestWithParam<NdToLegacyShardingParams> {};

TEST_P(NdToLegacyShardingTests, NdToLegacySharding) {
    const auto& params = GetParam();

    CoreRangeSet cores(CoreRange(CoreCoord{0, 0}, CoreCoord{params.grid_size.x - 1, params.grid_size.y - 1}));
    NdShardSpec nd_shard_spec{
        params.shard_shape_nd, cores, ShardOrientation::ROW_MAJOR, params.shard_distribution_strategy};
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

class NDShardingCoreInfoTests : public ::testing::TestWithParam<NDShardingCoreInfoParams> {};

TEST_P(NDShardingCoreInfoTests, TestCoreInfo) {
    const auto& params = GetParam();

    CoreRangeSet cores(CoreRange(CoreCoord{0, 0}, CoreCoord{params.grid_size.x - 1, params.grid_size.y - 1}));
    BufferDistributionSpec dspec(
        params.shape_in_pages, params.shard_shape_in_pages, cores, ShardOrientation::ROW_MAJOR);

    EXPECT_EQ(dspec.max_num_shards_per_core(), params.expected_max_num_shards_per_core);
    EXPECT_EQ(dspec.num_cores(), params.grid_size.x * params.grid_size.y);
    EXPECT_EQ(dspec.num_cores(), params.expected_num_shards_per_core.size());
    for (size_t i = 0; i < dspec.num_cores(); i++) {
        EXPECT_EQ(dspec.num_shards_per_core(i), params.expected_num_shards_per_core[i]);
    }

    const auto& core_groups = dspec.core_groups();
    EXPECT_EQ(core_groups.cores_with_data, params.expected_groups.cores_with_data);
    EXPECT_EQ(core_groups.cores_in_group_1, params.expected_groups.cores_in_group_1);
    EXPECT_EQ(core_groups.cores_in_group_2, params.expected_groups.cores_in_group_2);
    EXPECT_EQ(core_groups.num_shards_per_core_in_group_1, params.expected_groups.num_shards_per_core_in_group_1);
    EXPECT_EQ(core_groups.num_shards_per_core_in_group_2, params.expected_groups.num_shards_per_core_in_group_2);
}

class NDShardingSqueezeRankTests : public ::testing::TestWithParam<NDShardingSqueezeRankParams> {};

TEST_P(NDShardingSqueezeRankTests, TestSqueezeRank) {
    const auto& params = GetParam();

    CoreRangeSet cores(CoreRange(CoreCoord{0, 0}, CoreCoord{6, 6}));
    BufferDistributionSpec dspec(
        params.tensor_shape_pages, params.shard_shape_pages, cores, ShardOrientation::ROW_MAJOR);
    EXPECT_EQ(dspec.tensor_shape_in_pages(), params.expected_tensor_shape_pages);
    EXPECT_EQ(dspec.shard_shape_in_pages(), params.expected_shard_shape_pages);

    if (params.tensor_shape_pages.rank() == params.shard_shape_pages.rank()) {
        auto expected_page_mapping =
            detail::compute_page_mapping(params.tensor_shape_pages, params.shard_shape_pages, dspec.cores());
        EXPECT_EQ(dspec.compute_page_mapping().core_host_page_indices, expected_page_mapping.core_host_page_indices);
    }
}

class NDShardingTensorSpecTests : public ::testing::TestWithParam<NDShardingTensorSpecParams> {};

TEST_P(NDShardingTensorSpecTests, TestTensorSpec) {
    const auto& params = GetParam();

    CoreRangeSet cores(CoreRange(CoreCoord{0, 0}, CoreCoord{3, 3}));
    TensorSpec tensor_spec(
        params.tensor_shape,
        TensorLayout(DataType::FLOAT32, PageConfig(params.layout, params.tile), MemoryConfig(BufferType::L1)));
    switch (params.method) {
        case ShardingTensorSpecMethod::ShardedAcrossDims:
            tensor_spec = tensor_spec.sharded_across_dims(params.dims, cores);
            break;
        case ShardingTensorSpecMethod::ShardedAcrossDimsExcept:
            tensor_spec = tensor_spec.sharded_across_dims_except(params.dims, cores);
            break;
        case ShardingTensorSpecMethod::WidthSharded: tensor_spec = tensor_spec.width_sharded(cores); break;
        case ShardingTensorSpecMethod::HeightSharded: tensor_spec = tensor_spec.height_sharded(cores); break;
        case ShardingTensorSpecMethod::BlockSharded: tensor_spec = tensor_spec.block_sharded(cores.ranges()[0]); break;
    }
    EXPECT_EQ(tensor_spec.memory_config().nd_shard_spec().value().shard_shape, params.expected_shard_shape);
}

class NDShardingSqueezeRankStressTests : public ::testing::Test {};

TEST_F(NDShardingSqueezeRankStressTests, TestSqueezeRankStress) {
    std::function<void(const Shape&, std::vector<uint32_t>&, const std::function<void(const Shape&)>&)>
        iterate_shapes_impl = [&](const Shape& upper_shape,
                                  std::vector<uint32_t>& current_shape,
                                  const std::function<void(const Shape&)>& callback) {
            if (upper_shape.rank() == current_shape.size()) {
                callback(Shape(current_shape));
                return;
            }

            for (int val = 1; val <= upper_shape[current_shape.size()]; val++) {
                current_shape.push_back(val);
                iterate_shapes_impl(upper_shape, current_shape, callback);
                current_shape.pop_back();
            }
        };
    auto iterate_shapes = [&](const Shape& upper_shape, const std::function<void(const Shape&)>& callback) {
        std::vector<uint32_t> tmp_shape;
        iterate_shapes_impl(upper_shape, tmp_shape, callback);
    };

    CoreRangeSet cores(CoreRange(CoreCoord{0, 0}, CoreCoord{6, 6}));
    iterate_shapes(Shape({4, 4, 4, 4}), [&](const Shape& tensor_shape) {
        iterate_shapes(tensor_shape, [&](const Shape& shard_shape) {
            BufferDistributionSpec dspec(tensor_shape, shard_shape, cores, ShardOrientation::ROW_MAJOR);
            auto expected_page_mapping =
                tt::tt_metal::detail::compute_page_mapping(tensor_shape, shard_shape, dspec.cores());
            EXPECT_EQ(
                dspec.compute_page_mapping().core_host_page_indices, expected_page_mapping.core_host_page_indices);
        });
    });
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
            .shard_shape_nd = Shape({32, 32}),
        },
        LegacyToNdShardingParams{
            .shape = Shape({2, 32 * 2, 32 * 2}),
            .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
            .shard_shape_2d = Shape2D{32 * 2, 32},
            .layout = Layout::TILE,
            .shard_shape_nd = Shape({32 * 2, 32}),
        },
        LegacyToNdShardingParams{
            .shape = Shape({2, 32 * 2, 32 * 2}),
            .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
            .shard_shape_2d = Shape2D{32 * 3, 32},
            .layout = Layout::TILE,
            .shard_shape_nd = Shape({32 * 3, 32}),
        },
        LegacyToNdShardingParams{
            .shape = Shape({2, 4, 4}),
            .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
            .shard_shape_2d = Shape2D{2, 2},
            .layout = Layout::ROW_MAJOR,
            .shard_shape_nd = Shape({2, 2}),
        },
        LegacyToNdShardingParams{
            .shape = Shape({2, 4}),
            .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
            .shard_shape_2d = Shape2D{2, 2},
            .layout = Layout::ROW_MAJOR,
            .shard_shape_nd = Shape({2, 2}),
        },
        LegacyToNdShardingParams{
            .shape = Shape({4}),
            .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
            .shard_shape_2d = Shape2D{1, 2},
            .layout = Layout::ROW_MAJOR,
            .shard_shape_nd = Shape({2}),
        },
        LegacyToNdShardingParams{
            .shape = Shape({}),
            .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
            .shard_shape_2d = Shape2D{1, 1},
            .layout = Layout::ROW_MAJOR,
            .shard_shape_nd = Shape({1}),
        },
        LegacyToNdShardingParams{
            .shape = Shape({2, 32 * 2, 32 * 2}),
            .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
            .shard_shape_2d = Shape2D{32, 32 * 2},
            .layout = Layout::TILE,
            .shard_shape_nd = Shape({32, 32 * 2}),
        },
        LegacyToNdShardingParams{
            .shape = Shape({2, 32 * 2, 32 * 2}),
            .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
            .shard_shape_2d = Shape2D{32 * 3, 32 * 2},
            .layout = Layout::TILE,
            .shard_shape_nd = Shape({32 * 3, 32 * 2}),
        },
        LegacyToNdShardingParams{
            .shape = Shape({2, 3, 4}),
            .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
            .shard_shape_2d = Shape2D{3, 4},
            .layout = Layout::ROW_MAJOR,
            .shard_shape_nd = Shape({3, 4}),
        },
        LegacyToNdShardingParams{
            .shape = Shape({2, 3, 4}),
            .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
            .shard_shape_2d = Shape2D{4, 4},
            .layout = Layout::ROW_MAJOR,
            .shard_shape_nd = Shape({4, 4}),
        },
        LegacyToNdShardingParams{
            .shape = Shape({2, 32 * 2, 32 * 2}),
            .memory_layout = TensorMemoryLayout::WIDTH_SHARDED,
            .shard_shape_2d = Shape2D{32 * 4, 32},
            .layout = Layout::TILE,
            .shard_shape_nd = Shape({32 * 4, 32}),
        },
        LegacyToNdShardingParams{
            .shape = Shape({2, 32 * 2, 32 * 2}),
            .memory_layout = TensorMemoryLayout::WIDTH_SHARDED,
            .shard_shape_2d = Shape2D{32 * 4, 32 * 3},
            .layout = Layout::TILE,
            .shard_shape_nd = Shape({32 * 4, 32 * 3}),
        },
        LegacyToNdShardingParams{
            .shape = Shape({2, 3, 4}),
            .memory_layout = TensorMemoryLayout::WIDTH_SHARDED,
            .shard_shape_2d = Shape2D{6, 4},
            .layout = Layout::ROW_MAJOR,
            .shard_shape_nd = Shape({6, 4}),
        },
        LegacyToNdShardingParams{
            .shape = Shape({2, 3, 4}),
            .memory_layout = TensorMemoryLayout::WIDTH_SHARDED,
            .shard_shape_2d = Shape2D{6, 5},
            .layout = Layout::ROW_MAJOR,
            .shard_shape_nd = Shape({6, 5}),
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
            .shape = Shape({2, 32 * 2, 32 * 2}),
            .shard_shape_nd = Shape({1, 32, 32}),
            .layout = Layout::TILE,
            .grid_size = CoreCoord{3, 4},
            .shard_distribution_strategy = ShardDistributionStrategy::GRID_2D,
            .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
            .shard_shape_2d = Shape2D{32, 32},
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

INSTANTIATE_TEST_SUITE_P(
    TensorShardingTests,
    NDShardingCoreInfoTests,
    ::testing::Values(
        NDShardingCoreInfoParams{
            .shape_in_pages = Shape({2, 2, 2}),
            .shard_shape_in_pages = Shape({1, 1, 1}),
            .grid_size = CoreCoord{2, 2},
            .expected_max_num_shards_per_core = 2,
            .expected_num_shards_per_core = {2, 2, 2, 2},
            .expected_groups =
                {
                    .cores_with_data = CoreRangeSet(CoreRange({0, 0}, {1, 1})),
                    .cores_in_group_1 = CoreRangeSet(CoreRange({0, 0}, {1, 1})),
                    .num_shards_per_core_in_group_1 = 2,
                },
        },
        NDShardingCoreInfoParams{
            .shape_in_pages = Shape({3, 3, 3}),
            .shard_shape_in_pages = Shape({2, 2, 2}),
            .grid_size = CoreCoord{2, 2},
            .expected_max_num_shards_per_core = 2,
            .expected_num_shards_per_core = {2, 2, 2, 2},
            .expected_groups =
                {
                    .cores_with_data = CoreRangeSet(CoreRange({0, 0}, {1, 1})),
                    .cores_in_group_1 = CoreRangeSet(CoreRange({0, 0}, {1, 1})),
                    .num_shards_per_core_in_group_1 = 2,
                },
        },
        NDShardingCoreInfoParams{
            .shape_in_pages = Shape({0, 0, 0}),
            .shard_shape_in_pages = Shape({2, 2, 2}),
            .grid_size = CoreCoord{2, 2},
            .expected_max_num_shards_per_core = 0,
            .expected_num_shards_per_core = {0, 0, 0, 0},
            .expected_groups = {},
        },
        NDShardingCoreInfoParams{
            .shape_in_pages = Shape({2, 2, 2}),
            .shard_shape_in_pages = Shape({2, 2, 2}),
            .grid_size = CoreCoord{2, 2},
            .expected_max_num_shards_per_core = 1,
            .expected_num_shards_per_core = {1, 0, 0, 0},
            .expected_groups =
                {
                    .cores_with_data = CoreRangeSet(CoreRange({0, 0})),
                    .cores_in_group_1 = CoreRangeSet(CoreRange({0, 0})),
                    .num_shards_per_core_in_group_1 = 1,
                },
        },
        NDShardingCoreInfoParams{
            .shape_in_pages = Shape({2, 2, 4}),
            .shard_shape_in_pages = Shape({2, 2, 2}),
            .grid_size = CoreCoord{2, 2},
            .expected_max_num_shards_per_core = 1,
            .expected_num_shards_per_core = {1, 1, 0, 0},
            .expected_groups =
                {
                    .cores_with_data = CoreRangeSet(CoreRange({0, 0}, {1, 0})),
                    .cores_in_group_1 = CoreRangeSet(CoreRange({0, 0}, {1, 0})),
                    .num_shards_per_core_in_group_1 = 1,
                },
        },
        NDShardingCoreInfoParams{
            .shape_in_pages = Shape({2, 6, 2}),
            .shard_shape_in_pages = Shape({2, 2, 2}),
            .grid_size = CoreCoord{2, 2},
            .expected_max_num_shards_per_core = 1,
            .expected_num_shards_per_core = {1, 1, 1, 0},
            .expected_groups =
                {
                    .cores_with_data = CoreRangeSet(std::vector<CoreRange>{
                        CoreRange({0, 0}, {1, 0}), CoreRange({0, 1})}),
                    .cores_in_group_1 = CoreRangeSet(std::vector<CoreRange>{
                        CoreRange({0, 0}, {1, 0}), CoreRange({0, 1})}),
                    .num_shards_per_core_in_group_1 = 1,
                },
        },
        NDShardingCoreInfoParams{
            .shape_in_pages = Shape({8, 2, 2}),
            .shard_shape_in_pages = Shape({2, 2, 2}),
            .grid_size = CoreCoord{2, 2},
            .expected_max_num_shards_per_core = 1,
            .expected_num_shards_per_core = {1, 1, 1, 1},
            .expected_groups =
                {
                    .cores_with_data = CoreRangeSet(CoreRange({0, 0}, {1, 1})),
                    .cores_in_group_1 = CoreRangeSet(CoreRange({0, 0}, {1, 1})),
                    .num_shards_per_core_in_group_1 = 1,
                },
        },
        NDShardingCoreInfoParams{
            .shape_in_pages = Shape({1, 1, 33}),
            .shard_shape_in_pages = Shape({2, 2, 2}),
            .grid_size = CoreCoord{2, 2},
            .expected_max_num_shards_per_core = 5,
            .expected_num_shards_per_core = {5, 4, 4, 4},
            .expected_groups =
                {
                    .cores_with_data = CoreRangeSet(CoreRange({0, 0}, {1, 1})),
                    .cores_in_group_1 = CoreRangeSet(CoreRange({0, 0})),
                    .cores_in_group_2 = CoreRangeSet(std::vector<CoreRange>{
                        CoreRange({0, 1}, {1, 1}), CoreRange({1, 0})}),
                    .num_shards_per_core_in_group_1 = 5,
                    .num_shards_per_core_in_group_2 = 4,
                },
        },
        NDShardingCoreInfoParams{
            .shape_in_pages = Shape({1, 35, 1}),
            .shard_shape_in_pages = Shape({2, 2, 2}),
            .grid_size = CoreCoord{2, 2},
            .expected_max_num_shards_per_core = 5,
            .expected_num_shards_per_core = {5, 5, 4, 4},
            .expected_groups =
                {
                    .cores_with_data = CoreRangeSet(CoreRange({0, 0}, {1, 1})),
                    .cores_in_group_1 = CoreRangeSet(CoreRange({0, 0}, {1, 0})),
                    .cores_in_group_2 = CoreRangeSet(CoreRange({0, 1}, {1, 1})),
                    .num_shards_per_core_in_group_1 = 5,
                    .num_shards_per_core_in_group_2 = 4,
                },
        },
        NDShardingCoreInfoParams{
            .shape_in_pages = Shape({37, 1, 1}),
            .shard_shape_in_pages = Shape({2, 2, 2}),
            .grid_size = CoreCoord{2, 2},
            .expected_max_num_shards_per_core = 5,
            .expected_num_shards_per_core = {5, 5, 5, 4},
            .expected_groups =
                {
                    .cores_with_data = CoreRangeSet(CoreRange({0, 0}, {1, 1})),
                    .cores_in_group_1 = CoreRangeSet(std::vector<CoreRange>{
                        CoreRange({0, 0}, {1, 0}), CoreRange({0, 1})}),
                    .cores_in_group_2 = CoreRangeSet(CoreRange({1, 1})),
                    .num_shards_per_core_in_group_1 = 5,
                    .num_shards_per_core_in_group_2 = 4,
                },
        }));

INSTANTIATE_TEST_SUITE_P(
    TensorShardingTests,
    NDShardingSqueezeRankTests,
    ::testing::Values(
        NDShardingSqueezeRankParams{
            .tensor_shape_pages = Shape({5}),
            .shard_shape_pages = Shape({3}),
            // Nothing to minimize
            .expected_tensor_shape_pages = Shape({5}),
            .expected_shard_shape_pages = Shape({3}),
        },
        NDShardingSqueezeRankParams{
            .tensor_shape_pages = Shape({3, 6}),
            .shard_shape_pages = Shape({2, 2}),
            // Nothing to minimize
            .expected_tensor_shape_pages = Shape({3, 6}),
            .expected_shard_shape_pages = Shape({2, 2}),
        },
        NDShardingSqueezeRankParams{
            .tensor_shape_pages = Shape({1, 3, 6}),
            .shard_shape_pages = Shape({2, 2}),
            // Leading tensor dimension higher than shard dimension must be folded
            .expected_tensor_shape_pages = Shape({3, 6}),
            .expected_shard_shape_pages = Shape({2, 2}),
        },
        NDShardingSqueezeRankParams{
            .tensor_shape_pages = Shape({5, 3, 6}),
            .shard_shape_pages = Shape({2, 2}),
            // Leading tensor dimension higher than shard dimension must be folded
            .expected_tensor_shape_pages = Shape({15, 6}),
            .expected_shard_shape_pages = Shape({2, 2}),
        },
        NDShardingSqueezeRankParams{
            .tensor_shape_pages = Shape({1, 3, 6}),
            .shard_shape_pages = Shape({1, 2, 2}),
            // Folding leading 1s in both shapes
            .expected_tensor_shape_pages = Shape({3, 6}),
            .expected_shard_shape_pages = Shape({2, 2}),
        },
        NDShardingSqueezeRankParams{
            .tensor_shape_pages = Shape({1, 1, 3, 6}),
            .shard_shape_pages = Shape({1, 1, 2, 2}),
            // Folding leading 1s in both shapes
            .expected_tensor_shape_pages = Shape({3, 6}),
            .expected_shard_shape_pages = Shape({2, 2}),
        },
        NDShardingSqueezeRankParams{
            .tensor_shape_pages = Shape({2, 3, 6}),
            .shard_shape_pages = Shape({1, 2, 2}),
            // Can't fold dim 0, because it would cause different paddings in dim 1
            .expected_tensor_shape_pages = Shape({2, 3, 6}),
            .expected_shard_shape_pages = Shape({1, 2, 2}),
        },
        NDShardingSqueezeRankParams{
            .tensor_shape_pages = Shape({2, 3, 6}),
            .shard_shape_pages = Shape({2, 2, 2}),
            // Can't fold dim 0, because it would cause different paddings in dim 1
            .expected_tensor_shape_pages = Shape({2, 3, 6}),
            .expected_shard_shape_pages = Shape({2, 2, 2}),
        },
        NDShardingSqueezeRankParams{
            .tensor_shape_pages = Shape({4, 3, 6}),
            .shard_shape_pages = Shape({2, 2, 2}),
            // True ND sharding, nothing to fold
            .expected_tensor_shape_pages = Shape({4, 3, 6}),
            .expected_shard_shape_pages = Shape({2, 2, 2}),
        },
        NDShardingSqueezeRankParams{
            .tensor_shape_pages = Shape({4, 4, 6}),
            .shard_shape_pages = Shape({1, 3, 2}),
            // Can't fold dim 0, because it would cause different paddings in dim 1
            .expected_tensor_shape_pages = Shape({4, 4, 6}),
            .expected_shard_shape_pages = Shape({1, 3, 2}),
        },
        NDShardingSqueezeRankParams{
            .tensor_shape_pages = Shape({4, 4, 6}),
            .shard_shape_pages = Shape({1, 2, 2}),
            // No padding in dim 1, so we can fold dim 0
            .expected_tensor_shape_pages = Shape({16, 6}),
            .expected_shard_shape_pages = Shape({2, 2}),
        },
        NDShardingSqueezeRankParams{
            .tensor_shape_pages = Shape({4, 4, 6}),
            .shard_shape_pages = Shape({2, 2, 2}),
            // True ND sharding, nothing to fold
            .expected_tensor_shape_pages = Shape({4, 4, 6}),
            .expected_shard_shape_pages = Shape({2, 2, 2}),
        },
        NDShardingSqueezeRankParams{
            .tensor_shape_pages = Shape({4, 4, 5, 5, 4, 6}),
            .shard_shape_pages = Shape({4, 3, 5, 5, 2, 2}),
            // Folding identical dimensions in tensor and shard shapes into the leading dimension
            .expected_tensor_shape_pages = Shape({4, 100, 4, 6}),
            .expected_shard_shape_pages = Shape({4, 75, 2, 2}),
        },
        NDShardingSqueezeRankParams{
            .tensor_shape_pages = Shape({4, 4, 5, 5, 4, 6}),
            .shard_shape_pages = Shape({4, 3, 1, 1, 2, 2}),
            // Folding 1 shard dimensions into the following dimension, because it doesn't have a padding
            .expected_tensor_shape_pages = Shape({4, 4, 100, 6}),
            .expected_shard_shape_pages = Shape({4, 3, 2, 2}),
        },
        NDShardingSqueezeRankParams{
            .tensor_shape_pages = Shape({4, 4, 5, 5, 4, 6}),
            .shard_shape_pages = Shape({4, 3, 1, 1, 3, 2}),
            // Folding 1 shard dimensions, but not into the following dimension, because it has a padding
            .expected_tensor_shape_pages = Shape({4, 4, 25, 4, 6}),
            .expected_shard_shape_pages = Shape({4, 3, 1, 3, 2}),
        },
        NDShardingSqueezeRankParams{
            .tensor_shape_pages = Shape({4, 4, 5, 5, 4, 6}),
            .shard_shape_pages = Shape({4, 4, 1, 1, 3, 2}),
            .expected_tensor_shape_pages = Shape({16, 25, 4, 6}),
            .expected_shard_shape_pages = Shape({16, 1, 3, 2}),
        },
        NDShardingSqueezeRankParams{
            .tensor_shape_pages = Shape({5, 1, 11, 11}),
            .shard_shape_pages = Shape({5, 1, 1}),
            .expected_tensor_shape_pages = Shape({5, 121}),
            .expected_shard_shape_pages = Shape({5, 1}),
        }));

INSTANTIATE_TEST_SUITE_P(
    TensorShardingTests,
    NDShardingTensorSpecTests,
    ::testing::Values(
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({5, 3, 4 * 32, 4 * 32}),
            .layout = Layout::TILE,
            .method = ShardingTensorSpecMethod::ShardedAcrossDims,
            .dims = {-1},
            .expected_shard_shape = Shape({5, 3, 4 * 32, 32}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({5, 3, 4 * 32, 4 * 32}),
            .layout = Layout::TILE,
            .method = ShardingTensorSpecMethod::ShardedAcrossDims,
            .dims = {-2},
            .expected_shard_shape = Shape({5, 3, 32, 4 * 32}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({5, 3, 4 * 32, 4 * 32}),
            .layout = Layout::TILE,
            .method = ShardingTensorSpecMethod::ShardedAcrossDims,
            .dims = {-1, -2},
            .expected_shard_shape = Shape({5, 3, 32, 32}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({5, 3, 4 * 32, 4 * 32}),
            .layout = Layout::TILE,
            .tile = Tile({16, 16}),
            .method = ShardingTensorSpecMethod::ShardedAcrossDims,
            .dims = {-1, -2},
            .expected_shard_shape = Shape({5, 3, 16, 16}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({5, 3, 4 * 32, 4 * 32}),
            .layout = Layout::TILE,
            .tile = Tile({16, 32}),
            .method = ShardingTensorSpecMethod::ShardedAcrossDims,
            .dims = {-1, -2},
            .expected_shard_shape = Shape({5, 3, 16, 32}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({5, 3, 4 * 32, 4 * 32}),
            .layout = Layout::TILE,
            .tile = Tile({32, 16}),
            .method = ShardingTensorSpecMethod::ShardedAcrossDims,
            .dims = {-1, -2},
            .expected_shard_shape = Shape({5, 3, 32, 16}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({5, 3, 4 * 32, 4 * 32}),
            .layout = Layout::TILE,
            .method = ShardingTensorSpecMethod::ShardedAcrossDims,
            .dims = {0},
            .expected_shard_shape = Shape({1, 3, 4 * 32, 4 * 32}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({5, 3, 4 * 32, 4 * 32}),
            .layout = Layout::TILE,
            .method = ShardingTensorSpecMethod::ShardedAcrossDims,
            .dims = {0, 1},
            .expected_shard_shape = Shape({1, 1, 4 * 32, 4 * 32}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({1, 1, 1, 1}),
            .layout = Layout::ROW_MAJOR,
            .method = ShardingTensorSpecMethod::ShardedAcrossDims,
            .dims = {0},
            .expected_shard_shape = Shape({1, 1, 1, 16}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({5, 3, 8, 32}),
            .layout = Layout::ROW_MAJOR,
            .method = ShardingTensorSpecMethod::ShardedAcrossDims,
            .dims = {-1},
            .expected_shard_shape = Shape({5, 3, 8, 16}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({5, 3, 8, 32}),
            .layout = Layout::ROW_MAJOR,
            .method = ShardingTensorSpecMethod::ShardedAcrossDims,
            .dims = {-2},
            .expected_shard_shape = Shape({5, 3, 1, 32}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({5, 3, 8, 32}),
            .layout = Layout::ROW_MAJOR,
            .method = ShardingTensorSpecMethod::ShardedAcrossDims,
            .dims = {-1, -2},
            .expected_shard_shape = Shape({5, 3, 1, 16}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({5, 3, 8, 32}),
            .layout = Layout::ROW_MAJOR,
            .method = ShardingTensorSpecMethod::ShardedAcrossDims,
            .dims = {0},
            .expected_shard_shape = Shape({1, 3, 8, 32}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({5, 3, 8, 32}),
            .layout = Layout::ROW_MAJOR,
            .method = ShardingTensorSpecMethod::ShardedAcrossDims,
            .dims = {0, 1},
            .expected_shard_shape = Shape({1, 1, 8, 32}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({5, 3, 4 * 32, 4 * 32}),
            .layout = Layout::TILE,
            .method = ShardingTensorSpecMethod::ShardedAcrossDimsExcept,
            .dims = {-1},
            .expected_shard_shape = Shape({1, 1, 32, 4 * 32}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({5, 3, 4 * 32, 4 * 32}),
            .layout = Layout::TILE,
            .method = ShardingTensorSpecMethod::ShardedAcrossDimsExcept,
            .dims = {-2},
            .expected_shard_shape = Shape({1, 1, 4 * 32, 32}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({5, 3, 4 * 32, 4 * 32}),
            .layout = Layout::TILE,
            .method = ShardingTensorSpecMethod::ShardedAcrossDimsExcept,
            .dims = {-1, -2},
            .expected_shard_shape = Shape({1, 1, 4 * 32, 4 * 32}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({5, 3, 4 * 32, 4 * 32}),
            .layout = Layout::TILE,
            .method = ShardingTensorSpecMethod::ShardedAcrossDimsExcept,
            .dims = {0},
            .expected_shard_shape = Shape({5, 1, 32, 32}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({5, 3, 4 * 32, 4 * 32}),
            .layout = Layout::TILE,
            .method = ShardingTensorSpecMethod::ShardedAcrossDimsExcept,
            .dims = {0, 1},
            .expected_shard_shape = Shape({5, 3, 32, 32}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({1, 1, 1, 1}),
            .layout = Layout::ROW_MAJOR,
            .method = ShardingTensorSpecMethod::ShardedAcrossDimsExcept,
            .dims = {0},
            .expected_shard_shape = Shape({1, 1, 1, 16}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({5, 3, 8, 32}),
            .layout = Layout::ROW_MAJOR,
            .method = ShardingTensorSpecMethod::ShardedAcrossDimsExcept,
            .dims = {-1},
            .expected_shard_shape = Shape({1, 1, 1, 32}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({5, 3, 8, 32}),
            .layout = Layout::ROW_MAJOR,
            .method = ShardingTensorSpecMethod::ShardedAcrossDimsExcept,
            .dims = {-2},
            .expected_shard_shape = Shape({1, 1, 8, 16}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({5, 3, 8, 32}),
            .layout = Layout::ROW_MAJOR,
            .method = ShardingTensorSpecMethod::ShardedAcrossDimsExcept,
            .dims = {-1, -2},
            .expected_shard_shape = Shape({1, 1, 8, 32}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({5, 3, 8, 32}),
            .layout = Layout::ROW_MAJOR,
            .method = ShardingTensorSpecMethod::ShardedAcrossDimsExcept,
            .dims = {0},
            .expected_shard_shape = Shape({5, 1, 1, 16}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({5, 3, 8, 32}),
            .layout = Layout::ROW_MAJOR,
            .method = ShardingTensorSpecMethod::ShardedAcrossDimsExcept,
            .dims = {0, 1},
            .expected_shard_shape = Shape({5, 3, 1, 16}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({4 * 32, 4 * 32}),
            .layout = Layout::TILE,
            .method = ShardingTensorSpecMethod::BlockSharded,
            .expected_shard_shape = Shape({32, 32}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({8 * 32, 4 * 32}),
            .layout = Layout::TILE,
            .method = ShardingTensorSpecMethod::BlockSharded,
            .expected_shard_shape = Shape({2 * 32, 32}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({4 * 32, 8 * 32}),
            .layout = Layout::TILE,
            .method = ShardingTensorSpecMethod::BlockSharded,
            .expected_shard_shape = Shape({32, 2 * 32}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({8 * 32, 8 * 32}),
            .layout = Layout::TILE,
            .method = ShardingTensorSpecMethod::BlockSharded,
            .expected_shard_shape = Shape({2 * 32, 2 * 32}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({500, 500}),
            .layout = Layout::TILE,
            .method = ShardingTensorSpecMethod::BlockSharded,
            .expected_shard_shape = Shape({4 * 32, 4 * 32}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({4, 5, 30, 500}),
            .layout = Layout::TILE,
            .method = ShardingTensorSpecMethod::BlockSharded,
            .expected_shard_shape = Shape({5 * 32, 4 * 32}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({1, 1}),
            .layout = Layout::ROW_MAJOR,
            .method = ShardingTensorSpecMethod::BlockSharded,
            .expected_shard_shape = Shape({1, 16}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({800, 1200}),
            .layout = Layout::ROW_MAJOR,
            .method = ShardingTensorSpecMethod::BlockSharded,
            .expected_shard_shape = Shape({200, 304}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({333, 555}),
            .layout = Layout::ROW_MAJOR,
            .method = ShardingTensorSpecMethod::BlockSharded,
            .expected_shard_shape = Shape({84, 144}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({2, 3, 333, 555}),
            .layout = Layout::ROW_MAJOR,
            .method = ShardingTensorSpecMethod::BlockSharded,
            .expected_shard_shape = Shape({500, 144}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({32 * 32, 128}),
            .layout = Layout::TILE,
            .method = ShardingTensorSpecMethod::HeightSharded,
            .expected_shard_shape = Shape({2 * 32, 128}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({37 * 32, 128}),
            .layout = Layout::TILE,
            .method = ShardingTensorSpecMethod::HeightSharded,
            .expected_shard_shape = Shape({3 * 32, 128}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({5, 37 * 32, 128}),
            .layout = Layout::TILE,
            .method = ShardingTensorSpecMethod::HeightSharded,
            .expected_shard_shape = Shape({12 * 32, 128}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({160, 17}),
            .layout = Layout::ROW_MAJOR,
            .method = ShardingTensorSpecMethod::HeightSharded,
            .expected_shard_shape = Shape({10, 17}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({167, 17}),
            .layout = Layout::ROW_MAJOR,
            .method = ShardingTensorSpecMethod::HeightSharded,
            .expected_shard_shape = Shape({11, 17}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({5, 167, 17}),
            .layout = Layout::ROW_MAJOR,
            .method = ShardingTensorSpecMethod::HeightSharded,
            .expected_shard_shape = Shape({53, 17}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({128, 32 * 32}),
            .layout = Layout::TILE,
            .method = ShardingTensorSpecMethod::WidthSharded,
            .expected_shard_shape = Shape({128, 2 * 32}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({128, 37 * 32}),
            .layout = Layout::TILE,
            .method = ShardingTensorSpecMethod::WidthSharded,
            .expected_shard_shape = Shape({128, 3 * 32}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({5, 128, 37 * 32}),
            .layout = Layout::TILE,
            .method = ShardingTensorSpecMethod::WidthSharded,
            .expected_shard_shape = Shape({5 * 128, 3 * 32}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({17, 160}),
            .layout = Layout::ROW_MAJOR,
            .method = ShardingTensorSpecMethod::WidthSharded,
            .expected_shard_shape = Shape({17, 10}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({17, 167}),
            .layout = Layout::ROW_MAJOR,
            .method = ShardingTensorSpecMethod::WidthSharded,
            .expected_shard_shape = Shape({17, 11}),
        },
        NDShardingTensorSpecParams{
            .tensor_shape = Shape({5, 17, 167}),
            .layout = Layout::ROW_MAJOR,
            .method = ShardingTensorSpecMethod::WidthSharded,
            .expected_shard_shape = Shape({5 * 17, 11}),
        }));
