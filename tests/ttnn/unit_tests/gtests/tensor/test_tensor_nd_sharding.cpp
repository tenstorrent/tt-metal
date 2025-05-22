// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "ttnn/tensor/tensor.hpp"

#include "ttnn_test_fixtures.hpp"

namespace {
struct NDShardingParams {
    Shape shape;
    Shape shard_shape;
    Layout layout = Layout::TILE;
};
}  // namespace
// namespace

class NDShardingTests : public ttnn::TTNNFixtureWithDevice, public ::testing::WithParamInterface<NDShardingParams> {};

TEST_P(NDShardingTests, ReadWriteTest) {
    const auto& params = GetParam();

    CoreRangeSet cores(CoreRange(CoreCoord{0, 0}, CoreCoord{6, 6}));

    for (auto sharding_orientation : {ShardOrientation::ROW_MAJOR, ShardOrientation::COL_MAJOR}) {
        MemoryConfig memory_config{BufferType::L1, NdShardSpec{params.shard_shape, cores, sharding_orientation}};
        TensorLayout tensor_layout(DataType::UINT16, PageConfig(params.layout), memory_config);
        TensorSpec tensor_spec(params.shape, tensor_layout);

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
}

INSTANTIATE_TEST_SUITE_P(
    TensorShardingTests,
    NDShardingTests,
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
        }));
