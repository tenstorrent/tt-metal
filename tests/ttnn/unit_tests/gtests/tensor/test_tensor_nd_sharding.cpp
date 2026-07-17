// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include <tt_stl/reflection.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn_test_fixtures.hpp"
#include <tt-metalium/distributed.hpp>

namespace {
struct NDShardingOpCompatParams {
    Shape shape;
    Shape shard_shape;
    CoreCoord grid_size;
};
}  // namespace

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
