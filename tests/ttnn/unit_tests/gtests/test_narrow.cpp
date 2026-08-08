// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include <tt-metalium/shape.hpp>

#include "ttnn/operations/data_movement/narrow/narrow.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::test {
namespace {

using TestNarrow = TTNNFixtureWithDevice;

TEST_F(TestNarrow, PreservesColMajorShardOrientation) {
    const auto input_memory_config = MemoryConfig(
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        ShardSpec(
            CoreRangeSet{CoreRange{CoreCoord{0, 0}, CoreCoord{3, 3}}},
            Shape2D{64, 128},
            ShardOrientation::COL_MAJOR));
    const auto input_spec = TensorSpec(
        ttnn::Shape({1, 8, 128, 128}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), input_memory_config));

    auto input_tensor = create_device_tensor(input_spec, device_);
    auto output_tensor = ttnn::narrow(input_tensor, /*narrow_dim=*/2, /*narrow_start=*/96, /*length=*/32);

    const auto& output_memory_config = output_tensor.memory_config();
    ASSERT_TRUE(output_memory_config.shard_spec().has_value());
    ASSERT_TRUE(input_memory_config.shard_spec().has_value());
    EXPECT_EQ(output_memory_config.memory_layout(), input_memory_config.memory_layout());
    EXPECT_EQ(output_memory_config.buffer_type(), input_memory_config.buffer_type());
    EXPECT_EQ(output_memory_config.shard_spec()->orientation, input_memory_config.shard_spec()->orientation);
}

}  // namespace
}  // namespace ttnn::test
