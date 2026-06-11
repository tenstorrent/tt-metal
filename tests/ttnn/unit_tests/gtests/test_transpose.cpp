// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include <tt-metalium/shape.hpp>

#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::test {
namespace {

using TestTranspose = TTNNFixtureWithDevice;

TEST_F(TestTranspose, PreservesColMajorShardOrientationOnFallbackGeneratedSpec) {
    const auto input_memory_config = MemoryConfig(
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::L1,
        ShardSpec(
            CoreRangeSet{CoreRange{CoreCoord{0, 0}, CoreCoord{0, 5}}},
            Shape2D{160, 16},
            ShardOrientation::COL_MAJOR));
    const auto input_spec = TensorSpec(
        ttnn::Shape({1, 1, 160, 96}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), input_memory_config));

    auto input_tensor = create_device_tensor(input_spec, device_);
    auto output_tensor = ttnn::transpose(input_tensor, /*dim1=*/2, /*dim2=*/3);

    const auto& output_memory_config = output_tensor.memory_config();
    ASSERT_TRUE(output_memory_config.is_sharded());
    ASSERT_TRUE(output_memory_config.shard_spec().has_value());
    EXPECT_EQ(output_memory_config.memory_layout(), input_memory_config.memory_layout());
    EXPECT_EQ(output_memory_config.buffer_type(), input_memory_config.buffer_type());
    EXPECT_EQ(output_memory_config.shard_spec()->orientation, input_memory_config.shard_spec()->orientation);
}

}  // namespace
}  // namespace ttnn::test
