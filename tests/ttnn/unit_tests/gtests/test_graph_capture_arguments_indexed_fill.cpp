// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <nlohmann/json.hpp>
#include "ttnn/tensor/tensor_ops.hpp"
#include <algorithm>
#include <vector>

#include <tt-metalium/graph_tracking.hpp>
#include "gtest/gtest.h"
#include <tt-metalium/shape.hpp>
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"
#include "ttnn/operations/data_movement/indexed_fill/indexed_fill.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::graph::arguments::test {
namespace {

using TestGraphCaptureArgumentsIndexedFill = TTNNFixtureWithDevice;

TEST_F(TestGraphCaptureArgumentsIndexedFill, IndexedFillPreservesNdOutputMemoryConfig) {
    const CoreRangeSet cores(CoreRange(CoreCoord{0, 0}, CoreCoord{2, 3}));
    const MemoryConfig input_memory_config(
        BufferType::L1, NdShardSpec{Shape({1, 1, 32, 32}), cores, ShardOrientation::ROW_MAJOR});
    const TensorSpec input_spec(
        ttnn::Shape({2, 1, 64, 64}),
        TensorLayout(tt::tt_metal::DataType::BFLOAT16, PageConfig(tt::tt_metal::Layout::TILE), input_memory_config));
    ASSERT_TRUE(input_spec.memory_config().created_with_nd_shard_spec());
    ASSERT_FALSE(input_spec.memory_config().shard_spec().has_value());

    const TensorSpec input_b_spec(
        ttnn::Shape({1, 1, 64, 64}),
        TensorLayout(tt::tt_metal::DataType::BFLOAT16, PageConfig(tt::tt_metal::Layout::TILE), L1_MEMORY_CONFIG));
    const TensorSpec batch_id_spec(
        ttnn::Shape({1, 1, 1, 1}),
        TensorLayout(tt::tt_metal::DataType::UINT32, PageConfig(tt::tt_metal::Layout::ROW_MAJOR), L1_MEMORY_CONFIG));

    auto input_tensor_a = create_device_tensor(input_spec, device_);
    auto input_tensor_b = create_device_tensor(input_b_spec, device_);
    auto batch_id = create_device_tensor(batch_id_spec, device_);

    ttnn::graph::GraphProcessor::begin_graph_capture(tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH);
    ttnn::indexed_fill(batch_id, input_tensor_a, input_tensor_b);
    auto trace = ttnn::graph::GraphProcessor::end_graph_capture();
    auto operations = ttnn::graph::extract_arguments(trace);

    const auto it = std::find_if(operations.begin(), operations.end(), [](const auto& op) {
        return op.operation_name == "IndexedFillDeviceOperation";
    });
    ASSERT_NE(it, operations.end()) << "IndexedFillDeviceOperation not found";

    const auto& operation = *it;
    ASSERT_EQ(operation.arguments.size(), 2u);

    const auto& op_attrs = operation.arguments[0];
    EXPECT_TRUE(op_attrs.find("TensorMemoryLayout::ND_SHARDED") != std::string::npos) << op_attrs;
    EXPECT_TRUE(op_attrs.find("created_with_nd_shard_spec=1") != std::string::npos) << op_attrs;
    EXPECT_TRUE(op_attrs.find("shard_spec=std::nullopt") != std::string::npos) << op_attrs;
    EXPECT_TRUE(op_attrs.find("nd_shard_spec={\"shard_shape\":[1, 1, 32, 32]") != std::string::npos) << op_attrs;

    const auto& tensor_args = operation.arguments[1];
    EXPECT_TRUE(tensor_args.find("Tensor(") != std::string::npos);
    EXPECT_TRUE(tensor_args.find("Shape([2, 1, 64, 64])") != std::string::npos);
    EXPECT_TRUE(tensor_args.find("DataType::BFLOAT16") != std::string::npos);
    EXPECT_TRUE(tensor_args.find("TilePageConfig") != std::string::npos);
    EXPECT_TRUE(tensor_args.find("DeviceStorage()") != std::string::npos);
}

}  // namespace
}  // namespace ttnn::graph::arguments::test
