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
#include "ttnn/operations/data_movement/untilize_with_unpadding/untilize_with_unpadding.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::graph::arguments::test {
namespace {

using TestGraphCaptureArgumentsUntilizeWithUnpadding = TTNNFixtureWithDevice;

TEST_F(TestGraphCaptureArgumentsUntilizeWithUnpadding, UntilizeWithUnpadding) {
    tt::tt_metal::TensorSpec tensor_spec(
        ttnn::Shape({1, 10240, 32}),
        TensorLayout(tt::tt_metal::DataType::BFLOAT16, PageConfig(tt::tt_metal::Layout::TILE), DRAM_MEMORY_CONFIG));
    auto input_tensor = create_device_tensor(tensor_spec, device_);

    ttnn::graph::GraphProcessor::begin_graph_capture(tt::tt_metal::IGraphProcessor::RunMode::NORMAL);
    ttnn::untilize_with_unpadding(
        input_tensor,
        ttnn::Shape({0, 10239, 31}),
        ttnn::DRAM_MEMORY_CONFIG,
        true,
        std::nullopt);
    auto trace = ttnn::graph::GraphProcessor::end_graph_capture();
    auto operations = ttnn::graph::extract_arguments(trace);

    const auto it = std::find_if(operations.begin(), operations.end(), [](const auto& op) {
        return op.operation_name == "UntilizeWithUnpaddingDeviceOperation";
    });
    ASSERT_NE(it, operations.end()) << "UntilizeWithUnpaddingDeviceOperation not found";

    const auto& operation = *it;
    ASSERT_EQ(operation.arguments.size(), 2u);

    const auto& op_attrs = operation.arguments[0];
    EXPECT_TRUE(op_attrs.find("UntilizeWithUnpaddingParams(") != std::string::npos);
    EXPECT_TRUE(op_attrs.find("Shape([0, 10239, 31])") != std::string::npos);
    EXPECT_TRUE(op_attrs.find("TensorMemoryLayout::INTERLEAVED") != std::string::npos);
    EXPECT_TRUE(op_attrs.find("BufferType::DRAM") != std::string::npos);

    // The host-side op identity now carries one space heuristic bit (height) instead of
    // separate width and height bits.
    EXPECT_TRUE(op_attrs.find(", 1, 0, 1, std::nullopt)") != std::string::npos);
    EXPECT_TRUE(op_attrs.find(", 1, 0, 0, 1, std::nullopt)") == std::string::npos);

    const auto& tensor_args = operation.arguments[1];
    EXPECT_TRUE(tensor_args.find("Tensor(") != std::string::npos);
    EXPECT_TRUE(tensor_args.find("Shape([1, 10240, 32])") != std::string::npos);
    EXPECT_TRUE(tensor_args.find("DataType::BFLOAT16") != std::string::npos);
    EXPECT_TRUE(tensor_args.find("TilePageConfig") != std::string::npos);
    EXPECT_TRUE(tensor_args.find("DeviceStorage()") != std::string::npos);
}

}  // namespace
}  // namespace ttnn::graph::arguments::test
