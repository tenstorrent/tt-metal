// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <nlohmann/json.hpp>
#include "ttnn/tensor/tensor_ops.hpp"
#include <map>
#include <optional>
#include <vector>

#include <tt-metalium/graph_tracking.hpp>
#include "gtest/gtest.h"
#include <tt-metalium/shape.hpp>
#include "ttnn/decorators.hpp"
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"
#include "ttnn/operations/moreh/moreh_dot/moreh_dot.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::graph::arguments::test {
namespace {

using TestGraphCaptureArgumentsMorehDot = TTNNFixtureWithDevice;

TEST_F(TestGraphCaptureArgumentsMorehDot, MorehDot) {
    TensorSpec tensor_spec(
        ttnn::Shape({1, 1, 1, 32}),
        TensorLayout(tt::tt_metal::DataType::BFLOAT16, PageConfig(tt::tt_metal::Layout::TILE), L1_MEMORY_CONFIG));
    auto tt_input1 = create_device_tensor(tensor_spec, device_);
    auto tt_input2 = create_device_tensor(tensor_spec, device_);

    ttnn::graph::GraphProcessor::begin_graph_capture(tt::tt_metal::IGraphProcessor::RunMode::NORMAL);
    ttnn::moreh_dot(tt_input1, tt_input2, std::nullopt, DataType::BFLOAT16, std::nullopt, std::nullopt);
    auto trace = ttnn::graph::GraphProcessor::end_graph_capture();
    auto operations = ttnn::graph::extract_arguments(trace);

    // High-level C++ functions (like ttnn::moreh_dot) are not traced.
    // Only device operations and low-level operations are captured.

    // operations[0]: MorehDotOperation (device operation)
    const auto& operation0 = operations[0];
    EXPECT_EQ(operation0.operation_name, "MorehDotOperation");
    EXPECT_EQ(operation0.arguments.size(), 2);
    EXPECT_EQ(
        operation0.arguments[0],
        "[ unsupported type , "
        "std::reference_wrapper<ttnn::operations::moreh::moreh_dot::MorehDotOperation::operation_attributes_t const>]");
    EXPECT_EQ(
        operation0.arguments[1],
        "[ unsupported type , std::reference_wrapper<std::vector<std::reference_wrapper<tt::tt_metal::Tensor const>, "
        "std::allocator<std::reference_wrapper<tt::tt_metal::Tensor const> > > >]");

    // operations[1]: tt::tt_metal::create_device_tensor (output tensor creation)
    const auto& operation1 = operations[1];
    EXPECT_EQ(operation1.operation_name, "tt::tt_metal::create_device_tensor");
    EXPECT_EQ(operation1.arguments.size(), 5);
    EXPECT_EQ(operation1.arguments[0], "Shape([1, 1, 1, 1])");
    EXPECT_EQ(operation1.arguments[1], "DataType::BFLOAT16");
    EXPECT_EQ(operation1.arguments[2], "Layout::TILE");
}

}  // namespace
}  // namespace ttnn::graph::arguments::test
