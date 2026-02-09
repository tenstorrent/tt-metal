// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <nlohmann/json.hpp>
#include "ttnn/tensor/tensor_ops.hpp"
#include <algorithm>
#include <map>
#include <vector>

#include <tt-metalium/graph_tracking.hpp>
#include "gtest/gtest.h"
#include <tt-metalium/shape.hpp>
#include "ttnn/decorators.hpp"
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::graph::arguments::test {
namespace {

using TestGraphCaptureArgumentsTranspose = TTNNFixtureWithDevice;

TEST_F(TestGraphCaptureArgumentsTranspose, Transpose) {
    TensorSpec tensor_spec(
        ttnn::Shape({1, 1, 2048, 512}),
        TensorLayout(tt::tt_metal::DataType::BFLOAT16, PageConfig(tt::tt_metal::Layout::ROW_MAJOR), L1_MEMORY_CONFIG));
    auto tt_input = create_device_tensor(tensor_spec, device_);

    ttnn::graph::GraphProcessor::begin_graph_capture(tt::tt_metal::IGraphProcessor::RunMode::NORMAL);
    ttnn::transpose(tt_input, 1, 2);
    auto trace = ttnn::graph::GraphProcessor::end_graph_capture();
    auto operations = ttnn::graph::extract_arguments(trace);

    // High-level C++ functions (like ttnn::transpose) are not traced.
    // Only device operations and low-level operations are captured.

    // operations[0]: PermuteDeviceOperation (device operation)
    const auto& operation0 = operations[0];
    EXPECT_EQ(operation0.operation_name, "PermuteDeviceOperation");
    EXPECT_EQ(operation0.arguments.size(), 2);
    EXPECT_EQ(
        operation0.arguments[0],
        "[ unsupported type , "
        "std::reference_wrapper<ttnn::operations::data_movement::PermuteDeviceOperation::operation_attributes_t "
        "const>]");
    EXPECT_EQ(
        operation0.arguments[1],
        "[ unsupported type , std::reference_wrapper<std::vector<std::reference_wrapper<tt::tt_metal::Tensor const>, "
        "std::allocator<std::reference_wrapper<tt::tt_metal::Tensor const> > > >]");

    // Find tt::tt_metal::create_device_tensor operation (output tensor creation)
    // Note: The index may vary, so we search by name
    auto it = std::find_if(operations.begin(), operations.end(), [](const auto& op) {
        return op.operation_name == "tt::tt_metal::create_device_tensor";
    });
    ASSERT_NE(it, operations.end()) << "create_device_tensor operation not found";
    const auto& create_tensor_op = *it;
    EXPECT_EQ(create_tensor_op.arguments.size(), 5);
    EXPECT_EQ(create_tensor_op.arguments[0], "Shape([1, 2048, 1, 512])");
    EXPECT_EQ(create_tensor_op.arguments[1], "DataType::BFLOAT16");
    EXPECT_EQ(create_tensor_op.arguments[2], "Layout::ROW_MAJOR");
}

}  // namespace
}  // namespace ttnn::graph::arguments::test
