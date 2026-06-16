// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <nlohmann/json.hpp>
#include "ttnn/tensor/tensor_ops.hpp"
#include <algorithm>
#include <map>
#include <vector>

#include <tt-metalium/graph_tracking.hpp>
#include "gtest/gtest.h"
#include <tt-metalium/shape.hpp>
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

TensorSpec make_nd_sharded_tensor_spec(
    const ttnn::Shape& shape, const ttnn::Shape& shard_shape, tt::tt_metal::BufferType buffer_type) {
    const auto cores = tt::tt_metal::CoreRangeSet(
        tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord{0, 0}, tt::tt_metal::CoreCoord{0, 0}));
    const auto memory_config = tt::tt_metal::MemoryConfig(
        buffer_type,
        tt::tt_metal::NdShardSpec{shard_shape, cores, tt::tt_metal::ShardOrientation::ROW_MAJOR});
    return TensorSpec(
        shape,
        TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            PageConfig(tt::tt_metal::Layout::TILE),
            memory_config));
}

const auto has_nd_provenance = [](const std::string& args) {
    return args.find("created_with_nd_shard_spec=1") != std::string::npos &&
           args.find("nd_shard_spec=std::optional") != std::string::npos;
};

TEST_F(TestGraphCaptureArgumentsTranspose, Transpose) {
    TensorSpec tensor_spec(
        ttnn::Shape({1, 1, 2048, 512}),
        TensorLayout(tt::tt_metal::DataType::BFLOAT16, PageConfig(tt::tt_metal::Layout::ROW_MAJOR), L1_MEMORY_CONFIG));
    auto tt_input = create_device_tensor(tensor_spec, device_);

    ttnn::graph::GraphProcessor::begin_graph_capture(tt::tt_metal::IGraphProcessor::RunMode::NORMAL);
    ttnn::transpose(tt_input, 1, 2);
    auto trace = ttnn::graph::GraphProcessor::end_graph_capture();
    auto operations = ttnn::graph::extract_arguments(trace);

    // operations[0]: PermuteDeviceOperation (device operation)
    const auto& operation0 = operations[0];
    EXPECT_EQ(operation0.operation_name, "PermuteDeviceOperation");
    EXPECT_EQ(operation0.arguments.size(), 2);

    // arguments[0]: operation_attributes_t with permutation, memory config, padding value
    EXPECT_TRUE(operation0.arguments[0].find("SmallVector([0, 2, 1, 3])") != std::string::npos);
    EXPECT_TRUE(operation0.arguments[0].find("MemoryConfig(") != std::string::npos);
    EXPECT_TRUE(operation0.arguments[0].find("TensorMemoryLayout::INTERLEAVED") != std::string::npos);
    EXPECT_TRUE(operation0.arguments[0].find("BufferType::L1") != std::string::npos);

    // arguments[1]: vector of input tensors with full tensor info
    EXPECT_TRUE(operation0.arguments[1].find("Tensor(") != std::string::npos);
    EXPECT_TRUE(operation0.arguments[1].find("Shape([1, 1, 2048, 512])") != std::string::npos);
    EXPECT_TRUE(operation0.arguments[1].find("DataType::BFLOAT16") != std::string::npos);
    EXPECT_TRUE(operation0.arguments[1].find("RowMajorPageConfig") != std::string::npos);
    EXPECT_TRUE(operation0.arguments[1].find("DeviceStorage()") != std::string::npos);

    // Find tt::tt_metal::create_device_tensor operation (output tensor creation)
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

TEST_F(TestGraphCaptureArgumentsTranspose, TransposeImplicitOutputConfigPreservesNdProvenanceForNonNativeShardedFallback) {
    auto tt_input = create_device_tensor(
        make_nd_sharded_tensor_spec(
            ttnn::Shape({1, 1, 32, 64}),
            ttnn::Shape({1, 1, 32, 64}),
            tt::tt_metal::BufferType::DRAM),
        device_);

    ttnn::graph::GraphProcessor::begin_graph_capture(tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH);
    ttnn::transpose(tt_input, 1, 2);
    auto trace = ttnn::graph::GraphProcessor::end_graph_capture();
    auto operations = ttnn::graph::extract_arguments(trace);

    const auto transpose_it = std::find_if(operations.begin(), operations.end(), [](const auto& op) {
        return op.operation_name == "TransposeDeviceOperation";
    });
    ASSERT_NE(transpose_it, operations.end()) << "TransposeDeviceOperation not found";
    EXPECT_TRUE(has_nd_provenance(transpose_it->arguments[0])) << transpose_it->arguments[0];

    auto create_tensor_it = std::find_if(operations.begin(), operations.end(), [](const auto& op) {
        return op.operation_name == "tt::tt_metal::create_device_tensor";
    });
    ASSERT_NE(create_tensor_it, operations.end()) << "create_device_tensor operation not found";
    EXPECT_EQ(create_tensor_it->arguments[0], "Shape([1, 32, 1, 64])");
    EXPECT_EQ(create_tensor_it->arguments[1], "DataType::BFLOAT16");
    EXPECT_EQ(create_tensor_it->arguments[2], "Layout::TILE");
}

}  // namespace
}  // namespace ttnn::graph::arguments::test
