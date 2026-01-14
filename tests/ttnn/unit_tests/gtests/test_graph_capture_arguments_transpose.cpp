// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <nlohmann/json.hpp>
#include "ttnn/tensor/tensor_ops.hpp"
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

    // operations[0]: ttnn::transpose
    const auto& operation0 = operations[0];
    EXPECT_EQ(operation0.operation_name, "ttnn::transpose");
    EXPECT_EQ(operation0.arguments.size(), 3);
    EXPECT_EQ(
        operation0.arguments[0],
        "Tensor(storage=DeviceStorage(),tensor_spec=TensorSpec(logical_shape=Shape([1, 1, 2048, "
        "512]),tensor_layout=TensorLayout(dtype=DataType::BFLOAT16,page_config=PageConfig(config=RowMajorPageConfig("
        "tile=Tile(tile_shape={32, 32},face_shape={16, "
        "16},num_faces=4))),memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type="
        "BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0),alignment="
        "Alignment([1]))))");
    EXPECT_EQ(operation0.arguments[1], "1");
    EXPECT_EQ(operation0.arguments[2], "2");

    // operations[2]: tt::tt_metal::create_device_tensor
    const auto& operation2 = operations[2];
    EXPECT_EQ(operation2.operation_name, "tt::tt_metal::create_device_tensor");
    EXPECT_EQ(operation2.arguments.size(), 5);
    EXPECT_EQ(operation2.arguments[0], "Shape([1, 2048, 1, 512])");
    EXPECT_EQ(operation2.arguments[1], "DataType::BFLOAT16");
    EXPECT_EQ(operation2.arguments[2], "Layout::ROW_MAJOR");
    EXPECT_EQ(operation2.arguments[0], "Shape([1, 2048, 1, 512])");
    EXPECT_EQ(operation2.arguments[1], "DataType::BFLOAT16");
}

}  // namespace
}  // namespace ttnn::graph::arguments::test
