// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "ttnn_test_fixtures.hpp"
#include "ttnn/device.hpp"
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_consts.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include <optional>
#include <string>

namespace ttnn::graph::arguments::test {

class TestGraphCaptureArgumentsTranspose : public TTNNFixtureWithTensor {
protected:
    tt::tt_metal::IGraphProcessor::RunMode Mode = tt::tt_metal::IGraphProcessor::RunMode::NORMAL;
};

TEST_P(TestGraphCaptureArgumentsTranspose, Transpose) {
    auto tt_input = CreateTensor();
    tt_input.reshape(ttnn::Shape{1, 2048, 4, 128});
    ttnn::graph::GraphProcessor::begin_graph_capture(Mode);
    ttnn::operations::data_movement::ExecuteTranspose::invoke(tt_input, 1, 2);
    auto trace = ttnn::graph::GraphProcessor::end_graph_capture();
    auto operations = ttnn::graph::extract_arguments(trace);

    EXPECT_EQ(operations[0].size(), 5);
    EXPECT_EQ(
        operations[0][0],
        "Tensor(storage=DeviceStorage(memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_"
        "type=BufferType::L1,shard_spec=std::nullopt)),tensor_spec=TensorSpec(logical_shape=Shape([1, 1, 2048, "
        "512]),tensor_layout=TensorLayout(dtype=BFLOAT16,page_config=PageConfig(config=RowMajorPageConfig(tile=Tile("
        "tile_shape={32, 32},face_shape={16, "
        "16},num_faces=4))),memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type="
        "BufferType::L1,shard_spec=std::nullopt),alignment=Alignment([1]))))");
    EXPECT_EQ(operations[0][1], "SmallVector([0, 2, 1, 3])");
    EXPECT_EQ(
        operations[0][2],
        "MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::"
        "nullopt)");
    EXPECT_EQ(operations[0][3], "[ unsupported type , std::__1::reference_wrapper<std::__1::nullopt_t const>]");
    EXPECT_EQ(operations[0][4], "0");

    EXPECT_EQ(operations[1].size(), 2);
    EXPECT_EQ(
        operations[1][0],
        "[ unsupported type , "
        "std::__1::reference_wrapper<ttnn::operations::data_movement::PermuteDeviceOperation::operation_attributes_t "
        "const>]");
    EXPECT_EQ(
        operations[1][1],
        "[ unsupported type , "
        "std::__1::reference_wrapper<ttnn::operations::data_movement::PermuteDeviceOperation::tensor_args_t const>]");

    EXPECT_EQ(operations[2].size(), 5);
    EXPECT_EQ(operations[2][0], "Shape([1, 2048, 1, 512])");
    EXPECT_EQ(operations[2][1], "BFLOAT16");
    EXPECT_EQ(operations[2][2], "Row Major");
    EXPECT_EQ(operations[2][3], "[ unsupported type , std::__1::reference_wrapper<tt::tt_metal::v0::IDevice*>]");
    EXPECT_EQ(
        operations[2][4],
        "MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::"
        "nullopt)");
}

INSTANTIATE_TEST_SUITE_P(
    TestGraphCaptureArgumentsTranspose_Transpose,
    TestGraphCaptureArgumentsTranspose,
    ::testing::Values(CreateTensorParameters{
        .input_shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 2048, 512}),
        .dtype = DataType::BFLOAT16,
        .layout = ROW_MAJOR_LAYOUT,
        .mem_cfg = L1_MEMORY_CONFIG}));
}  // namespace ttnn::graph::arguments::test
