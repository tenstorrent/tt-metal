// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "ttnn_test_fixtures.hpp"
#include "ttnn/device.hpp"
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_consts.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"
#include "ttnn/operations/moreh/moreh_dot/moreh_dot.hpp"
#include <optional>
#include <string>

namespace ttnn::graph::arguments::test {

class TestGraphCaptureArgumentsMorehDot : public TTNNFixtureWithTensor {
protected:
    tt::tt_metal::IGraphProcessor::RunMode Mode = tt::tt_metal::IGraphProcessor::RunMode::NORMAL;
};

TEST_P(TestGraphCaptureArgumentsMorehDot, MorehDot) {
    auto tt_input1 = CreateTensor();
    auto tt_input2 = CreateTensor();
    ttnn::graph::GraphProcessor::begin_graph_capture(Mode);
    ttnn::operations::moreh::moreh_dot::MorehDot::invoke(
        tt_input1, tt_input2, std::nullopt, DataType::BFLOAT16, std::nullopt, std::nullopt);
    auto trace = ttnn::graph::GraphProcessor::end_graph_capture();
    auto operations = ttnn::graph::extract_arguments(trace);
    /*
        int i =0;
        for(auto operation : operations)
        {
            std::cout << "operation: " <<  i << std::endl;

            int j = 0;
            for (auto argument : operation)
            {
                std::cout << "argument[" <<  j  << "]" << std::endl;
                std::cout << argument << std::endl;
                j++;
            }

            i++;
        }
    */

    EXPECT_EQ(operations[0].size(), 6);
    EXPECT_EQ(
        operations[0][0],
        "Tensor(storage=DeviceStorage(memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_"
        "type=BufferType::L1,shard_spec=std::nullopt)),tensor_spec=TensorSpec(logical_shape=Shape([1, 1, 1, "
        "32]),tensor_layout=TensorLayout(dtype=BFLOAT16,page_config=PageConfig(config=TilePageConfig(tile=Tile(tile_"
        "shape={32, 32},face_shape={16, "
        "16},num_faces=4))),memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type="
        "BufferType::L1,shard_spec=std::nullopt),alignment=Alignment([32, 32]))))");
    EXPECT_EQ(
        operations[0][1],
        "Tensor(storage=DeviceStorage(memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_"
        "type=BufferType::L1,shard_spec=std::nullopt)),tensor_spec=TensorSpec(logical_shape=Shape([1, 1, 1, "
        "32]),tensor_layout=TensorLayout(dtype=BFLOAT16,page_config=PageConfig(config=TilePageConfig(tile=Tile(tile_"
        "shape={32, 32},face_shape={16, "
        "16},num_faces=4))),memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type="
        "BufferType::L1,shard_spec=std::nullopt),alignment=Alignment([32, 32]))))");
    EXPECT_EQ(operations[0][2], "nullopt");
    EXPECT_EQ(operations[0][3], "BFLOAT16");
    EXPECT_EQ(operations[0][4], "nullopt");
    EXPECT_EQ(
        operations[0][5],
        "[ unsupported type , "
        "std::__1::reference_wrapper<std::__1::optional<std::__1::variant<ttnn::GrayskullComputeKernelConfig, "
        "ttnn::WormholeComputeKernelConfig>> const>]");

    EXPECT_EQ(operations[1].size(), 2);
    EXPECT_EQ(
        operations[1][0],
        "[ unsupported type , "
        "std::__1::reference_wrapper<ttnn::operations::moreh::moreh_dot::MorehDotOperation::operation_attributes_t "
        "const>]");
    EXPECT_EQ(
        operations[1][1],
        "[ unsupported type , "
        "std::__1::reference_wrapper<ttnn::operations::moreh::moreh_dot::MorehDotOperation::tensor_args_t const>]");

    EXPECT_EQ(operations[2].size(), 5);
    EXPECT_EQ(operations[2][0], "Shape([1, 1, 1, 1])");
    EXPECT_EQ(operations[2][1], "BFLOAT16");
    EXPECT_EQ(operations[2][2], "Tile");
    EXPECT_EQ(operations[2][3], "[ unsupported type , std::__1::reference_wrapper<tt::tt_metal::v0::IDevice*>]");
    EXPECT_EQ(
        operations[2][4],
        "MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::"
        "nullopt)");
}

INSTANTIATE_TEST_SUITE_P(
    TestGraphCaptureArgumentsMorehDot_MorehDot,
    TestGraphCaptureArgumentsMorehDot,
    ::testing::Values(CreateTensorParameters{
        .input_shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 1, 32}),
        .dtype = DataType::BFLOAT16,
        .layout = TILE_LAYOUT,
        .mem_cfg = L1_MEMORY_CONFIG}));
}  // namespace ttnn::graph::arguments::test
