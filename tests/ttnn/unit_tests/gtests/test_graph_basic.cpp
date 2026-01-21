// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <boost/move/utility_core.hpp>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <tt-metalium/graph_tracking.hpp>
#include "gtest/gtest.h"
#include <tt-metalium/shape.hpp>
#include "ttnn/decorators.hpp"
#include "ttnn/graph/graph_consts.hpp"
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"
#include "ttnn/operations/eltwise/ternary/ternary.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace tt::tt_metal {
class IDevice;
}  // namespace tt::tt_metal

namespace ttnn::graph::test {

struct BufferTestParam {
    ttnn::Shape shape_a;
    ttnn::Shape shape_b;
};

class BufferTestFixture
    : public TTNNFixtureWithSuiteDevice<BufferTestFixture>,
      public testing::WithParamInterface<std::tuple<BufferTestParam, tt::tt_metal::IGraphProcessor::RunMode>> {};

TEST_P(BufferTestFixture, BufferTest) {
    auto param_combination = GetParam();
    auto params = std::get<0>(param_combination);
    auto run_mode = std::get<1>(param_combination);

    tt::tt_metal::IDevice* device = device_;
    {
        ttnn::graph::GraphProcessor::begin_graph_capture(run_mode);
        {
            const auto input_a = ttnn::TensorSpec(
                params.shape_a,
                tt::tt_metal::TensorLayout(
                    tt::tt_metal::DataType::BFLOAT16,
                    tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                    ttnn::L1_MEMORY_CONFIG));
            const auto input_tensor_a = tt::tt_metal::create_device_tensor(input_a, device);
        }
        {
            const auto input_a = ttnn::TensorSpec(
                params.shape_a,
                tt::tt_metal::TensorLayout(
                    tt::tt_metal::DataType::BFLOAT16,
                    tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                    ttnn::L1_MEMORY_CONFIG));
            const auto input_tensor_a = tt::tt_metal::create_device_tensor(input_a, device);

            const auto input_b = ttnn::TensorSpec(
                params.shape_b,
                tt::tt_metal::TensorLayout(
                    tt::tt_metal::DataType::BFLOAT16,
                    tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                    ttnn::L1_MEMORY_CONFIG));

            const auto input_tensor_b = tt::tt_metal::create_device_tensor(input_b, device);
        }
        auto trace = ttnn::graph::GraphProcessor::end_graph_capture();

        auto find_nodes_by_type = [](const auto& trace, const std::string& type) {
            std::vector<nlohmann::json> nodes;
            for (const auto& node : trace) {
                if (node.at(kNodeType) == type) {
                    nodes.push_back(node);
                }
            }
            return nodes;
        };

        // Check if there are two buffer_allocate_nodes, and if each is connected to only one different buffer
        auto buffer_allocate_nodes = find_nodes_by_type(trace, kNodeBufferAllocate);
        EXPECT_EQ(buffer_allocate_nodes.size(), 3);
        for (const auto& node : buffer_allocate_nodes) {
            EXPECT_EQ(node.at(kConnections).size(), 1);
        }
        auto connection_a = buffer_allocate_nodes[0].at(kConnections)[0].get<int>();
        auto connection_a2 = buffer_allocate_nodes[1].at(kConnections)[0].get<int>();
        auto connection_c = buffer_allocate_nodes[2].at(kConnections)[0].get<int>();
        EXPECT_NE(connection_a, connection_a2);
        EXPECT_NE(connection_a, connection_c);
        EXPECT_NE(connection_a2, connection_c);

        // Check if there are two buffer nodes and they have correct sizes
        auto buffer_nodes = find_nodes_by_type(trace, kNodeBuffer);
        EXPECT_EQ(buffer_nodes.size(), 3);
        auto size_a = std::stoi(buffer_nodes[0].at(kParams).at(kSize).get<std::string>());
        EXPECT_EQ(params.shape_a.volume() * 2, size_a);
        auto size_a2 = std::stoi(buffer_nodes[1].at(kParams).at(kSize).get<std::string>());
        EXPECT_EQ(params.shape_a.volume() * 2, size_a2);
        auto size_b = std::stoi(buffer_nodes[2].at(kParams).at(kSize).get<std::string>());
        EXPECT_EQ(params.shape_b.volume() * 2, size_b);

        // Print the trace for reference
        std::cout << trace << std::endl;
    }
}

INSTANTIATE_TEST_SUITE_P(
    BufferTest,
    BufferTestFixture,
    ::testing::Combine(
        ::testing::Values(BufferTestParam{
            .shape_a = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 32, 32}),
            .shape_b = ttnn::Shape(tt::tt_metal::Array4D{32, 1, 32, 32})}),
        ::testing::Values(
            tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH, tt::tt_metal::IGraphProcessor::RunMode::NORMAL)),
    [](const testing::TestParamInfo<std::tuple<BufferTestParam, tt::tt_metal::IGraphProcessor::RunMode>>& info) {
        std::stringstream ss;

        static uint32_t uid = 0;
        ss << uid++;

        const auto& run_mode = std::get<1>(info.param);
        switch (run_mode) {
            case tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH: ss << "_NO_DISPATCH"; break;
            case tt::tt_metal::IGraphProcessor::RunMode::NORMAL: ss << "_NORMAL"; break;
            default: break;
        }
        return ss.str();
    });
}  // namespace ttnn::graph::test

class TestScopedGraphCapture : public ttnn::TTNNFixtureWithDevice {};
TEST_F(TestScopedGraphCapture, ScopedGraphCapture) {
    tt::tt_metal::IDevice* device = device_;

    auto operation = [&device](tt::tt_metal::DataType datatype) {
        const auto input_a = ttnn::TensorSpec(
            ttnn::Shape(tt::tt_metal::Array4D{1, 4, 512, 512}),
            tt::tt_metal::TensorLayout(
                datatype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), ttnn::L1_MEMORY_CONFIG));
        const auto input_tensor_a = tt::tt_metal::create_device_tensor(input_a, device);
        const auto output_tensor = ttnn::softmax(input_tensor_a, -1);
    };

    // build reference
    std::vector<std::string> ref_calltrace;
    nlohmann::json ref_json_trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        operation(tt::tt_metal::DataType::BFLOAT16);
        ref_json_trace = capture.end_graph_capture();
        ref_calltrace = ttnn::graph::extract_calltrace(ref_json_trace);
    }
    for (const auto& call : ref_calltrace) {
        std::cout << call << std::endl;
    }

    // with manual exception in the nested loop
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        try {
            auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
            operation(tt::tt_metal::DataType::BFLOAT16);
            throw std::runtime_error("Expected");
        } catch (const std::exception& e) {
            EXPECT_EQ(std::string(e.what()), "Expected");
        }
        auto json_trace = capture.end_graph_capture();
        EXPECT_EQ(ttnn::graph::extract_calltrace(json_trace), ref_calltrace);
    }

    // with exception in the operation #2
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        try {
            auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
            operation(tt::tt_metal::DataType::UINT8);  // fails in the softmax::validate (not supported data type)
        } catch (const std::exception& e) {
            EXPECT_TRUE(std::string(e.what()).find("FATAL") != std::string::npos);
        }
        auto json_trace = capture.end_graph_capture();

        EXPECT_EQ(
            ttnn::graph::extract_calltrace(json_trace),
            std::vector<std::string>(
                {"tt::tt_metal::create_device_tensor",
                 "ttnn::softmax",
                 "SoftmaxDeviceOperation",
                 "tt::tt_metal::create_device_tensor"}));
    }

    // check original again to ensure it's not affected by the thrown exceptions
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        operation(tt::tt_metal::DataType::BFLOAT16);
        auto json_trace = capture.end_graph_capture();
        // std::cout << json_trace.dump(4);
        EXPECT_EQ(ttnn::graph::extract_calltrace(json_trace), ref_calltrace);

        EXPECT_EQ(json_trace.size(), ref_json_trace.size());
        // tensor ids can be different, therfore checking if general structure is the same
        for (size_t i = 0; i < json_trace.size(); i++) {
            const auto& v = json_trace[i];
            const auto& ref_v = ref_json_trace[i];
            EXPECT_EQ(v[ttnn::graph::kCounter], ref_v[ttnn::graph::kCounter]);
            EXPECT_EQ(v[ttnn::graph::kConnections], ref_v[ttnn::graph::kConnections]);
            EXPECT_EQ(v[ttnn::graph::kNodeType], ref_v[ttnn::graph::kNodeType]);
        }
    }
}

TEST_F(TestScopedGraphCapture, OrderOfArgsTest) {
    tt::tt_metal::IDevice* device = device_;

    const auto tensor_spec = ttnn::TensorSpec(
        ttnn::Shape(tt::tt_metal::Array2D{32, 64}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            ttnn::L1_MEMORY_CONFIG));
    const auto tensor_a = tt::tt_metal::create_device_tensor(tensor_spec, device);
    const auto tensor_b = tt::tt_metal::create_device_tensor(tensor_spec, device);
    auto operation = [](const auto& a, const auto& b) {
        const auto output_tensor_1 = ttnn::subtract(a, b);
        const auto output_tensor_2 = ttnn::subtract(b, a);
    };

    // build reference
    nlohmann::json ref_json_trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        operation(tensor_a, tensor_b);
        ref_json_trace = capture.end_graph_capture();
    }

    // Verify that input_tensors field exists and captures the correct order
    // Find the two subtract operations
    std::vector<nlohmann::json> subtract_ops;
    for (const auto& node : ref_json_trace) {
        if (node["node_type"] == "function_start" && node["params"]["name"] == "ttnn::subtract") {
            subtract_ops.push_back(node);
        }
    }

    ASSERT_EQ(subtract_ops.size(), 2);

    // Both operations should have input_tensors field
    ASSERT_TRUE(subtract_ops[0].contains("input_tensors"));
    ASSERT_TRUE(subtract_ops[1].contains("input_tensors"));

    // Get the tensor counters for a and b
    int tensor_a_counter = -1, tensor_b_counter = -1;
    for (const auto& node : ref_json_trace) {
        if (node["node_type"] == "tensor" && node["params"].contains("tensor_id")) {
            if (tensor_a_counter == -1) {
                tensor_a_counter = node["counter"];
            } else if (tensor_b_counter == -1) {
                tensor_b_counter = node["counter"];
                break;
            }
        }
    }
    ASSERT_NE(tensor_a_counter, -1);
    ASSERT_NE(tensor_b_counter, -1);

    // First subtract(a, b) should have input_tensors = [tensor_a_counter, tensor_b_counter]
    auto first_input_tensors = subtract_ops[0]["input_tensors"];
    ASSERT_EQ(first_input_tensors.size(), 2);
    EXPECT_EQ(first_input_tensors[0].get<int>(), tensor_a_counter);
    EXPECT_EQ(first_input_tensors[1].get<int>(), tensor_b_counter);

    // Second subtract(b, a) should have input_tensors = [tensor_b_counter, tensor_a_counter]
    auto second_input_tensors = subtract_ops[1]["input_tensors"];
    ASSERT_EQ(second_input_tensors.size(), 2);
    EXPECT_EQ(second_input_tensors[0].get<int>(), tensor_b_counter);
    EXPECT_EQ(second_input_tensors[1].get<int>(), tensor_a_counter);

    // Verify that the orders are different
    EXPECT_NE(first_input_tensors, second_input_tensors);
}

TEST_F(TestScopedGraphCapture, SameTensorMultipleTimesTest) {
    tt::tt_metal::IDevice* device = device_;

    const auto tensor_spec = ttnn::TensorSpec(
        ttnn::Shape(tt::tt_metal::Array2D{32, 64}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            ttnn::L1_MEMORY_CONFIG));
    const auto tensor_a = tt::tt_metal::create_device_tensor(tensor_spec, device);

    nlohmann::json trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        // Test add(a, a) - same tensor used twice
        const auto output = ttnn::add(tensor_a, tensor_a, std::nullopt, std::nullopt);
        trace = capture.end_graph_capture();
    }

    // Find the add operation
    nlohmann::json add_op;
    for (const auto& node : trace) {
        if (node["node_type"] == "function_start" &&
            node["params"]["name"].get<std::string>().find("add") != std::string::npos) {
            add_op = node;
            break;
        }
    }

    ASSERT_FALSE(add_op.is_null());
    ASSERT_TRUE(add_op.contains("input_tensors"));

    auto input_tensors = add_op["input_tensors"];
    ASSERT_EQ(input_tensors.size(), 2);

    // Both inputs should refer to the same tensor counter
    EXPECT_EQ(input_tensors[0].get<int>(), input_tensors[1].get<int>());
}

TEST_F(TestScopedGraphCapture, TernaryOpDifferentOrderTest) {
    tt::tt_metal::IDevice* device = device_;

    const auto tensor_spec = ttnn::TensorSpec(
        ttnn::Shape(tt::tt_metal::Array2D{32, 64}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            ttnn::L1_MEMORY_CONFIG));
    const auto tensor_a = tt::tt_metal::create_device_tensor(tensor_spec, device);
    const auto tensor_b = tt::tt_metal::create_device_tensor(tensor_spec, device);
    const auto tensor_c = tt::tt_metal::create_device_tensor(tensor_spec, device);

    nlohmann::json trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        const auto output1 = ttnn::addcmul(tensor_a, tensor_b, tensor_c);
        const auto output2 = ttnn::addcmul(tensor_c, tensor_b, tensor_a);
        trace = capture.end_graph_capture();
    }

    // Find the addcmul operations
    std::vector<nlohmann::json> addcmul_ops;
    for (const auto& node : trace) {
        if (node["node_type"] == "function_start" &&
            node["params"]["name"].get<std::string>().find("addcmul") != std::string::npos) {
            addcmul_ops.push_back(node);
        }
    }

    ASSERT_EQ(addcmul_ops.size(), 2);
    ASSERT_TRUE(addcmul_ops[0].contains("input_tensors"));
    ASSERT_TRUE(addcmul_ops[1].contains("input_tensors"));

    // Get tensor counters
    int tensor_a_counter = -1, tensor_b_counter = -1, tensor_c_counter = -1;
    for (const auto& node : trace) {
        if (node["node_type"] == "tensor" && node["params"].contains("tensor_id")) {
            if (tensor_a_counter == -1) {
                tensor_a_counter = node["counter"];
            } else if (tensor_b_counter == -1) {
                tensor_b_counter = node["counter"];
            } else if (tensor_c_counter == -1) {
                tensor_c_counter = node["counter"];
                break;
            }
        }
    }

    ASSERT_NE(tensor_a_counter, -1);
    ASSERT_NE(tensor_b_counter, -1);
    ASSERT_NE(tensor_c_counter, -1);

    // First addcmul(a, b, c) should have input_tensors = [a, b, c]
    auto first_input_tensors = addcmul_ops[0]["input_tensors"];
    ASSERT_EQ(first_input_tensors.size(), 3);
    EXPECT_EQ(first_input_tensors[0].get<int>(), tensor_a_counter);
    EXPECT_EQ(first_input_tensors[1].get<int>(), tensor_b_counter);
    EXPECT_EQ(first_input_tensors[2].get<int>(), tensor_c_counter);

    // Second addcmul(c, b, a) should have input_tensors = [c, b, a]
    auto second_input_tensors = addcmul_ops[1]["input_tensors"];
    ASSERT_EQ(second_input_tensors.size(), 3);
    EXPECT_EQ(second_input_tensors[0].get<int>(), tensor_c_counter);
    EXPECT_EQ(second_input_tensors[1].get<int>(), tensor_b_counter);
    EXPECT_EQ(second_input_tensors[2].get<int>(), tensor_a_counter);

    EXPECT_NE(first_input_tensors, second_input_tensors);
}

TEST_F(TestScopedGraphCapture, TernaryOpRepeatedTensorsTest) {
    tt::tt_metal::IDevice* device = device_;

    const auto tensor_spec = ttnn::TensorSpec(
        ttnn::Shape(tt::tt_metal::Array2D{32, 64}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            ttnn::L1_MEMORY_CONFIG));
    const auto tensor_a = tt::tt_metal::create_device_tensor(tensor_spec, device);
    const auto tensor_b = tt::tt_metal::create_device_tensor(tensor_spec, device);

    nlohmann::json trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        const auto output1 = ttnn::addcmul(tensor_a, tensor_b, tensor_a);
        const auto output2 = ttnn::addcmul(tensor_b, tensor_a, tensor_a);
        trace = capture.end_graph_capture();
    }

    // Find the addcmul operations
    std::vector<nlohmann::json> addcmul_ops;
    for (const auto& node : trace) {
        if (node["node_type"] == "function_start" &&
            node["params"]["name"].get<std::string>().find("addcmul") != std::string::npos) {
            addcmul_ops.push_back(node);
        }
    }

    ASSERT_EQ(addcmul_ops.size(), 2);

    // Get tensor counters
    int tensor_a_counter = -1, tensor_b_counter = -1;
    for (const auto& node : trace) {
        if (node["node_type"] == "tensor" && node["params"].contains("tensor_id")) {
            if (tensor_a_counter == -1) {
                tensor_a_counter = node["counter"];
            } else if (tensor_b_counter == -1) {
                tensor_b_counter = node["counter"];
                break;
            }
        }
    }

    ASSERT_NE(tensor_a_counter, -1);
    ASSERT_NE(tensor_b_counter, -1);

    // First addcmul(a, b, a) should have input_tensors = [a, b, a]
    auto first_input_tensors = addcmul_ops[0]["input_tensors"];
    ASSERT_EQ(first_input_tensors.size(), 3);
    EXPECT_EQ(first_input_tensors[0].get<int>(), tensor_a_counter);
    EXPECT_EQ(first_input_tensors[1].get<int>(), tensor_b_counter);
    EXPECT_EQ(first_input_tensors[2].get<int>(), tensor_a_counter);

    // Second addcmul(b, a, a) should have input_tensors = [b, a, a]
    auto second_input_tensors = addcmul_ops[1]["input_tensors"];
    ASSERT_EQ(second_input_tensors.size(), 3);
    EXPECT_EQ(second_input_tensors[0].get<int>(), tensor_b_counter);
    EXPECT_EQ(second_input_tensors[1].get<int>(), tensor_a_counter);
    EXPECT_EQ(second_input_tensors[2].get<int>(), tensor_a_counter);

    EXPECT_NE(first_input_tensors, second_input_tensors);
}

TEST_F(TestScopedGraphCapture, MatmulDifferentOrdersTest) {
    tt::tt_metal::IDevice* device = device_;

    const auto tensor_spec = ttnn::TensorSpec(
        ttnn::Shape(tt::tt_metal::Array2D{64, 64}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            ttnn::L1_MEMORY_CONFIG));
    const auto tensor_a = tt::tt_metal::create_device_tensor(tensor_spec, device);
    const auto tensor_b = tt::tt_metal::create_device_tensor(tensor_spec, device);

    nlohmann::json trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        const auto output1 = ttnn::matmul(tensor_a, tensor_b);
        const auto output2 = ttnn::matmul(tensor_b, tensor_a);
        const auto output3 = ttnn::matmul(tensor_a, tensor_a);
        trace = capture.end_graph_capture();
    }

    // Find the matmul operations
    std::vector<nlohmann::json> matmul_ops;
    for (const auto& node : trace) {
        if (node["node_type"] == "function_start" &&
            node["params"]["name"].get<std::string>().find("ttnn::matmul") != std::string::npos) {
            matmul_ops.push_back(node);
        }
    }

    ASSERT_EQ(matmul_ops.size(), 3);
    ASSERT_TRUE(matmul_ops[0].contains("input_tensors"));
    ASSERT_TRUE(matmul_ops[1].contains("input_tensors"));
    ASSERT_TRUE(matmul_ops[2].contains("input_tensors"));

    // Get tensor counters
    int tensor_a_counter = -1, tensor_b_counter = -1;
    for (const auto& node : trace) {
        if (node["node_type"] == "tensor" && node["params"].contains("tensor_id")) {
            if (tensor_a_counter == -1) {
                tensor_a_counter = node["counter"];
            } else if (tensor_b_counter == -1) {
                tensor_b_counter = node["counter"];
                break;
            }
        }
    }

    ASSERT_NE(tensor_a_counter, -1);
    ASSERT_NE(tensor_b_counter, -1);

    // First matmul(a, b) should have input_tensors = [a, b]
    auto first_input_tensors = matmul_ops[0]["input_tensors"];
    ASSERT_EQ(first_input_tensors.size(), 2);
    EXPECT_EQ(first_input_tensors[0].get<int>(), tensor_a_counter);
    EXPECT_EQ(first_input_tensors[1].get<int>(), tensor_b_counter);

    // Second matmul(b, a) should have input_tensors = [b, a]
    auto second_input_tensors = matmul_ops[1]["input_tensors"];
    ASSERT_EQ(second_input_tensors.size(), 2);
    EXPECT_EQ(second_input_tensors[0].get<int>(), tensor_b_counter);
    EXPECT_EQ(second_input_tensors[1].get<int>(), tensor_a_counter);

    // Third matmul(a, a) should have input_tensors = [a, a]
    auto third_input_tensors = matmul_ops[2]["input_tensors"];
    ASSERT_EQ(third_input_tensors.size(), 2);
    EXPECT_EQ(third_input_tensors[0].get<int>(), tensor_a_counter);
    EXPECT_EQ(third_input_tensors[1].get<int>(), tensor_a_counter);

    // Verify all three have different patterns
    EXPECT_NE(first_input_tensors, second_input_tensors);
    EXPECT_NE(first_input_tensors, third_input_tensors);
    EXPECT_NE(second_input_tensors, third_input_tensors);
}

TEST_F(TestScopedGraphCapture, SubtractArgumentOrderWithCapturedTensorsTest) {
    // This test verifies that when tensors are created within the capture,
    // the graph correctly tracks the argument order for non-commutative operations like subtract.
    // We test subtract(a, b) vs subtract(b, a) to ensure the order is preserved.

    tt::tt_metal::IDevice* device = device_;

    auto operation = [&device]() {
        // Create tensors INSIDE the capture
        const auto tensor_spec = ttnn::TensorSpec(
            ttnn::Shape(tt::tt_metal::Array2D{32, 64}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                ttnn::L1_MEMORY_CONFIG));

        const auto tensor_a = tt::tt_metal::create_device_tensor(tensor_spec, device);
        const auto tensor_b = tt::tt_metal::create_device_tensor(tensor_spec, device);

        // Perform subtract operations in different orders
        const auto output_tensor_1 = ttnn::subtract(tensor_a, tensor_b, std::nullopt, std::nullopt);
        const auto output_tensor_2 = ttnn::subtract(tensor_b, tensor_a, std::nullopt, std::nullopt);
    };

    nlohmann::json trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        operation();
        trace = capture.end_graph_capture();
    }

    // Find all subtract operations in the trace
    std::vector<nlohmann::json> subtract_ops;
    for (const auto& node : trace) {
        if (node["node_type"] == "function_start" &&
            node["params"]["name"].get<std::string>().find("subtract") != std::string::npos) {
            subtract_ops.push_back(node);
        }
    }

    ASSERT_EQ(subtract_ops.size(), 2);
    ASSERT_TRUE(subtract_ops[0].contains("input_tensors"));
    ASSERT_TRUE(subtract_ops[1].contains("input_tensors"));

    // Get tensor counters for tensor_a and tensor_b
    int tensor_a_counter = -1, tensor_b_counter = -1;
    for (const auto& node : trace) {
        if (node["node_type"] == "tensor" && node["params"].contains("tensor_id")) {
            if (tensor_a_counter == -1) {
                tensor_a_counter = node["counter"];
            } else if (tensor_b_counter == -1) {
                tensor_b_counter = node["counter"];
                break;
            }
        }
    }

    ASSERT_NE(tensor_a_counter, -1);
    ASSERT_NE(tensor_b_counter, -1);

    // First subtract(a, b) should have input_tensors = [a, b]
    auto first_input_tensors = subtract_ops[0]["input_tensors"];
    ASSERT_EQ(first_input_tensors.size(), 2);
    EXPECT_EQ(first_input_tensors[0].get<int>(), tensor_a_counter);
    EXPECT_EQ(first_input_tensors[1].get<int>(), tensor_b_counter);

    // Second subtract(b, a) should have input_tensors = [b, a]
    auto second_input_tensors = subtract_ops[1]["input_tensors"];
    ASSERT_EQ(second_input_tensors.size(), 2);
    EXPECT_EQ(second_input_tensors[0].get<int>(), tensor_b_counter);
    EXPECT_EQ(second_input_tensors[1].get<int>(), tensor_a_counter);

    // Verify the two subtract operations have different input patterns (argument order matters!)
    EXPECT_NE(first_input_tensors, second_input_tensors);

    // Verify that subtract(a, b) has reversed order compared to subtract(b, a)
    EXPECT_EQ(first_input_tensors[0], second_input_tensors[1]);
    EXPECT_EQ(first_input_tensors[1], second_input_tensors[0]);
}
