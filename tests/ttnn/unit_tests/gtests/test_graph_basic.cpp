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

        // Note: High-level function tracing (ttnn::softmax) was removed from decorators.hpp
        // Now only device operations are captured
        EXPECT_EQ(
            ttnn::graph::extract_calltrace(json_trace),
            std::vector<std::string>(
                {"tt::tt_metal::create_device_tensor",
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

    // Note: High-level function tracing (ttnn::subtract) was removed from decorators.hpp
    // Now only device operations are captured. This test is disabled as it tests a removed feature.
    // Find the two BinaryNgDeviceOperation operations (device-level operations for subtract)
    std::vector<nlohmann::json> subtract_ops;
    for (const auto& node : ref_json_trace) {
        if (node["node_type"] == "function_start" &&
            node["params"]["name"].get<std::string>().find("BinaryNgDeviceOperation") != std::string::npos) {
            subtract_ops.push_back(node);
        }
    }

    // Since high-level function tracing was removed, we can't test argument order at that level
    // The test is effectively disabled - device operations don't preserve the same argument order semantics
    GTEST_SKIP()
        << "High-level function tracing was removed - argument order testing at function level is no longer available";

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

    // Note: High-level function tracing (ttnn::add) was removed from decorators.hpp
    // This test is disabled as it tests a removed feature
    GTEST_SKIP()
        << "High-level function tracing was removed - argument order testing at function level is no longer available";

    // Find the add operation (code below is unreachable but kept for reference)
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
    // Note: High-level function tracing (ttnn::addcmul) was removed from decorators.hpp
    // This test is disabled as it tests a removed feature
    GTEST_SKIP()
        << "High-level function tracing was removed - argument order testing at function level is no longer available";

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

    // Find the addcmul operations (code below is unreachable but kept for reference)
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
    // Note: High-level function tracing (ttnn::addcmul) was removed from decorators.hpp
    // This test is disabled as it tests a removed feature
    GTEST_SKIP()
        << "High-level function tracing was removed - argument order testing at function level is no longer available";

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

    // Find the addcmul operations (code below is unreachable but kept for reference)
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
    // Note: High-level function tracing (ttnn::matmul) was removed from decorators.hpp
    // This test is disabled as it tests a removed feature
    GTEST_SKIP()
        << "High-level function tracing was removed - argument order testing at function level is no longer available";

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

    // Find the matmul operations (code below is unreachable but kept for reference)
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

    // Note: High-level function tracing (ttnn::subtract) was removed from decorators.hpp
    // This test is disabled as it tests a removed feature
    GTEST_SKIP()
        << "High-level function tracing was removed - argument order testing at function level is no longer available";

    // Find all subtract operations in the trace (code below is unreachable but kept for reference)
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

class DurationTrackingTest : public ttnn::TTNNFixtureWithDevice,
                             public testing::WithParamInterface<tt::tt_metal::IGraphProcessor::RunMode> {};

TEST_P(DurationTrackingTest, DurationTracking) {
    auto run_mode = GetParam();
    tt::tt_metal::IDevice* device = device_;

    nlohmann::json trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(run_mode);

        const auto tensor_spec = ttnn::TensorSpec(
            ttnn::Shape(tt::tt_metal::Array4D{1, 4, 512, 512}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                ttnn::L1_MEMORY_CONFIG));
        const auto input_tensor = tt::tt_metal::create_device_tensor(tensor_spec, device);
        const auto output_tensor = ttnn::softmax(input_tensor, -1);

        trace = capture.end_graph_capture();
    }

    // Find function_start and function_end nodes
    bool found_function_start = false;
    bool found_end_with_duration = false;
    bool found_capture_end_with_duration = false;

    for (const auto& node : trace) {
        if (node.at(ttnn::graph::kNodeType) == ttnn::graph::kNodeFunctionStart) {
            found_function_start = true;
        }
        if (node.at(ttnn::graph::kNodeType) == ttnn::graph::kNodeFunctionEnd) {
            if (node.contains(ttnn::graph::kDurationNs)) {
                auto duration = node.at(ttnn::graph::kDurationNs).get<uint64_t>();
                EXPECT_GE(duration, 0u);
                found_end_with_duration = true;
            }
        }
        if (node.at(ttnn::graph::kNodeType) == ttnn::graph::kNodeCaptureEnd) {
            if (node.contains(ttnn::graph::kDurationNs)) {
                auto total_duration = node.at(ttnn::graph::kDurationNs).get<uint64_t>();
                EXPECT_GE(total_duration, 0u);
                found_capture_end_with_duration = true;
            }
        }
    }

    EXPECT_TRUE(found_function_start);
    EXPECT_TRUE(found_end_with_duration);
    EXPECT_TRUE(found_capture_end_with_duration);

    ASSERT_FALSE(trace.empty());
    EXPECT_EQ(trace[0].at(ttnn::graph::kNodeType), ttnn::graph::kNodeCaptureStart);
}

INSTANTIATE_TEST_SUITE_P(
    DurationTracking,
    DurationTrackingTest,
    ::testing::Values(
        tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH, tt::tt_metal::IGraphProcessor::RunMode::NORMAL),
    [](const testing::TestParamInfo<tt::tt_metal::IGraphProcessor::RunMode>& info) {
        switch (info.param) {
            case tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH: return "NO_DISPATCH";
            case tt::tt_metal::IGraphProcessor::RunMode::NORMAL: return "NORMAL";
            default: return "UNKNOWN";
        }
    });

TEST_F(TestScopedGraphCapture, GetReportTest) {
    // Test get_current_report API
    tt::tt_metal::IDevice* device = device_;

    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);

        const auto tensor_spec = ttnn::TensorSpec(
            ttnn::Shape(tt::tt_metal::Array4D{1, 1, 32, 32}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                ttnn::L1_MEMORY_CONFIG));
        const auto input_tensor = tt::tt_metal::create_device_tensor(tensor_spec, device);

        // Get report while capture is still active
        auto report = capture.get_report();

        // Verify report structure
        ASSERT_TRUE(report.contains(ttnn::graph::kReportVersion));
        ASSERT_TRUE(report.contains(ttnn::graph::kReportGraph));
        ASSERT_TRUE(report.contains(ttnn::graph::kReportDevices));
        ASSERT_TRUE(report.contains(ttnn::graph::kReportMetadata));

        // Check version
        EXPECT_EQ(report.at(ttnn::graph::kReportVersion).get<int>(), ttnn::graph::kCurrentReportVersion);

        // Graph should have some nodes
        EXPECT_GT(report.at(ttnn::graph::kReportGraph).size(), 0u);

        // Metadata should have timestamp
        auto& metadata = report.at(ttnn::graph::kReportMetadata);
        ASSERT_TRUE(metadata.contains(ttnn::graph::kReportTimestampNs));
        EXPECT_GT(metadata.at(ttnn::graph::kReportTimestampNs).get<uint64_t>(), 0u);

        // End capture normally
        capture.end_graph_capture();
    }
}

// Test stack trace capture feature
class StackTraceTest : public ttnn::TTNNFixtureWithDevice,
                       public testing::WithParamInterface<tt::tt_metal::IGraphProcessor::RunMode> {};

TEST_P(StackTraceTest, StackTracesEnabledTest) {
    auto run_mode = GetParam();
    tt::tt_metal::IDevice* device = device_;

    // Enable stack traces before capture
    ttnn::graph::GraphProcessor::enable_stack_traces();
    ASSERT_TRUE(ttnn::graph::GraphProcessor::is_stack_trace_enabled());

    nlohmann::json trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(run_mode);

        const auto tensor_spec = ttnn::TensorSpec(
            ttnn::Shape(tt::tt_metal::Array4D{1, 1, 32, 32}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                ttnn::L1_MEMORY_CONFIG));
        const auto input_tensor = tt::tt_metal::create_device_tensor(tensor_spec, device);
        const auto output_tensor = ttnn::softmax(input_tensor, -1);

        trace = capture.end_graph_capture();
    }

    // Disable after capture
    ttnn::graph::GraphProcessor::disable_stack_traces();
    ASSERT_FALSE(ttnn::graph::GraphProcessor::is_stack_trace_enabled());

    // Find function_start nodes and verify they have stack traces
    bool found_stack_trace = false;
    for (const auto& node : trace) {
        if (node.at(ttnn::graph::kNodeType) == ttnn::graph::kNodeFunctionStart) {
            if (node.contains(ttnn::graph::kStackTrace)) {
                auto& stack_trace = node.at(ttnn::graph::kStackTrace);
                if (!stack_trace.empty()) {
                    found_stack_trace = true;
                    // Stack trace should be an array of strings
                    ASSERT_TRUE(stack_trace.is_array());
                    EXPECT_GT(stack_trace.size(), 0u);
                    // Each entry should be a string
                    for (const auto& entry : stack_trace) {
                        EXPECT_TRUE(entry.is_string());
                    }
                }
            }
        }
    }

    EXPECT_TRUE(found_stack_trace) << "Expected at least one function_start node with stack trace";
}

TEST_P(StackTraceTest, StackTracesDisabledTest) {
    auto run_mode = GetParam();
    tt::tt_metal::IDevice* device = device_;

    // Ensure stack traces are disabled
    ttnn::graph::GraphProcessor::disable_stack_traces();
    ASSERT_FALSE(ttnn::graph::GraphProcessor::is_stack_trace_enabled());

    nlohmann::json trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(run_mode);

        const auto tensor_spec = ttnn::TensorSpec(
            ttnn::Shape(tt::tt_metal::Array4D{1, 1, 32, 32}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                ttnn::L1_MEMORY_CONFIG));
        const auto input_tensor = tt::tt_metal::create_device_tensor(tensor_spec, device);

        trace = capture.end_graph_capture();
    }

    // No function_start node should have stack traces when disabled
    for (const auto& node : trace) {
        if (node.at(ttnn::graph::kNodeType) == ttnn::graph::kNodeFunctionStart) {
            if (node.contains(ttnn::graph::kStackTrace)) {
                auto& stack_trace = node.at(ttnn::graph::kStackTrace);
                EXPECT_TRUE(stack_trace.empty()) << "Stack trace should be empty when disabled";
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    StackTraceTest,
    StackTraceTest,
    ::testing::Values(
        tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH, tt::tt_metal::IGraphProcessor::RunMode::NORMAL),
    [](const testing::TestParamInfo<tt::tt_metal::IGraphProcessor::RunMode>& info) {
        switch (info.param) {
            case tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH: return "NO_DISPATCH";
            case tt::tt_metal::IGraphProcessor::RunMode::NORMAL: return "NORMAL";
            default: return "UNKNOWN";
        }
    });

// Test full tensor info capture (dtype, layout, memory_config, device_id, address, buffer_type)
class TensorInfoTest : public ttnn::TTNNFixtureWithDevice,
                       public testing::WithParamInterface<tt::tt_metal::IGraphProcessor::RunMode> {};

TEST_P(TensorInfoTest, FullTensorInfoCaptured) {
    auto run_mode = GetParam();
    tt::tt_metal::IDevice* device = device_;

    nlohmann::json trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(run_mode);

        const auto tensor_spec = ttnn::TensorSpec(
            ttnn::Shape(tt::tt_metal::Array4D{1, 1, 32, 32}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                ttnn::L1_MEMORY_CONFIG));
        const auto input_tensor = tt::tt_metal::create_device_tensor(tensor_spec, device);

        trace = capture.end_graph_capture();
    }

    // Find tensor nodes and verify they have full info
    bool found_tensor_with_full_info = false;
    for (const auto& node : trace) {
        if (node.at(ttnn::graph::kNodeType) == ttnn::graph::kNodeTensor) {
            auto& params = node.at(ttnn::graph::kParams);

            // Required fields
            ASSERT_TRUE(params.contains(ttnn::graph::kShape));
            ASSERT_TRUE(params.contains(ttnn::graph::kTensorId));

            // Check for extended tensor info
            if (params.contains(ttnn::graph::kDtype)) {
                found_tensor_with_full_info = true;

                // dtype should be a string like "DataType::BFLOAT16"
                EXPECT_TRUE(params.at(ttnn::graph::kDtype).is_string());
                std::string dtype = params.at(ttnn::graph::kDtype).get<std::string>();
                EXPECT_FALSE(dtype.empty());

                // layout should be present
                ASSERT_TRUE(params.contains(ttnn::graph::kLayout));
                EXPECT_TRUE(params.at(ttnn::graph::kLayout).is_string());

                // For device tensors, these fields should be present
                if (params.contains(ttnn::graph::kDeviceId)) {
                    EXPECT_TRUE(params.at(ttnn::graph::kDeviceId).is_string());

                    // address should also be present for device tensors
                    ASSERT_TRUE(params.contains(ttnn::graph::kAddress));
                    EXPECT_TRUE(params.at(ttnn::graph::kAddress).is_string());

                    // buffer_type should be present
                    ASSERT_TRUE(params.contains(ttnn::graph::kBufferType));
                    EXPECT_TRUE(params.at(ttnn::graph::kBufferType).is_string());

                    // memory_config should be present for allocated device tensors
                    ASSERT_TRUE(params.contains(ttnn::graph::kMemoryConfig));
                    EXPECT_TRUE(params.at(ttnn::graph::kMemoryConfig).is_string());
                }
            }
        }
    }

    EXPECT_TRUE(found_tensor_with_full_info)
        << "Expected at least one tensor node with full info (dtype, layout, etc.)";
}

INSTANTIATE_TEST_SUITE_P(
    TensorInfoTest,
    TensorInfoTest,
    ::testing::Values(
        tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH, tt::tt_metal::IGraphProcessor::RunMode::NORMAL),
    [](const testing::TestParamInfo<tt::tt_metal::IGraphProcessor::RunMode>& info) {
        switch (info.param) {
            case tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH: return "NO_DISPATCH";
            case tt::tt_metal::IGraphProcessor::RunMode::NORMAL: return "NORMAL";
            default: return "UNKNOWN";
        }
    });

// Test buffer pages capture control
TEST_F(TestScopedGraphCapture, BufferPagesControlTest) {
    // Test the enable/disable API for buffer pages
    EXPECT_FALSE(ttnn::graph::GraphProcessor::is_buffer_pages_enabled())
        << "Buffer pages should be disabled by default";

    ttnn::graph::GraphProcessor::enable_buffer_pages();
    EXPECT_TRUE(ttnn::graph::GraphProcessor::is_buffer_pages_enabled());

    ttnn::graph::GraphProcessor::disable_buffer_pages();
    EXPECT_FALSE(ttnn::graph::GraphProcessor::is_buffer_pages_enabled());
}

// Test error tracking during graph capture
TEST_F(TestScopedGraphCapture, ErrorTrackingTest) {
    tt::tt_metal::IDevice* device = device_;

    nlohmann::json trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);

        const auto tensor_spec = ttnn::TensorSpec(
            ttnn::Shape(tt::tt_metal::Array4D{1, 1, 32, 32}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                ttnn::L1_MEMORY_CONFIG));
        const auto input_tensor = tt::tt_metal::create_device_tensor(tensor_spec, device);

        // Manually track an error
        ttnn::graph::GraphProcessor::track_error("TestError", "This is a test error message", "test_operation");

        trace = capture.end_graph_capture();
    }

    // Find the error node
    bool found_error_node = false;
    for (const auto& node : trace) {
        if (node.at(ttnn::graph::kNodeType) == ttnn::graph::kNodeError) {
            found_error_node = true;
            auto& params = node.at(ttnn::graph::kParams);
            EXPECT_EQ(params.at(ttnn::graph::kErrorType).get<std::string>(), "TestError");
            EXPECT_EQ(params.at(ttnn::graph::kErrorMessage).get<std::string>(), "This is a test error message");
            EXPECT_EQ(params.at(ttnn::graph::kErrorOperation).get<std::string>(), "test_operation");
        }
    }

    EXPECT_TRUE(found_error_node) << "Expected to find an error node in the trace";
}

// Test report contains cluster_descriptor when devices are present
TEST_F(TestScopedGraphCapture, ReportContainsClusterDescriptor) {
    tt::tt_metal::IDevice* device = device_;

    nlohmann::json report;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);

        const auto tensor_spec = ttnn::TensorSpec(
            ttnn::Shape(tt::tt_metal::Array4D{1, 1, 32, 32}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                ttnn::L1_MEMORY_CONFIG));
        const auto input_tensor = tt::tt_metal::create_device_tensor(tensor_spec, device);

        report = capture.get_report();
        capture.end_graph_capture();
    }

    // Check that the report has devices
    ASSERT_TRUE(report.contains(ttnn::graph::kReportDevices));
    auto& devices = report.at(ttnn::graph::kReportDevices);
    EXPECT_GT(devices.size(), 0u) << "Expected at least one device to be captured";

    // cluster_descriptor may or may not be present depending on environment
    // but if devices are captured, the structure should be valid
    if (report.contains("cluster_descriptor")) {
        EXPECT_TRUE(report.at("cluster_descriptor").is_string());
    }
}
