// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <boost/move/utility_core.hpp>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <stddef.h>
#include <stdint.h>
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
#include "ttnn/operations/normalization/softmax/softmax.hpp"
#include "ttnn/tensor/enum_types.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace tt {
namespace tt_metal {
class IDevice;
}  // namespace tt_metal
}  // namespace tt

namespace ttnn::graph::test {

struct BufferTestParam {
    ttnn::Shape shape_a;
    ttnn::Shape shape_b;
};

class BufferTestFixture
    : public TTNNFixtureWithDevice,
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
                 "ttnn::prim::old_infra_device_operation",
                 "Softmax",
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
