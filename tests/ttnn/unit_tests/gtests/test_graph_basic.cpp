// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "ttnn_test_fixtures.hpp"
#include "ttnn/device.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_consts.hpp"

#include <string>

namespace ttnn::graph::test {

struct BufferTestParam {
    ttnn::SimpleShape shape_a;
    ttnn::SimpleShape shape_b;
};

class BufferTestFixture
    : public TTNNFixtureWithDevice,
      public testing::WithParamInterface<std::tuple<BufferTestParam, tt::tt_metal::IGraphProcessor::RunMode>> {};

TEST_P(BufferTestFixture, BufferTest) {
    auto param_combination = GetParam();
    auto params = std::get<0>(param_combination);
    auto run_mode = std::get<1>(param_combination);

    tt::tt_metal::Device* device = &(this->getDevice());
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
            .shape_a = ttnn::SimpleShape(tt::tt_metal::Array4D{1, 1, 32, 32}),
            .shape_b = ttnn::SimpleShape(tt::tt_metal::Array4D{32, 1, 32, 32})}),
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
