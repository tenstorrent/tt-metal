// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "tt_metal/common/logger.hpp"
#include "tests/tt_metal/tt_metal/unit_tests_common/common/common_fixture.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn_test_fixtures.hpp"

#include "ttnn/tensor/types.hpp"

#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"
#include "ttnn/graph/graph_operation_queries.hpp"

#include "ttnn/tensor/types.hpp"

#include <cstdint>
#include <string>

namespace ttnn {
namespace operations {
namespace binary {
namespace test {

struct AddOpGraphTestParam {
    ttnn::Shape a_Shape;
    ttnn::Shape b_Shape;
    ttnn::MemoryConfig memory_config; //DRAM_MEMORY_CONFIG, L1_MEMORY_CONFIG, L1_BLOCK_SHARDED_MEMORY_CONFIG, L1_HEIGHT_SHARDED_MEMORY_CONFIG, L1_WIDTH_SHARDED_MEMORY_CONFIG
    std::vector<std::string> expected_calltrace;
    uint32_t expected_peak_L1_memory_usage = 0;
    uint32_t expected_intermediate_tensors_count = 0;
    std::vector<graph::TensorInfo> expected_output_info;
};

class AddOpGraphTestFixture : public TTNNFixtureWithDevice,
                              public testing::WithParamInterface<std::tuple<AddOpGraphTestParam, tt::tt_metal::IGraphProcessor::RunMode>> {};


TEST_P(AddOpGraphTestFixture, AddGraphTrace) {
    auto param_combination = GetParam();
    auto params = std::get<0>(param_combination);
    auto run_mode = std::get<1>(param_combination);

    {
        const auto input_tensor_a = ttnn::zeros(params.a_Shape, ttnn::bfloat16, ttnn::TILE_LAYOUT, this->getDevice(), params.memory_config);
        const auto input_tensor_b = ttnn::zeros(params.b_Shape, ttnn::bfloat16, ttnn::TILE_LAYOUT, this->getDevice(), params.memory_config);

        auto call = [&] {
            const auto output_tensor = ttnn::add(input_tensor_a, input_tensor_b);
            return output_tensor;
        };

        auto json_trace = graph::query_trace(call);

        tt::log_info("Trace: {}", json_trace.dump(4));

        // Direct calls
        {
            EXPECT_EQ(graph::extract_calltrace(json_trace), params.expected_calltrace);
            EXPECT_EQ(graph::extract_peak_L1_memory_usage(json_trace), params.expected_peak_L1_memory_usage);
            EXPECT_EQ(graph::extract_output_tensors(json_trace).size(), 1);

            auto [intermediate_tensors_count, output_tensors_count] = graph::count_intermediate_and_output_tensors(json_trace);
            EXPECT_EQ(intermediate_tensors_count, params.expected_intermediate_tensors_count);
            EXPECT_EQ(output_tensors_count, 1);
        }

        // Query calls
        {
            auto peak_L1_memory_usage = graph::query_peak_L1_memory_usage(call);
            auto output_info = graph::query_output_info(call);

            EXPECT_EQ(peak_L1_memory_usage, params.expected_peak_L1_memory_usage);


            if(output_info.size() != params.expected_output_info.size()) {
                auto print = [](const auto& infos){
                    for (const auto& info : infos) {
                        tt::log_info("{}", info);
                    }
                };

                tt::log_info("Output info size mismatch. Expected {} but got {}", params.expected_output_info.size(), output_info.size());

                tt::log_info("Expected output info:");
                print(params.expected_output_info);

                tt::log_info("Actual output info:");
                print(output_info);
                ASSERT_TRUE(false);
            }

            for (int i = 0; i < output_info.size(); ++i) {
                EXPECT_EQ(output_info[i], params.expected_output_info[i]);
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    AddOpGraphTests, // Prefix for the instantiated test suite
    AddOpGraphTestFixture, // Test suite name
    ::testing::Combine(
        ::testing::Values(
            // AddOpGraphTestParam instances for different test cases
            AddOpGraphTestParam{
                .a_Shape = ttnn::Shape(tt::tt_metal::Array4D{1, 3, 32, 32}),
                .b_Shape = ttnn::Shape(tt::tt_metal::Array4D{1, 3, 32, 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
                .expected_calltrace = { "ttnn::add", "ttnn::prim::binary", "BinaryDeviceOperation", "tt::tt_metal::create_device_tensor" },
                .expected_peak_L1_memory_usage = 30720,
                .expected_intermediate_tensors_count = 0,
                .expected_output_info = {
                    graph::TensorInfo{
                        .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 3, 32, 32}),
                        .size = 6144,
                        .type = tt::tt_metal::BufferType::L1}
                }
            },
            AddOpGraphTestParam{
                .a_Shape = ttnn::Shape(tt::tt_metal::Array4D{4, 3, 32, 32}),
                .b_Shape = ttnn::Shape(tt::tt_metal::Array4D{1, 3, 32, 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
                .expected_calltrace = { "ttnn::add", "ttnn::repeat", "ttnn::prim::old_infra_device_operation", "RepeatDeviceOperation", "tt::tt_metal::create_device_tensor", "ttnn::prim::binary", "BinaryDeviceOperation", "tt::tt_metal::create_device_tensor"},
                .expected_peak_L1_memory_usage = 92160,
                .expected_intermediate_tensors_count = 0,
                .expected_output_info = {
                    graph::TensorInfo{
                        .shape = ttnn::Shape(tt::tt_metal::Array4D{4, 3, 32, 32}),
                        .size = 24576,
                        .type = tt::tt_metal::BufferType::L1
                    }
                },
            }
        ),
        ::testing::Values(tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH, tt::tt_metal::IGraphProcessor::RunMode::NORMAL)
    )
);


}  // namespace test
}  // namespace binary
}  // namespace operations
}  // namespace ttnn
