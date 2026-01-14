// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <boost/move/utility_core.hpp>
#include <fmt/base.h>
#include <nlohmann/json.hpp>
#include <cstddef>
#include <tt-logger/tt-logger.hpp>
#include <cstdint>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/graph_tracking.hpp>
#include "gtest/gtest.h"
#include <tt-metalium/shape.hpp>
#include "ttnn/decorators.hpp"
#include "ttnn/graph/graph_operation_queries.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::binary::test {

struct AddOpGraphTestParam {
    ttnn::Shape a_Shape;
    ttnn::Shape b_Shape;
    ttnn::MemoryConfig memory_config;  // DRAM_MEMORY_CONFIG, L1_MEMORY_CONFIG, L1_BLOCK_SHARDED_MEMORY_CONFIG,
                                       // L1_HEIGHT_SHARDED_MEMORY_CONFIG, L1_WIDTH_SHARDED_MEMORY_CONFIG
    std::vector<std::string> expected_calltrace;
    uint32_t expected_peak_L1_memory_usage = 0;
    uint32_t expected_intermediate_tensors_count = 0;
    uint32_t expected_cb_peak_per_core = 0;
    uint32_t expected_l1_output_per_core = 0;
    uint32_t expected_l1_peak_per_core = 0;
    std::vector<graph::TensorInfo> expected_output_info;
};

class AddOpGraphTestFixture
    : public TTNNFixtureWithSuiteDevice<AddOpGraphTestFixture>,
      public testing::WithParamInterface<std::tuple<AddOpGraphTestParam, tt::tt_metal::IGraphProcessor::RunMode>> {};

TEST_P(AddOpGraphTestFixture, AddGraphTrace) {
    auto param_combination = GetParam();
    auto params = std::get<0>(param_combination);

    {
        const auto input_tensor_a =
            ttnn::zeros(params.a_Shape, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_, params.memory_config);
        const auto input_tensor_b =
            ttnn::zeros(params.b_Shape, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_, params.memory_config);

        auto call = [&] {
            constexpr tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> none{};
            const auto output_tensor = ttnn::add(
                input_tensor_a, input_tensor_b, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);
            return output_tensor;
        };

        auto json_trace = graph::query_trace(call);

        log_info(tt::LogTest, "Trace: {}", json_trace.dump(4));

        // Direct calls
        {
            EXPECT_EQ(graph::extract_calltrace(json_trace), params.expected_calltrace);
            EXPECT_EQ(graph::extract_peak_L1_memory_usage(json_trace), params.expected_peak_L1_memory_usage);
            EXPECT_EQ(graph::extract_output_tensors(json_trace).size(), 1);

            auto [intermediate_tensors_count, output_tensors_count] =
                graph::count_intermediate_and_output_tensors(json_trace);
            EXPECT_EQ(intermediate_tensors_count, params.expected_intermediate_tensors_count);
            EXPECT_EQ(output_tensors_count, 1);
        }

        // per core buffer allocation size
        {
            auto compute_with_storage_grid_size = device_->compute_with_storage_grid_size();
            size_t interleaved_storage_cores = compute_with_storage_grid_size.x * compute_with_storage_grid_size.y;

            const auto& [cb_peak_size_per_core, l1_peak_per_core, peak_memory_usage_per_core] =
                graph::extract_resource_usage_per_core(json_trace, interleaved_storage_cores);

            EXPECT_EQ(cb_peak_size_per_core, params.expected_cb_peak_per_core);
            EXPECT_EQ(l1_peak_per_core, params.expected_l1_peak_per_core);
        }

        // Query calls
        {
            auto peak_L1_memory_usage = graph::query_peak_L1_memory_usage(call);
            auto output_info = graph::query_output_info(call);

            EXPECT_EQ(peak_L1_memory_usage, params.expected_peak_L1_memory_usage);

            if (output_info.size() != params.expected_output_info.size()) {
                auto print = [](const auto& infos) {
                    for (const auto& info : infos) {
                        log_info(tt::LogTest, "{}", info);
                    }
                };

                log_info(
                    tt::LogTest,
                    "Output info size mismatch. Expected {} but got {}",
                    params.expected_output_info.size(),
                    output_info.size());

                log_info(tt::LogTest, "Expected output info:");
                print(params.expected_output_info);

                log_info(tt::LogTest, "Actual output info:");
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
    AddOpGraphTests,        // Prefix for the instantiated test suite
    AddOpGraphTestFixture,  // Test suite name
    ::testing::Combine(
        ::testing::Values(
            // AddOpGraphTestParam instances for different test cases
            AddOpGraphTestParam{
                .a_Shape = ttnn::Shape(tt::tt_metal::Array4D{1, 3, 32, 32}),
                .b_Shape = ttnn::Shape(tt::tt_metal::Array4D{1, 3, 32, 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
                .expected_calltrace = {"ttnn::add", "BinaryNgDeviceOperation", "tt::tt_metal::create_device_tensor"},
                .expected_peak_L1_memory_usage = 30720,
                .expected_intermediate_tensors_count = 0,
                .expected_cb_peak_per_core = 3 * 4096,
                .expected_l1_output_per_core = 2048,
                .expected_l1_peak_per_core = 2048,
                .expected_output_info = {graph::TensorInfo{
                    .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 3, 32, 32}),
                    .size = 6144,
                    .type = tt::tt_metal::BufferType::L1}}},
            AddOpGraphTestParam{
                .a_Shape = ttnn::Shape(tt::tt_metal::Array4D{4, 3, 32, 32}),
                .b_Shape = ttnn::Shape(tt::tt_metal::Array4D{1, 3, 32, 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
                .expected_calltrace = {"ttnn::add", "BinaryNgDeviceOperation", "tt::tt_metal::create_device_tensor"},
                .expected_peak_L1_memory_usage = 67584,
                .expected_intermediate_tensors_count = 0,
                .expected_cb_peak_per_core = 3 * 4096,
                .expected_l1_output_per_core = 2048,
                .expected_l1_peak_per_core = 2048,
                .expected_output_info = {graph::TensorInfo{
                    .shape = ttnn::Shape(tt::tt_metal::Array4D{4, 3, 32, 32}),
                    .size = 24576,
                    .type = tt::tt_metal::BufferType::L1}},
            },
            AddOpGraphTestParam{
                .a_Shape = ttnn::Shape(tt::tt_metal::Array4D{3, 1, 32 * 32, 32 * 32}),
                .b_Shape = ttnn::Shape(tt::tt_metal::Array4D{3, 1, 32 * 32, 32 * 32}),
                .memory_config =
                    tt::tt_metal::MemoryConfig{
                        tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                        tt::tt_metal::BufferType::L1,
                        tt::tt_metal::ShardSpec{
                            CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{3, 3}}}},
                            {6 * 32, 32 * 32},
                            ShardOrientation::COL_MAJOR}},
                .expected_calltrace = {"ttnn::add", "BinaryNgDeviceOperation", "tt::tt_metal::create_device_tensor"},
                .expected_peak_L1_memory_usage = 20054016,
                .expected_intermediate_tensors_count = 0,
                .expected_cb_peak_per_core = 0,
                .expected_l1_output_per_core = 2 * (3 * 32 * 32 * 32 * 32) / 16,
                .expected_l1_peak_per_core = 2 * (3 * 32 * 32 * 32 * 32) / 16,
                .expected_output_info = {graph::TensorInfo{
                    .shape = ttnn::Shape(tt::tt_metal::Array4D{3, 1, 32 * 32, 32 * 32}),
                    .size = 2 * (3 * 32 * 32 * 32 * 32),
                    .type = tt::tt_metal::BufferType::L1}}}),
        ::testing::Values(
            tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH, tt::tt_metal::IGraphProcessor::RunMode::NORMAL)),
    // Test case name generator: uid_buffer_type_memory_layout_run_mode
    [](const testing::TestParamInfo<std::tuple<AddOpGraphTestParam, tt::tt_metal::IGraphProcessor::RunMode>>& info) {
        std::stringstream ss;

        static uint32_t uid = 0;
        ss << uid++;

        const auto& param = std::get<0>(info.param);
        switch (param.memory_config.buffer_type()) {
            case tt::tt_metal::BufferType::DRAM: ss << "_DRAM"; break;
            case tt::tt_metal::BufferType::L1: ss << "_L1"; break;
            default: break;
        }

        switch (param.memory_config.memory_layout()) {
            case tt::tt_metal::TensorMemoryLayout::INTERLEAVED: ss << "_I"; break;
            case tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED: ss << "_HS"; break;
            case tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED: ss << "_WS"; break;
            case tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED: ss << "_BS"; break;
            default: break;
        }

        const auto& run = std::get<1>(info.param);
        switch (run) {
            case tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH: ss << "_NO_DISPATCH"; break;
            case tt::tt_metal::IGraphProcessor::RunMode::NORMAL: ss << "_NORMAL"; break;
            default: break;
        }

        return ss.str();
    });

}  // namespace ttnn::operations::binary::test
