// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/logger.hpp"
#include "tests/tt_metal/tt_metal/unit_tests_common/common/common_fixture.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn_test_fixtures.hpp"

#include "ttnn/tensor/types.hpp"

#include "ttnn/graph_processor.hpp"
#include "third_party/json/json.hpp"

#include "ttnn/tensor/types.hpp"

#include <cstdint>
#include <string>

namespace ttnn {
namespace operations {
namespace binary {
namespace test {

namespace {

uint32_t extract_peak_memory_usage(const nlohmann::json& trace) {
    uint32_t total_cb = 0;
    uint32_t total_buffer = 0;
    uint32_t peak_memory_usage = 0;
    std::vector<std::string> current_op;

    for (size_t i = 0; i < trace.size(); ++i) {
        const auto& v = trace[i];

        if (v["name"] == "begin_function") {
            if (current_op.empty()) {
                while (++i < trace.size()) {
                    const auto& inner_v = trace[i];
                    if (inner_v["name"] == "buffer" && inner_v["params"]["type"] == "L1") {
                        total_buffer += std::stoi(inner_v["params"]["size"].get<std::string>());
                    } else if (inner_v["name"].get<std::string>().find("tensor") != std::string::npos) {
                        continue;
                    } else {
                        break;
                    }
                }
                --i;  // adjust for loop increment
            }
            current_op.push_back(v["params"]["name"]);
        } else if (v["name"] == "circular_buffer_allocate") {
            total_cb += stoi(v["params"]["size"].get<std::string>());
        } else if (v["name"] == "circular_buffer_deallocate_all") {
            total_cb = 0;
        } else if (v["name"] == "buffer_allocate" && v["params"]["type"] == "L1") {
            total_buffer += stoi(v["params"]["size"].get<std::string>());
        } else if (v["name"] == "buffer_deallocate") {
            auto connection = v["connections"][0].get<int>();
            auto buffer = trace[connection];
            if(buffer["params"]["type"] == "L1") {
                total_buffer -= stoi(buffer["params"]["size"].get<std::string>());
            }
        } else if (v["name"] == "end_function") {
            current_op.pop_back();
        }

        peak_memory_usage = std::max(peak_memory_usage, total_cb + total_buffer);
    }

    return peak_memory_usage;
}

// Returns count of intermediate and output tensors
std::pair<int, int> count_intermediate_and_output_tensors(const nlohmann::json& trace) {
    bool first_begin_found = false;
    bool last_end_found = false;

    std::unordered_set<int> intermediate_tensors;
    std::unordered_set<int> output_tensors;

    int first_begin_index = -1;
    int last_end_index = -1;

    for (int i = 0; i < trace.size(); ++i) {
        const auto& v = trace[i];
        if (v["name"] == "begin_function" && !first_begin_found) {
            first_begin_found = true;
            first_begin_index = i;
        } else if (v["name"] == "end_function") {
            last_end_found = true;
            last_end_index = i;

            if("create_device_tensor") {
                auto id = v["connections"][0].get<int>();
                intermediate_tensors.insert(id);
            }
        }
    }

    TT_ASSERT(first_begin_found);
    TT_ASSERT(last_end_found);

    for(int index : trace[last_end_index]["connections"]) {
        output_tensors.insert(index);
    }

    for(int index : output_tensors) {
        intermediate_tensors.erase(index);
    }

    // Return the counts of intermediate and output tensors
    return {static_cast<int>(intermediate_tensors.size()), static_cast<int>(output_tensors.size())};
}

std::vector<std::string> extract_calltrace(const nlohmann::json& trace){
    std::vector<std::string> op_calls;
    size_t i = 0;

    while (i < trace.size()) {
        const auto& v = trace[i];
        i++;

        if (v["name"] == "begin_function") {
            op_calls.push_back(v["params"]["name"]);
        }
    }

    return op_calls;
}
}

struct AddOpGraphTestParam {
    ttnn::Shape a_Shape;
    ttnn::Shape b_Shape;
    ttnn::MemoryConfig memory_config; //DRAM_MEMORY_CONFIG, L1_MEMORY_CONFIG, L1_BLOCK_SHARDED_MEMORY_CONFIG, L1_HEIGHT_SHARDED_MEMORY_CONFIG, L1_WIDTH_SHARDED_MEMORY_CONFIG
    std::vector<std::string> expected_calltrace;
    uint32_t expected_peak_memory_usage = 0;
    uint32_t expected_intermediate_tensors_count = 0;
    uint32_t expected_output_tensors_count = 0;
};

class AddOpGraphTestFixture : public TTNNFixtureWithDevice,
                              public testing::WithParamInterface<AddOpGraphTestParam> {};


TEST_P(AddOpGraphTestFixture, AddGraphTrace) {
    auto params = GetParam();
    const auto device_id = 0;

    {
        const auto input_tensor_a = ttnn::zeros(params.a_Shape, ttnn::bfloat16, ttnn::TILE_LAYOUT, this->getDevice(), params.memory_config);
        const auto input_tensor_b = ttnn::zeros(params.b_Shape, ttnn::bfloat16, ttnn::TILE_LAYOUT, this->getDevice(), params.memory_config);

        ttnn::GraphProcessor::begin_graph_capture(tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH);
        {
            const auto output_tensor = ttnn::add(input_tensor_a, input_tensor_b);
        }
        auto json_trace = ttnn::GraphProcessor::end_graph_capture();

        // tt::log_info("Trace: {}", json_trace.dump(4));

        EXPECT_EQ(extract_calltrace(json_trace), params.expected_calltrace);
        EXPECT_EQ(extract_peak_memory_usage(json_trace), params.expected_peak_memory_usage);

        auto [intermediate_tensors_count, output_tensors_count] = count_intermediate_and_output_tensors(json_trace);
        EXPECT_EQ(intermediate_tensors_count, params.expected_intermediate_tensors_count);
        EXPECT_EQ(output_tensors_count, params.expected_output_tensors_count);
    }
}

INSTANTIATE_TEST_SUITE_P(
    AddOpGraphTests, // Prefix for the instantiated test suite
    AddOpGraphTestFixture, // Test suite name
    ::testing::Values(
        // AddOpGraphTestParam instances for different test cases
        AddOpGraphTestParam{
            .a_Shape = ttnn::Shape(tt::tt_metal::Array4D{1, 3, 32, 32}),
            .b_Shape = ttnn::Shape(tt::tt_metal::Array4D{1, 3, 32, 32}),
            .memory_config = ttnn::L1_MEMORY_CONFIG,
            .expected_calltrace = { "ttnn::add", "ttnn::prim::binary", "Device Operation", "create_device_tensor" },
            .expected_peak_memory_usage = 30720,
            .expected_intermediate_tensors_count = 0,
            .expected_output_tensors_count = 1
        },
        AddOpGraphTestParam{
            .a_Shape = ttnn::Shape(tt::tt_metal::Array4D{4, 3, 32, 32}),
            .b_Shape = ttnn::Shape(tt::tt_metal::Array4D{1, 3, 32, 32}),
            .memory_config = ttnn::L1_MEMORY_CONFIG,
            .expected_calltrace = { "ttnn::add", "ttnn::repeat", "ttnn::prim::old_infra_device_operation", "Device Operation", "create_device_tensor", "ttnn::prim::binary", "Device Operation", "create_device_tensor"},
            .expected_peak_memory_usage = 96256,
            .expected_intermediate_tensors_count = 1,
            .expected_output_tensors_count = 1
        }
    )
);


}  // namespace test
}  // namespace binary
}  // namespace operations
}  // namespace ttnn
