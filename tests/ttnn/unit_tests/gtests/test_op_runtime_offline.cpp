// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <tt-logger/tt-logger.hpp>
#include <cstdlib>
#include <tuple>
#include <nlohmann/json.hpp>

#include "gtest/gtest.h"
#include <tt-metalium/host_api.hpp>
#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/shape.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/device.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/tensor/enum_types.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "umd/device/types/arch.h"
#include "tt_stl/tt_stl/small_vector.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

// Include the interface for get_runtime_from_model
#ifdef BUILD_MLP_OP_PERF
#include "interface.hpp"
#endif

namespace ttnn::operations::binary::test {

// ============================================================================
// Test data
// ============================================================================

class TTNNFixtureWithOfflineModel : public ::testing::Test {
protected:
    tt::tt_metal::distributed::MeshDevice* device_ = nullptr;
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> device_holder_;
    tt::ARCH arch_ = tt::ARCH::Invalid;
    size_t num_devices_ = 0;

    void SetUp() override {
        std::srand(0);
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        num_devices_ = 1;
        device_holder_ = ttnn::open_mesh_device(0, DEFAULT_L1_SMALL_SIZE, /* trace region size= */ 200000);
        device_ = device_holder_.get();
    }

    void TearDown() override { ttnn::close_device(*device_); }

public:
    static const ttnn::TensorSpec m_interleaved_1_3_1024_1024_tiled;
};

const ttnn::TensorSpec TTNNFixtureWithOfflineModel::m_interleaved_1_3_1024_1024_tiled = ttnn::TensorSpec(
    ttnn::Shape({1, 3, 1024, 1024}),
    tt::tt_metal::TensorLayout(
        tt::tt_metal::DataType::BFLOAT16,
        tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
        ttnn::L1_MEMORY_CONFIG));

// ============================================================================
// Unary Exp Op test using get_runtime_from_model
// ============================================================================

#ifdef BUILD_MLP_OP_PERF

// Utility function to create serialized tensor in the format expected by mlp-op-perf
// This matches the format used in the mlp-op-perf test_interface.cpp
nlohmann::json create_serialized_tensor(const std::vector<int>& dimensions, const int& dtype, const int& buffer_type) {
    nlohmann::json tensor_json = {
        {"tensor_spec",
         {{"logical_shape", dimensions},
          {"tensor_layout",
           {{"alignment", {{"value", {1, 1, 32, 32}}}},
            {"dtype", dtype},
            {"memory_config",
             {{"buffer_type", buffer_type}, {"created_with_nd_shard_spec", false}, {"memory_layout", 0}}},
            {"page_config",
             {{"config",
               {{"index", 1},
                {"value", {{"tile", {{"face_shape", {16, 16}}, {"num_faces", 4}, {"tile_shape", {32, 32}}}}}}}}}}}}}},
        {"storage", {{"index", 1}, {"value", nlohmann::json::object()}}}};

    return tensor_json;
}

class TestExpOpRuntimeFromModel : public TTNNFixtureWithOfflineModel,
                                  public testing::WithParamInterface<ttnn::TensorSpec> {};

TEST_P(TestExpOpRuntimeFromModel, ExpOpFromModel) {
    const auto& input_spec = GetParam();

    // Convert TensorSpec to the format expected by mlp-op-perf
    std::vector<int> shape_vec(input_spec.logical_shape().cbegin(), input_spec.logical_shape().cend());
    int dtype = static_cast<int>(input_spec.data_type());
    int buffer_type = static_cast<int>(input_spec.memory_config().buffer_type());

    // Create the serialized tensor using the proper format
    nlohmann::json tensor_json = create_serialized_tensor(shape_vec, dtype, buffer_type);
    nlohmann::json output_layout_json = nlohmann::json::object();
    std::cout << tensor_json.dump(4) << std::endl;

    // Use the proper interface function
    auto runtime_ns = op_perf::get_runtime_from_model("ttnn::exp", tensor_json, output_layout_json);

    EXPECT_GT(runtime_ns, 0);
    log_info(tt::LogTest, "Model runtime for ttnn::exp: {} ns", runtime_ns);
}

INSTANTIATE_TEST_SUITE_P(
    GetRuntimeFromModel,
    TestExpOpRuntimeFromModel,
    ::testing::Values(TestExpOpRuntimeFromModel::m_interleaved_1_3_1024_1024_tiled));

// ============================================================================
// Json Serialization Test
// ============================================================================

class TestTensorJsonSerialization : public TTNNFixtureWithOfflineModel,
                                    public testing::WithParamInterface<ttnn::TensorSpec> {};

TEST_P(TestTensorJsonSerialization, TensorSpecToJson) {
    const auto& input_spec = GetParam();

    // Serialize the TensorSpec to JSON using tt::stl::json::to_json
    nlohmann::json tensor_json = ttsl::json::to_json(input_spec);
    nlohmann::json arg = nlohmann::json::object();
    auto new_arg = ttsl::json::to_json(arg);

    nlohmann::json arg2 = {{"one", 1}, {"two", 2}};
    auto new_arg2 = ttsl::json::to_json(arg2);

    std::cout << arg2.dump(2) << std::endl;
    std::cout << new_arg2.dump(2) << std::endl;

    std::cout << new_arg.dump(2);

    // Print or inspect the JSON (for demonstration)
    std::cout << "Serialized TensorSpec JSON: " << tensor_json.dump(2) << std::endl;

    // Basic check: ensure the result is a JSON object or string (if unsupported)
    EXPECT_TRUE(tensor_json.is_object() || tensor_json.is_string());
}

INSTANTIATE_TEST_SUITE_P(
    TensorSpecJsonSerialization,
    TestTensorJsonSerialization,
    ::testing::Values(TestTensorJsonSerialization::m_interleaved_1_3_1024_1024_tiled));

// ============================================================================
// Unary Exp Op test using query_op_runtime
// ============================================================================

class TestExpOpQueryOpRuntime : public TTNNFixtureWithOfflineModel,
                                public testing::WithParamInterface<ttnn::TensorSpec> {};

TEST_P(TestExpOpQueryOpRuntime, ExpOpQueryOpRuntime) {
    const auto& input_spec = GetParam();

    // Prepare an empty output layout as expected by the op
    nlohmann::json output_layout = nlohmann::json::object();

    // Call the query_op_runtime interface for ttnn::exp
    auto query = ttnn::graph::query_op_runtime(ttnn::exp, device_, input_spec, output_layout);

    // Check that the query was successful and runtime is positive
    EXPECT_EQ(query.status, ttnn::graph::ExecutionStatus::Success) << "failed here" << std::endl;
    EXPECT_GT(query.runtime, 0);
    log_info(tt::LogTest, "QueryOpRuntime for ttnn::exp: {} ns", query.runtime);
}

INSTANTIATE_TEST_SUITE_P(
    QueryOpRuntimeExp,
    TestExpOpQueryOpRuntime,
    ::testing::Values(TestExpOpQueryOpRuntime::m_interleaved_1_3_1024_1024_tiled));

#endif  // BUILD_MLP_OP_PERF

}  // namespace ttnn::operations::binary::test
