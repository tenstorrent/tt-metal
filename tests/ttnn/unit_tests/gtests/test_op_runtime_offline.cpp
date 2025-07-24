// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <tt-logger/tt-logger.hpp>
#include <cstdlib>
#include <tuple>
#include <nlohmann/json.hpp>

#include "gtest/gtest.h"
#include "ttnn/tensor/tensor.hpp"
#include "umd/device/types/arch.h"

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
    // tt::tt_metal::distributed::MeshDevice* device_ = nullptr;
    // std::shared_ptr<tt::tt_metal::distributed::MeshDevice> device_holder_;
    tt::ARCH arch_ = tt::ARCH::Invalid;
    size_t num_devices_ = 0;

    void SetUp() override {
        std::srand(0);
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        num_devices_ = 1;
        // device_holder_ = ttnn::open_mesh_device(0, DEFAULT_L1_SMALL_SIZE, /* trace region size= */ 200000);
        // device_ = device_holder_.get();
    }

    void TearDown() override {}

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
    // Create the tensor JSON with the expected structure
    nlohmann::json tensor_json = {
        {"storage", {{"index", 0}, {"value", nlohmann::json::object()}}},
        {"tensor_spec",
         {{"logical_shape",
           {{"value",
             "tt::stl::json::to_json_t: Unsupported type "
             "SmallVector<unsigned int>"}}},
          {"tensor_layout",
           {{"alignment",
             {{"value",
               "tt::stl::json::to_json_t: Unsupported type "
               "SmallVector<unsigned int>"}}},
            {"dtype", dtype},
            {"memory_config", {{"buffer_type", buffer_type}, {"memory_layout", 0}}},
            {"page_config",
             {{"config",
               {{"index", 0},
                {"value", {{"tile", {{"face_shape", {16, 16}}, {"num_faces", 4}, {"tile_shape", {32, 32}}}}}}}}}}}}}}};

    nlohmann::json shape_json = dimensions;
    return nlohmann::json::array({tensor_json, shape_json});
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

    // Use the proper interface function
    auto runtime_ns = op_perf::get_runtime_from_model("ttnn::exp", tensor_json, output_layout_json);

    EXPECT_GT(runtime_ns, 0);
    log_info(tt::LogTest, "Model runtime for ttnn::exp: {} ns", runtime_ns);
}

INSTANTIATE_TEST_SUITE_P(
    GetRuntimeFromModel,
    TestExpOpRuntimeFromModel,
    ::testing::Values(TestExpOpRuntimeFromModel::m_interleaved_1_3_1024_1024_tiled));
#endif  // BUILD_MLP_OP_PERF

}  // namespace ttnn::operations::binary::test
