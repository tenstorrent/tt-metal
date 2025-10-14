// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <tt-logger/tt-logger.hpp>
#include <cstdlib>
#include <memory>
#include <tuple>

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
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include <umd/device/types/arch.hpp>

namespace tt {
namespace tt_metal {
class IDevice;
}  // namespace tt_metal
}  // namespace tt

namespace ttnn::operations::binary::test {

// ============================================================================
// Test data
// ============================================================================

class TTNNFixtureWithTraceEnabledDevice : public ::testing::Test {
protected:
    tt::tt_metal::distributed::MeshDevice* device_ = nullptr;
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> device_holder_;
    tt::ARCH arch_ = tt::ARCH::Invalid;
    size_t num_devices_ = 0;

    void SetUp() override {
        std::srand(0);
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        num_devices_ = tt::tt_metal::GetNumAvailableDevices();
        device_holder_ = ttnn::open_mesh_device(0, DEFAULT_L1_SMALL_SIZE, /* trace region size= */ 200000);
        device_ = device_holder_.get();
    }

    void TearDown() override { ttnn::close_device(*device_); }

public:
    static const ttnn::TensorSpec m_interleaved_1_3_1024_1024_tiled;
};

const ttnn::TensorSpec TTNNFixtureWithTraceEnabledDevice::m_interleaved_1_3_1024_1024_tiled = ttnn::TensorSpec(
    ttnn::Shape({1, 3, 1024, 1024}),
    tt::tt_metal::TensorLayout(
        tt::tt_metal::DataType::BFLOAT16,
        tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
        ttnn::L1_MEMORY_CONFIG));

// ============================================================================
// Binary Eltwise Op tests
// ============================================================================

class BinaryOpTraceRuntime : public TTNNFixtureWithTraceEnabledDevice,
                             public testing::WithParamInterface<std::tuple<ttnn::TensorSpec, ttnn::TensorSpec>> {};

TEST_P(BinaryOpTraceRuntime, Add) {
    const auto& [input_spec_a, input_spec_b] = GetParam();

    {
        auto device = device_;
        auto query = ttnn::graph::query_op_runtime(ttnn::add, device, input_spec_a, input_spec_b);

        EXPECT_EQ(query.status, ttnn::graph::ExecutionStatus::Success);
        log_info(tt::LogTest, "Trace runtime: {} ns", query.runtime);
    }
}

TEST_P(BinaryOpTraceRuntime, AddChain) {
    const auto& [input_spec_a, input_spec_b] = GetParam();

    {
        auto add_chain = [](const auto& i0, const auto& i1) { return ttnn::add(i0, ttnn::add(i0, i1)); };
        auto device = device_;
        auto query = ttnn::graph::query_op_runtime(add_chain, device, input_spec_a, input_spec_b);

        EXPECT_EQ(query.status, ttnn::graph::ExecutionStatus::Success);
        log_info(tt::LogTest, "Trace runtime: {} ns", query.runtime);
    }
}

INSTANTIATE_TEST_SUITE_P(
    QueryOpRuntime,
    BinaryOpTraceRuntime,
    ::testing::Values(std::make_tuple(
        BinaryOpTraceRuntime::m_interleaved_1_3_1024_1024_tiled,
        BinaryOpTraceRuntime::m_interleaved_1_3_1024_1024_tiled)));

}  // namespace ttnn::operations::binary::test

namespace ttnn::operations::reshape::test {

class ReshapeOpTraceRuntime : public binary::test::TTNNFixtureWithTraceEnabledDevice {};

TEST_F(ReshapeOpTraceRuntime, Reshape) {
    tt::tt_metal::distributed::MeshDevice* device = device_;

    // Create input tensor spec
    const auto input_spec = ttnn::TensorSpec(
        ttnn::Shape{64, 1024},
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            ttnn::DRAM_MEMORY_CONFIG));

    // Create output tensor spec for DRAM
    const auto output_specDRAM = ttnn::TensorSpec(
        ttnn::Shape{1, 1, 256, 256},
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            ttnn::DRAM_MEMORY_CONFIG));

    // Create output tensor spec for L1
    const auto output_specL1 = ttnn::TensorSpec(
        ttnn::Shape{1, 1, 256, 256},
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            ttnn::L1_MEMORY_CONFIG));

    // hang sequence: runtime -> constraints -> runtime
    auto runQueryDRAM = ttnn::graph::query_op_runtime(
        ttnn::reshape,
        device,
        input_spec,
        output_specDRAM.logical_shape(),
        output_specDRAM.tensor_layout().get_memory_config());
    auto constraintsL1 = ttnn::graph::query_op_constraints(
        ttnn::reshape,
        device,
        input_spec,
        output_specL1.logical_shape(),
        output_specL1.tensor_layout().get_memory_config());
    auto runQueryL1 = ttnn::graph::query_op_runtime(
        ttnn::reshape,
        device,
        input_spec,
        output_specL1.logical_shape(),
        output_specL1.tensor_layout().get_memory_config());

    // Verify the operations were successful
    EXPECT_EQ(runQueryDRAM.status, ttnn::graph::ExecutionStatus::Success);
    EXPECT_EQ(constraintsL1.status, ttnn::graph::ExecutionStatus::Success);
    EXPECT_EQ(runQueryL1.status, ttnn::graph::ExecutionStatus::Success);

    // Verify we got a valid runtime measurements
    EXPECT_GT(runQueryDRAM.runtime, 0);
    EXPECT_GT(runQueryL1.runtime, 0);
}

}  // namespace ttnn::operations::reshape::test
