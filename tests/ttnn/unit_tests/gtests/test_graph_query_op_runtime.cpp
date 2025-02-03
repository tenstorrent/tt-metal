// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include "gtest/gtest.h"
#include <tt-metalium/event.hpp>
#include <tt-metalium/program_impl.hpp>
#include "tests/tt_metal/tt_metal/common/dispatch_fixture.hpp"
#include <tt-metalium/logger.hpp>
#include "ttnn/device.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::binary::test {

// ============================================================================
// Test data
// ============================================================================

class TTNNFixtureWithTraceEnabledDevice : public TTNNFixture {
protected:
    ttnn::IDevice* device_ = nullptr;

    void SetUp() override {
        TTNNFixture::SetUp();
        device_ = &ttnn::open_device(0, DEFAULT_L1_SMALL_SIZE, /* trace region size= */ 200000);
    }

    void TearDown() override {
        ttnn::close_device(*device_);
        TTNNFixture::TearDown();
    }

    ttnn::IDevice& getDevice() { return *device_; }

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
        tt::tt_metal::IDevice* device = &getDevice();
        auto query = ttnn::graph::query_op_runtime(ttnn::add, device, input_spec_a, input_spec_b);

        EXPECT_EQ(query.status, ttnn::graph::ExecutionStatus::Success);
        tt::log_info(tt::LogTest, "Trace runtime: {} ns", query.runtime);
    }
}

TEST_P(BinaryOpTraceRuntime, AddChain) {
    const auto& [input_spec_a, input_spec_b] = GetParam();

    {
        auto add_chain = [](const auto& i0, const auto& i1) { return ttnn::add(i0, ttnn::add(i0, i1)); };
        tt::tt_metal::IDevice* device = &getDevice();
        auto query = ttnn::graph::query_op_runtime(add_chain, device, input_spec_a, input_spec_b);

        EXPECT_EQ(query.status, ttnn::graph::ExecutionStatus::Success);
        tt::log_info(tt::LogTest, "Trace runtime: {} ns", query.runtime);
    }
}

INSTANTIATE_TEST_SUITE_P(
    QueryOpRuntime,
    BinaryOpTraceRuntime,
    ::testing::Values(std::make_tuple(
        BinaryOpTraceRuntime::m_interleaved_1_3_1024_1024_tiled,
        BinaryOpTraceRuntime::m_interleaved_1_3_1024_1024_tiled)));

}  // namespace ttnn::operations::binary::test
